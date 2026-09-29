#!/usr/bin/env python3
"""Stereo SGM depth node — Phase 5, Step 2.

Turns the OAK-D Pro W stereo pair into a depth image + point cloud using OpenCV
semi-global block matching (SGBM). This deliberately mirrors the REAL drone's
depth path: the physical OAK-D Pro W computes depth on-device from its stereo
pair, so building on SGM now means we see realistic depth error (holes on
textureless surfaces, error growing ~range^2) from day one — not the perfect
per-pixel truth a Gazebo depth sensor would hand us. (Gazebo depth_gt exists only
as a validation reference; this node does NOT consume it.)

Pipeline per synced left/right pair:
  gray -> downsample -> StereoSGBM.compute -> disparity
       -> depth  Z = fx * baseline / disparity   (invalid disparity -> NaN)
       -> organized XYZ cloud in the left optical frame

Inputs (ApproximateTime-synced on the two images; camera_info cached separately):
  /cam_front/left/image_raw    (mono8)
  /cam_front/right/image_raw   (mono8)
  /cam_front/left/camera_info  (intrinsics, read once)

Outputs:
  /cam_front/depth/image       (32FC1, metres; NaN where disparity is invalid)
  /cam_front/points            (organized PointCloud2, XYZ, left optical frame)

Run (sim up + full bridge incl. stereo + spawn_landmarks for texture):
  ros2 run module4_perception stereo_depth_node
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

import message_filters


def _texture_score(img: np.ndarray, block: int) -> np.ndarray:
    """Per-pixel texture confidence: mean |horizontal gradient| over a
    block x block window. Horizontal because stereo matching searches along
    rows -- no left-right intensity change means nothing to match."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    return cv2.blur(np.abs(gx), (block, block))


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')

        # ── params ────────────────────────────────────────────────────────────
        self.declare_parameter('downsample', 0.5)          # scale factor on WxH
        self.declare_parameter('baseline_m', 0.075)        # OAK-D Pro W: 7.5 cm
        self.declare_parameter('num_disparities', 64)      # must be /16; search range
        self.declare_parameter('block_size', 7)            # SGBM matched block (odd)
        self.declare_parameter('min_range_m', 0.3)         # clip near
        self.declare_parameter('max_range_m', 20.0)        # clip far
        # Texture confidence (2026-09-29): reject disparity where the left
        # image has too little horizontal texture for matching to mean
        # anything. SGBM's smoothness term otherwise "streaks" an object's
        # disparity sideways into featureless background (sky) along image
        # rows -- a 1.5 m test box came out 6.6 m wide, 1.8 m off-centre.
        # Real stereo pipelines gate on this too (OpenCV StereoBM's
        # textureThreshold, OAK-D's on-device confidence threshold).
        # Score = block-mean |Sobel x| on the downsampled left image.
        # 0 = disabled (previous behaviour).
        self.declare_parameter('texture_threshold', 0.0)
        self.declare_parameter('publish_cloud', True)
        self.declare_parameter('left_image',  '/cam_front/left/image_raw')
        self.declare_parameter('right_image', '/cam_front/right/image_raw')
        self.declare_parameter('left_info',   '/cam_front/left/camera_info')
        self.declare_parameter('depth_topic', '/cam_front/depth/image')
        self.declare_parameter('cloud_topic', '/cam_front/points')

        g = lambda n: self.get_parameter(n).value
        self.scale = float(g('downsample'))
        self.baseline = float(g('baseline_m'))
        self.min_range = float(g('min_range_m'))
        self.max_range = float(g('max_range_m'))
        self.publish_cloud = bool(g('publish_cloud'))
        self.texture_threshold = float(g('texture_threshold'))

        # SGBM needs num_disparities divisible by 16; round up defensively.
        nd = int(g('num_disparities'))
        nd = max(16, ((nd + 15) // 16) * 16)
        bs = int(g('block_size')) | 1  # force odd
        self.block_size = bs
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=nd,
            blockSize=bs,
            P1=8 * bs * bs,             # smoothness (small disparity change)
            P2=32 * bs * bs,            # smoothness (large disparity change)
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.bridge = CvBridge()
        self.K = None            # (fx, fy, cx, cy) at DOWNSAMPLED resolution
        self._t_last_log = time.time()
        self._n_since_log = 0

        # ── I/O ───────────────────────────────────────────────────────────────
        # camera_info is static intrinsics — cache it, don't sync every frame.
        self.create_subscription(CameraInfo, g('left_info'), self._info_cb, 1)

        qs = 3  # small queue -> stale pairs drop (never build up latency)
        sub_l = message_filters.Subscriber(self, Image, g('left_image'))
        sub_r = message_filters.Subscriber(self, Image, g('right_image'))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_l, sub_r], queue_size=qs, slop=0.02)
        self.sync.registerCallback(self._pair_cb)

        self.pub_depth = self.create_publisher(Image, g('depth_topic'), 1)
        self.pub_cloud = (self.create_publisher(PointCloud2, g('cloud_topic'), 1)
                          if self.publish_cloud else None)

        self.get_logger().info(
            f'stereo_depth_node up: scale={self.scale} baseline={self.baseline}m '
            f'numDisp={nd} block={bs} range=[{self.min_range},{self.max_range}]m '
            f'texture_threshold={self.texture_threshold} cloud={self.publish_cloud}')

    # ── intrinsics ─────────────────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        if self.K is not None:
            return
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        # scale intrinsics to the downsampled image we actually match on
        s = self.scale
        self.K = (fx * s, fy * s, cx * s, cy * s)
        self.get_logger().info(
            f'intrinsics cached (scaled): fx={self.K[0]:.1f} fy={self.K[1]:.1f} '
            f'cx={self.K[2]:.1f} cy={self.K[3]:.1f}')

    # ── main callback ──────────────────────────────────────────────────────────
    def _pair_cb(self, left: Image, right: Image):
        if self.K is None:
            return  # wait for camera_info

        limg = self.bridge.imgmsg_to_cv2(left, desired_encoding='mono8')
        rimg = self.bridge.imgmsg_to_cv2(right, desired_encoding='mono8')

        if self.scale != 1.0:
            limg = cv2.resize(limg, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
            rimg = cv2.resize(rimg, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)

        # SGBM returns disparity as int16 fixed-point (actual disparity * 16).
        disp = self.matcher.compute(limg, rimg).astype(np.float32) / 16.0

        fx, fy, cx, cy = self.K
        h, w = disp.shape

        # depth: Z = fx * baseline / disparity. Invalid disparity -> NaN (honest
        # holes, not fake zeros — real stereo has no data on textureless regions).
        valid = disp > 0.0
        if self.texture_threshold > 0.0:
            valid &= _texture_score(limg, self.block_size) >= self.texture_threshold
        depth = np.full((h, w), np.nan, dtype=np.float32)
        np.divide(fx * self.baseline, disp, out=depth, where=valid)
        # clip to a sane working range; out-of-range -> NaN
        depth[(depth < self.min_range) | (depth > self.max_range)] = np.nan

        self._publish_depth(depth, left.header)
        if self.pub_cloud is not None:
            self._publish_cloud(depth, left.header, fx, fy, cx, cy)

        self._tick()

    # ── publishers ─────────────────────────────────────────────────────────────
    def _publish_depth(self, depth, header):
        msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        msg.header = header  # keep the left image's stamp + frame_id
        self.pub_depth.publish(msg)

    def _publish_cloud(self, depth, header, fx, fy, cx, cy):
        h, w = depth.shape
        # back-project every pixel into the LEFT OPTICAL frame (x right, y down,
        # z forward — the ROS optical convention).
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        z = depth
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        xyz = np.stack((x, y, z), axis=-1).astype(np.float32)  # (h, w, 3), NaN kept

        msg = PointCloud2()
        msg.header = header
        msg.height = h            # organized cloud (row/col structure preserved)
        msg.width = w
        msg.is_bigendian = False
        msg.is_dense = False      # contains NaNs
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 12 * w
        msg.data = xyz.tobytes()
        self.pub_cloud.publish(msg)

    def _tick(self):
        self._n_since_log += 1
        now = time.time()
        dt = now - self._t_last_log
        if dt >= 1.0:
            self.get_logger().info(f'depth @ {self._n_since_log / dt:.1f} Hz')
            self._t_last_log = now
            self._n_since_log = 0


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
