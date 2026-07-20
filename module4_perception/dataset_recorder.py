#!/usr/bin/env python3
"""Auto-labeled dataset recorder — Phase 5 custom-model pipeline, Stage 1.

Captures (left mono image, obstacle mask) training pairs using the sim's
GROUND-TRUTH depth sensor as a free pixel-perfect labeler: Gazebo knows the
true depth of every pixel, so `obstacle = valid depth < range AND not ground`
is an exact segmentation label with ZERO human annotation. This is the
sim-as-labeled-data-goldmine strategy (docs/phase5_perception_plan.md): fly
the sim, harvest a dataset, train a custom class-agnostic obstacle perceiver,
deploy it through the same ONNX path as yolo_detector_node.

The pair is (LEFT MONO, mask) — NOT RGB — because depth_gt is co-located and
pixel-aligned with cam_left (same pose/FOV/resolution in the SDF). The RGB
camera is a different sensor (center-mounted, 95 deg vs 120 deg FOV): its
pixels don't correspond to depth_gt's, and on the real OAK-D obstacle
perception runs on the wide-FOV global-shutter mono pair anyway, so training
on the left image matches deployment.

Masking, per synchronized (mono, depth) frame:
  valid    = finite depth in (near, range_thresh)
  ground   = gravity-aligned height cut, like obstacle_extractor_node:
             rotate per-pixel rays to NED via PX4 attitude, then
             - if local position is fresh: ground = points within
               ground_margin of the known floor (z_ned ~ -drone_z), or
             - attitude only: floor = low height percentile (relative cut).
             If neither is fresh, no ground removal (warned; range-only mask).
  obstacle = valid & ~ground

Saved layout under dataset_dir/ :
  images/000042.png    left mono, native 800x600
  masks/000042.png     uint8 {0, 255}, pixel-aligned with the image
  previews/000042.jpg  mono + mask overlay in red — for EYEBALLING label
                       quality before training on it
  meta.jsonl           one JSON line per frame (stamp, pose, mask stats)

Diversity throttle: a frame is saved only if the drone moved >= min_move_m or
yawed >= min_yaw_deg since the last save (and at most save_rate_hz) — so the
dataset is varied viewpoints, not 1000 copies of the parked view.

Batch job, not a service: exits by itself after max_frames saves.

  ros2 run module4_perception dataset_recorder \
       --ros-args -p dataset_dir:=$HOME/ros2_ws/datasets/obstacles_v1
"""

import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Optical (x right, y down, z forward) -> body FRD (x fwd, y right, z down),
# same convention as obstacle_extractor_node:
#   [bx]   [0 0 1][ox]
#   [by] = [1 0 0][oy]
#   [bz]   [0 1 0][oz]
R_BODY_CAM = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])


def _quat_to_R(w, x, y, z):
    """Hamilton quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = np.array([w, x, y, z]) / np.sqrt(n)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class DatasetRecorder(Node):
    def __init__(self):
        super().__init__('dataset_recorder')

        # ── params ────────────────────────────────────────────────────────────
        self.declare_parameter('dataset_dir',
                               os.path.expanduser('~/ros2_ws/datasets/obstacles_v1'))
        self.declare_parameter('image_topic', '/cam_front/left/image_raw')
        self.declare_parameter('depth_topic', '/cam_front/depth_gt/image')
        self.declare_parameter('info_topic', '/cam_front/depth_gt/camera_info')
        self.declare_parameter('attitude_topic', '/fmu/out/vehicle_attitude')
        self.declare_parameter('position_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('range_thresh', 18.0)   # m; < depth_gt far clip (20)
        self.declare_parameter('min_range', 0.15)      # m; > near clip noise
        self.declare_parameter('ground_margin', 0.30)  # m above floor still = ground
        self.declare_parameter('floor_percentile', 5.0)
        self.declare_parameter('px4_timeout_s', 1.0)   # attitude/position freshness
        self.declare_parameter('max_frames', 300)      # exit after this many saves
        self.declare_parameter('save_rate_hz', 2.0)
        self.declare_parameter('min_move_m', 0.20)     # OR min_yaw_deg to save again
        self.declare_parameter('min_yaw_deg', 5.0)
        self.declare_parameter('save_preview', True)

        g = lambda n: self.get_parameter(n).value
        self.root = os.path.expanduser(str(g('dataset_dir')))
        self.range_thresh = float(g('range_thresh'))
        self.min_range = float(g('min_range'))
        self.ground_margin = float(g('ground_margin'))
        self.floor_pct = float(g('floor_percentile'))
        self.px4_timeout = float(g('px4_timeout_s'))
        self.max_frames = int(g('max_frames'))
        self.min_save_dt = 1.0 / max(float(g('save_rate_hz')), 1e-3)
        self.min_move = float(g('min_move_m'))
        self.min_yaw = math.radians(float(g('min_yaw_deg')))
        self.save_preview = bool(g('save_preview'))

        subdirs = ['images', 'masks'] + (['previews'] if self.save_preview else [])
        for sub in subdirs:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
        self.meta = open(os.path.join(self.root, 'meta.jsonl'), 'a')

        self.bridge = CvBridge()
        self.rays = None            # (H,W,2) per-pixel ((u-cx)/fx, (v-cy)/fy)
        self.R_world_cam = None     # NED <- optical
        self._att_stamp = 0.0
        self.pos = None             # (x, y, z) NED
        self.yaw = None
        self._pos_stamp = 0.0
        self._last_save_t = 0.0
        self._last_save_pose = None  # (x, y, z, yaw)
        self.n_saved = self._next_index()
        self._warned_no_att = False

        self.create_subscription(CameraInfo, str(g('info_topic')),
                                 self._info_cb, qos_profile_sensor_data)
        self.create_subscription(VehicleAttitude, str(g('attitude_topic')),
                                 self._att_cb, qos_profile_sensor_data)
        self.create_subscription(VehicleLocalPosition, str(g('position_topic')),
                                 self._pos_cb, qos_profile_sensor_data)

        self.sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, str(g('image_topic'))),
             Subscriber(self, Image, str(g('depth_topic')))],
            queue_size=5, slop=0.05)
        self.sync.registerCallback(self._pair_cb)

        self.get_logger().info(
            f'recording -> {self.root} (resuming at frame {self.n_saved}); '
            f'target {self.max_frames} frames, obstacle = depth < '
            f'{self.range_thresh}m minus ground (margin {self.ground_margin}m)')

    # ── state callbacks ────────────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        if self.rays is not None:
            return
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        u = (np.arange(msg.width, dtype=np.float32) - cx) / fx
        v = (np.arange(msg.height, dtype=np.float32) - cy) / fy
        uu, vv = np.meshgrid(u, v)
        self.rays = np.stack([uu, vv], axis=-1)
        self.get_logger().info(
            f'intrinsics locked: {msg.width}x{msg.height} fx={fx:.1f} fy={fy:.1f}')

    def _att_cb(self, msg: VehicleAttitude):
        w, x, y, z = (float(msg.q[0]), float(msg.q[1]),
                      float(msg.q[2]), float(msg.q[3]))
        self.R_world_cam = _quat_to_R(w, x, y, z) @ R_BODY_CAM
        self.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self._att_stamp = time.time()

    def _pos_cb(self, msg: VehicleLocalPosition):
        self.pos = (float(msg.x), float(msg.y), float(msg.z))
        self._pos_stamp = time.time()

    # ── recording ─────────────────────────────────────────────────────────────
    def _pair_cb(self, img_msg: Image, depth_msg: Image):
        if self.rays is None or self.n_saved >= self.max_frames:
            return
        now = time.time()
        if now - self._last_save_t < self.min_save_dt:
            return
        if not self._moved_enough():
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        mono = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
        if depth.shape != mono.shape:
            self.get_logger().error(
                f'image/depth size mismatch {mono.shape} vs {depth.shape} — '
                'depth_gt must stay co-located with cam_left in the SDF')
            return

        valid = np.isfinite(depth) & (depth > self.min_range) \
            & (depth < self.range_thresh)
        mask = valid & ~self._ground_mask(depth, valid)

        self._save(mono, mask, img_msg, ground_removed=self._att_fresh())
        self._last_save_t = now
        if self.pos is not None and self.yaw is not None:
            self._last_save_pose = (*self.pos, self.yaw)
        if self.n_saved >= self.max_frames:
            self.get_logger().info(
                f'target reached: {self.n_saved} frames in {self.root} — done, '
                'shutting down')
            raise SystemExit(0)

    def _att_fresh(self):
        return (self.R_world_cam is not None
                and time.time() - self._att_stamp < self.px4_timeout)

    def _ground_mask(self, depth, valid):
        """Boolean 'this pixel is floor' via a gravity-aligned height cut."""
        if not self._att_fresh():
            if not self._warned_no_att:
                self.get_logger().warn(
                    'no fresh PX4 attitude -> NO ground removal (mask is '
                    'range-only). Floor pixels will pollute labels; start PX4 '
                    'or ignore if the scene has no visible floor.')
                self._warned_no_att = True
            return np.zeros(depth.shape, dtype=bool)

        d = np.where(valid, depth, 0.0).astype(np.float32)
        pts = np.empty((*depth.shape, 3), dtype=np.float32)  # optical frame
        pts[..., 0] = self.rays[..., 0] * d
        pts[..., 1] = self.rays[..., 1] * d
        pts[..., 2] = d
        z_ned = pts @ self.R_world_cam[2].astype(np.float32)  # down, rel. camera

        if self.pos is not None and time.time() - self._pos_stamp < self.px4_timeout:
            # absolute cut: floor sits at z_ned ~ -drone_z below the camera
            floor = -self.pos[2]
        else:
            # relative cut: call the lowest percentile of valid points the floor
            vals = z_ned[valid]
            if vals.size == 0:
                return np.zeros(depth.shape, dtype=bool)
            floor = np.percentile(vals, 100.0 - self.floor_pct)
        return valid & (z_ned > floor - self.ground_margin)

    def _moved_enough(self):
        if self._last_save_pose is None:
            return True
        if self.pos is None or self.yaw is None:
            return True  # no odometry -> rate throttle only
        x0, y0, z0, yaw0 = self._last_save_pose
        dp = math.dist(self.pos, (x0, y0, z0))
        dyaw = abs(math.atan2(math.sin(self.yaw - yaw0),
                              math.cos(self.yaw - yaw0)))
        return dp >= self.min_move or dyaw >= self.min_yaw

    def _next_index(self):
        imgs = os.path.join(self.root, 'images')
        existing = [int(f[:6]) for f in os.listdir(imgs)
                    if f[:6].isdigit()] if os.path.isdir(imgs) else []
        return max(existing) + 1 if existing else 0

    def _save(self, mono, mask, img_msg, ground_removed):
        name = f'{self.n_saved:06d}'
        mask_u8 = (mask.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(self.root, 'images', name + '.png'), mono)
        cv2.imwrite(os.path.join(self.root, 'masks', name + '.png'), mask_u8)
        if self.save_preview:
            over = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            over[mask] = (0.4 * over[mask] + 0.6 *
                          np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(os.path.join(self.root, 'previews', name + '.jpg'), over)

        frac = float(mask.mean())
        rec = {'frame': self.n_saved,
               'stamp': img_msg.header.stamp.sec
               + img_msg.header.stamp.nanosec * 1e-9,
               'pos': list(self.pos) if self.pos else None,
               'yaw': self.yaw,
               'obstacle_frac': round(frac, 4),
               'ground_removed': ground_removed}
        self.meta.write(json.dumps(rec) + '\n')
        self.meta.flush()
        self.n_saved += 1
        if self.n_saved % 25 == 0 or self.n_saved <= 3:
            self.get_logger().info(
                f'saved {self.n_saved}/{self.max_frames} '
                f'(obstacle {100 * frac:.1f}% of frame)')


def main(args=None):
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.meta.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
