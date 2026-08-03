#!/usr/bin/env python3
"""Semantic fusion node — Phase 5, Step 6.

Associates YOLO 2D detections (/perception/detections, what + confidence) with
geometric 3D obstacle clusters (/perception/obstacles, where + range) to produce
labeled 3D obstacles. This is the "AI x geometry, cross-checking" layer from
docs/phase5_perception_plan.md — geometry is the deterministic safety net and is
NEVER gated by AI: every geometric cluster is still published, semantic match or
not. A YOLO miss cannot make an obstacle disappear; a YOLO hallucination cannot
conjure one that geometry didn't already see.

Association: project each 3D cluster centroid (left-optical frame, from
obstacle_extractor_node) into the RGB image using cached /cam_front/rgb/camera_info
intrinsics, then pick the detection whose 2D box contains that projected point
(closest box center wins on ties/overlap). No cross-camera extrinsic calibration
exists yet (RGB and left-stereo are co-located on the same rigid mount but not
extrinsically calibrated) -- so this assumes the two optical centers are
approximately coincident, which is fine for a coarse "what is this cluster"
label, not for sub-pixel accuracy.

Inputs:
  /perception/obstacles     (visualization_msgs/MarkerArray, left-optical frame)
  /perception/detections    (vision_msgs/Detection2DArray, RGB image space)
  /cam_front/rgb/camera_info
Output:
  /perception/obstacles_labeled  (visualization_msgs/MarkerArray)
    same boxes as /perception/obstacles, plus per-marker text (class_id conf) via
    m.text, and colored by match state (labeled=blue-ish tint kept, unmatched
    keeps the original range coloring) so RViz shows fusion state directly.

Run (stereo_depth_node + obstacle_extractor_node + yolo_detector_node up):
  ros2 run module4_perception fusion_node
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

_DET_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.declare_parameter('obstacles_topic', '/perception/obstacles')
        self.declare_parameter('detections_topic', '/perception/detections')
        self.declare_parameter('rgb_info_topic', '/cam_front/rgb/camera_info')
        self.declare_parameter('output_topic', '/perception/obstacles_labeled')
        # detections older than this vs the obstacles frame are not used to label it
        self.declare_parameter('detection_max_age_s', 0.5)

        g = lambda n: self.get_parameter(n).value
        self.max_age = float(g('detection_max_age_s'))

        self.K = None  # (fx, fy, cx, cy) for the RGB camera
        self._latest_dets = None       # Detection2DArray
        self._latest_dets_stamp = 0.0  # wall time of arrival

        self.create_subscription(CameraInfo, g('rgb_info_topic'), self._info_cb, 1)
        self.create_subscription(Detection2DArray, g('detections_topic'),
                                 self._det_cb, _DET_QOS)
        self.create_subscription(MarkerArray, g('obstacles_topic'), self._obs_cb, 1)
        self.pub = self.create_publisher(MarkerArray, g('output_topic'), 1)

        self._t_last_log = time.time()
        self._n_since_log = 0
        self._n_labeled_since_log = 0

        self.get_logger().info('fusion_node up: waiting for rgb camera_info + detections + obstacles')

    def _info_cb(self, msg: CameraInfo):
        if self.K is not None:
            return
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        self.get_logger().info(f'rgb intrinsics cached: fx={self.K[0]:.1f} fy={self.K[1]:.1f} '
                               f'cx={self.K[2]:.1f} cy={self.K[3]:.1f}')

    def _det_cb(self, msg: Detection2DArray):
        self._latest_dets = msg
        self._latest_dets_stamp = time.time()

    def _project(self, x, y, z):
        """Left-optical-frame point (x right, y down, z forward) -> RGB pixel (u,v)."""
        fx, fy, cx, cy = self.K
        if z <= 0:
            return None
        u = fx * x / z + cx
        v = fy * y / z + cy
        return u, v

    def _match(self, u, v):
        """Return (class_id, score) of the detection box containing (u,v), or None.
        Ties broken by whichever box center is closest to (u,v)."""
        if self._latest_dets is None:
            return None
        if (time.time() - self._latest_dets_stamp) > self.max_age:
            return None
        best = None
        best_d2 = None
        for d in self._latest_dets.detections:
            bx, by = d.bbox.center.position.x, d.bbox.center.position.y
            hw, hh = d.bbox.size_x / 2.0, d.bbox.size_y / 2.0
            if not (bx - hw <= u <= bx + hw and by - hh <= v <= by + hh):
                continue
            if not d.results:
                continue
            d2 = (u - bx) ** 2 + (v - by) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                hyp = d.results[0].hypothesis
                best = (hyp.class_id, float(hyp.score))
        return best

    def _obs_cb(self, msg: MarkerArray):
        out = MarkerArray()
        n_labeled = 0
        n_boxes = 0
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                out.markers.append(m)
                continue
            n_boxes += 1
            label = None
            if self.K is not None:
                proj = self._project(m.pose.position.x, m.pose.position.y,
                                     m.pose.position.z)
                if proj is not None:
                    label = self._match(*proj)

            box = Marker()
            box.header = m.header
            box.ns = 'obstacles_labeled'
            box.id = m.id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose = m.pose
            box.scale = m.scale
            if label is not None:
                n_labeled += 1
                box.color.r, box.color.g, box.color.b, box.color.a = 0.2, 0.4, 1.0, 0.6
            else:
                box.color = m.color
            out.markers.append(box)

            if label is not None:
                cls_id, score = label
                txt = Marker()
                txt.header = m.header
                txt.ns = 'obstacles_labeled_text'
                txt.id = m.id
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose = m.pose
                txt.pose.position.z = m.pose.position.z - (m.scale.z / 2.0) - 0.1
                txt.scale.z = 0.3
                txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
                txt.text = f'{cls_id} {score:.2f}'
                out.markers.append(txt)

        self.pub.publish(out)
        self._tick(n_boxes, n_labeled)

    def _tick(self, n_boxes, n_labeled):
        self._n_since_log += n_boxes
        self._n_labeled_since_log += n_labeled
        now = time.time()
        dt = now - self._t_last_log
        if dt >= 1.0:
            self.get_logger().info(
                f'fusion: {self._n_since_log} boxes, {self._n_labeled_since_log} labeled '
                f'in last {dt:.1f}s')
            self._t_last_log = now
            self._n_since_log = 0
            self._n_labeled_since_log = 0


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
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
