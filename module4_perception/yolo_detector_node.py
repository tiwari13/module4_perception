#!/usr/bin/env python3
"""YOLO detector node — Phase 5 learned-perception layer.

Runs a YOLO11 object detector on the RGB camera via ONNX Runtime and publishes
standard vision_msgs detections. This is the AI half of the two-layer perception
architecture (see docs/phase5_perception_plan.md): the classical stereo-geometry
layer is the deterministic safety net; THIS adds learned semantics (what an
obstacle IS + how confident). A downstream fusion node combines these detections
with the stereo depth to produce 3D semantic obstacles.

Runtime = ONNX Runtime deliberately (NOT PyTorch): it's the SAME engine that runs
on the flight computer (Jetson via TensorRT). Train in PyTorch, export to ONNX,
deploy through ONNX Runtime — the sim node and the flight node are the same code.
The model is a CONFIG SWAP (`model_path`): yolo11n now, segmentation / a custom
drone-hazard model (wires, branches, other drones — beyond stock YOLO's 80 COCO
classes) later, with no code change.

Inputs:
  /cam_front/rgb/image_raw     (rgb8)
Outputs:
  /perception/detections       (vision_msgs/Detection2DArray)

Requires ~/drone_venv python (onnxruntime-gpu) + torch CUDA libs on LD_LIBRARY_PATH
for the GPU provider — see reference-onnx-yolo-env. Falls back to CPU otherwise.

  ros2 run module4_perception yolo_detector_node \
       --ros-args -p model_path:=/path/to/yolo11n.onnx
"""

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)

import onnxruntime as ort

# COCO-80 class names (stock YOLO11). A custom model would ship its own list.
COCO = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


def _letterbox(img, new=640):
    """Resize keeping aspect ratio, pad to new x new. Returns (img, scale, padx, pady)."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), 114, dtype=np.uint8)
    padx, pady = (new - nw) // 2, (new - nh) // 2
    canvas[pady:pady + nh, padx:padx + nw] = resized
    return canvas, r, padx, pady


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # ── params ────────────────────────────────────────────────────────────
        default_model = os.path.expanduser('~/ros2_ws/models/yolo11n.onnx')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('image_topic', '/cam_front/rgb/image_raw')
        self.declare_parameter('detections_topic', '/perception/detections')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf_thresh', 0.25)
        self.declare_parameter('iou_thresh', 0.45)
        # comma-separated ORT providers; CUDA first, CPU fallback. On Jetson: Tensorrt.
        self.declare_parameter('providers',
                               'CUDAExecutionProvider,CPUExecutionProvider')

        g = lambda n: self.get_parameter(n).value
        self.imgsz = int(g('imgsz'))
        self.conf_thresh = float(g('conf_thresh'))
        self.iou_thresh = float(g('iou_thresh'))
        model_path = str(g('model_path'))
        providers = [p.strip() for p in str(g('providers')).split(',') if p.strip()]

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f'YOLO ONNX model not found: {model_path} — export it with '
                "YOLO('yolo11n.pt').export(format='onnx') and set model_path.")

        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.inp_name = self.sess.get_inputs()[0].name
        active = self.sess.get_providers()[0]

        self.bridge = CvBridge()
        self._t_last_log = time.time()
        self._n_since_log = 0

        self.pub = self.create_publisher(Detection2DArray, g('detections_topic'), 5)
        self.create_subscription(Image, g('image_topic'), self._cb, 1)

        self.get_logger().info(
            f'yolo_detector up: model={os.path.basename(model_path)} '
            f'provider={active} imgsz={self.imgsz} conf={self.conf_thresh}')
        if active == 'CPUExecutionProvider' and 'CUDAExecutionProvider' in providers:
            self.get_logger().warn(
                'CUDA provider unavailable -> running on CPU. Put torch CUDA libs on '
                'LD_LIBRARY_PATH (see reference-onnx-yolo-env) for GPU inference.')

    def _cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h0, w0 = img.shape[:2]

        lb, r, padx, pady = _letterbox(img, self.imgsz)
        blob = lb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW

        out = self.sess.run(None, {self.inp_name: blob})[0]  # (1,84,8400)
        boxes, scores, classes = self._decode(out[0], r, padx, pady, w0, h0)

        self._publish(boxes, scores, classes, msg.header)
        self._tick(len(boxes))

    def _decode(self, pred, r, padx, pady, w0, h0):
        """pred: (84, 8400) -> filtered (boxes xyxy, scores, class ids) in orig coords."""
        pred = pred.T                          # (8400, 84)
        xywh = pred[:, :4]
        cls_scores = pred[:, 4:]               # (8400, 80)
        cls = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(cls_scores.shape[0]), cls]

        keep = conf >= self.conf_thresh
        xywh, conf, cls = xywh[keep], conf[keep], cls[keep]
        if xywh.shape[0] == 0:
            return [], [], []

        # xywh (center) in letterboxed 640-space -> xyxy in original image
        cx, cy, ww, hh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = (cx - ww / 2 - padx) / r
        y1 = (cy - hh / 2 - pady) / r
        x2 = (cx + ww / 2 - padx) / r
        y2 = (cy + hh / 2 - pady) / r
        x1 = np.clip(x1, 0, w0); x2 = np.clip(x2, 0, w0)
        y1 = np.clip(y1, 0, h0); y2 = np.clip(y2, 0, h0)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS (OpenCV expects xywh int boxes)
        nms_boxes = [[int(a), int(b), int(c - a), int(d - b)]
                     for a, b, c, d in boxes_xyxy]
        idxs = cv2.dnn.NMSBoxes(nms_boxes, conf.tolist(),
                                self.conf_thresh, self.iou_thresh)
        if len(idxs) == 0:
            return [], [], []
        idxs = np.array(idxs).reshape(-1)
        return boxes_xyxy[idxs], conf[idxs], cls[idxs]

    def _publish(self, boxes, scores, classes, header):
        arr = Detection2DArray()
        arr.header = header
        for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, classes):
            d = Detection2D()
            d.header = header
            d.bbox.center.position.x = float((x1 + x2) / 2)
            d.bbox.center.position.y = float((y1 + y2) / 2)
            d.bbox.size_x = float(x2 - x1)
            d.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            cid = int(cl)
            hyp.hypothesis.class_id = COCO[cid] if cid < len(COCO) else str(cid)
            hyp.hypothesis.score = float(sc)
            d.results.append(hyp)
            arr.detections.append(d)
        self.pub.publish(arr)

    def _tick(self, n):
        self._n_since_log += 1
        now = time.time()
        dt = now - self._t_last_log
        if dt >= 1.0:
            self.get_logger().info(
                f'detections @ {self._n_since_log / dt:.1f} Hz ({n} objs)')
            self._t_last_log = now
            self._n_since_log = 0


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
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
