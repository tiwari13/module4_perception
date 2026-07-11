#!/usr/bin/env python3
"""Obstacle extractor — Phase 5, Steps 3-5.

Turns the stereo SGM point cloud (/cam_front/points) into discrete 3D obstacles
(/perception/obstacles) via: clean -> voxel downsample -> ground removal ->
DBSCAN clustering -> per-cluster axis-aligned box.

Ground removal is GRAVITY-ALIGNED: it rotates the cloud into the drone's NED
frame using PX4 attitude (/fmu/out/vehicle_attitude), then removes points within
a margin of the floor height. This is robust to the camera pitching down (in the
raw optical frame the floor is a big tilted plane that a single RANSAC plane fit
can't separate from the pillars standing on it -> a giant floor-box swallowing
everything). If attitude is unavailable/stale it FALLS BACK to optical-frame
RANSAC. Cluster boxes are still emitted in the optical frame (markers' frame_id).

Deliberately dependency-light: NumPy + scikit-learn only (no PCL/Open3D), which
are already on the Jazzy image. The SGM cloud is ORGANIZED (h x w with NaN holes)
but we treat it as an unorganized (N,3) set after dropping NaNs — simplest and
robust.

Frames: markers are published in the CLOUD's own frame_id (the left optical
frame). We do NOT transform to body/world here — that needs the camera->base_link
TF (Phase 2 T_base_link_unit), which isn't published live yet. Emitting in the
optical frame keeps this node self-contained and verifiable now; the TF hop is a
clean follow-up when wiring into the navigator (Step 6+).

Output: visualization_msgs/MarkerArray — one CUBE per cluster (pose=centroid,
scale=AABB), a leading DELETEALL so stale boxes vanish. This is a pragmatic,
RViz-visible contract; graduate to a custom ObstacleArray msg when the planner
interface is built (Phase 6/7).

Run (sim + bridge + spawn_landmarks + stereo_depth_node up):
  ros2 run module4_perception obstacle_extractor_node
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from sklearn.cluster import DBSCAN

# PX4 /fmu/out/* are best-effort.
_PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=1)

# Fixed camera-optical -> body-FRD rotation for the forward OAK-D.
# Optical frame is x-right, y-down, z-forward; body FRD is x-forward, y-right,
# z-down. So: body_x = opt_z, body_y = opt_x, body_z = opt_y.
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


def _read_xyz(cloud: PointCloud2) -> np.ndarray:
    """Extract an (N,3) float32 array of finite XYZ from a PointCloud2.

    Assumes x,y,z are the first three FLOAT32 fields at offsets 0/4/8 (as the SGM
    node publishes). Reads the raw buffer directly for speed rather than the slow
    point-by-point iterator.
    """
    # locate x/y/z offsets defensively (SGM uses 0/4/8 but don't hard-assume)
    off = {f.name: f.offset for f in cloud.fields}
    if not all(k in off for k in ('x', 'y', 'z')):
        return np.empty((0, 3), np.float32)
    step = cloud.point_step
    n = cloud.width * cloud.height
    buf = np.frombuffer(cloud.data, dtype=np.uint8).reshape(n, step)
    def col(name):
        o = off[name]
        return buf[:, o:o + 4].copy().view(np.float32).reshape(-1)
    xyz = np.stack((col('x'), col('y'), col('z')), axis=-1)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def _voxel_downsample(xyz: np.ndarray, size: float) -> np.ndarray:
    """Keep one point per occupied voxel (NumPy, no PCL). Returns voxel-representative
    points (the first point seen per cell — cheap and adequate for clustering)."""
    if xyz.shape[0] == 0 or size <= 0:
        return xyz
    keys = np.floor(xyz / size).astype(np.int64)
    # unique rows -> indices of first occurrence
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def _ransac_ground(xyz: np.ndarray, thresh: float, iters: int,
                   up_axis: int, min_vertical: float, rng: np.random.Generator):
    """Return a boolean mask of GROUND (plane) inliers.

    Hand-rolled RANSAC plane fit. Accepts a plane only if its normal is mostly
    aligned with the camera 'up' axis (|n[up_axis]| >= min_vertical) so a big wall
    (vertical plane) is NOT mistaken for the floor. If no vertical plane is found,
    returns an all-False mask (remove nothing) rather than deleting a wall.
    """
    n = xyz.shape[0]
    if n < 3:
        return np.zeros(n, bool)
    best_inliers = np.zeros(n, bool)
    best_count = 0
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm = nrm / ln
        if abs(nrm[up_axis]) < min_vertical:
            continue  # not a near-horizontal plane -> not the ground
        d = np.abs((xyz - p0) @ nrm)
        inl = d < thresh
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inliers = c, inl
    return best_inliers


class ObstacleExtractorNode(Node):
    def __init__(self):
        super().__init__('obstacle_extractor_node')

        # ── params ────────────────────────────────────────────────────────────
        self.declare_parameter('cloud_topic', '/cam_front/points')
        self.declare_parameter('markers_topic', '/perception/obstacles')
        self.declare_parameter('voxel_size', 0.15)        # m; downsample grid
        self.declare_parameter('max_range', 20.0)         # m; ignore far points
        self.declare_parameter('ground_thresh', 0.10)     # m; RANSAC inlier band
        self.declare_parameter('ground_iters', 60)        # RANSAC iterations
        self.declare_parameter('ground_min_vertical', 0.85)  # |normal.up| gate
        self.declare_parameter('remove_ground', True)
        # gravity-aligned ground removal (preferred over RANSAC when attitude is live)
        self.declare_parameter('attitude_topic', '/fmu/out/vehicle_attitude')
        self.declare_parameter('ground_margin', 0.30)     # m above floor still = ground
        self.declare_parameter('floor_percentile', 5.0)   # low %ile of height = floor
        self.declare_parameter('attitude_timeout_s', 1.0) # older -> fall back to RANSAC
        self.declare_parameter('cluster_eps', 0.5)        # m; DBSCAN neighborhood
        self.declare_parameter('cluster_min_samples', 10)
        self.declare_parameter('min_cluster_points', 15)  # drop tiny clusters
        # optical frame: x right, y DOWN, z forward -> 'up' is the y axis (index 1)
        self.declare_parameter('up_axis', 1)

        g = lambda n: self.get_parameter(n).value
        self.voxel = float(g('voxel_size'))
        self.max_range = float(g('max_range'))
        self.gthresh = float(g('ground_thresh'))
        self.giters = int(g('ground_iters'))
        self.gminvert = float(g('ground_min_vertical'))
        self.remove_ground = bool(g('remove_ground'))
        self.eps = float(g('cluster_eps'))
        self.min_samples = int(g('cluster_min_samples'))
        self.min_cluster_pts = int(g('min_cluster_points'))
        self.up_axis = int(g('up_axis'))
        self.ground_margin = float(g('ground_margin'))
        self.floor_pct = float(g('floor_percentile'))
        self.att_timeout = float(g('attitude_timeout_s'))

        self.rng = np.random.default_rng(0)
        self._t_last_log = time.time()
        self._n_since_log = 0
        self._used_gravity = False   # which ground method ran (for the Hz log)

        # attitude: cache the latest world<-camera rotation for gravity alignment
        self.R_world_cam = None
        self._att_stamp = 0.0
        try:
            from px4_msgs.msg import VehicleAttitude
            self.create_subscription(VehicleAttitude, g('attitude_topic'),
                                     self._att_cb, _PX4_QOS)
        except Exception as e:
            self.get_logger().warn(f'no px4_msgs attitude ({e}); RANSAC fallback only')

        self.sub = self.create_subscription(
            PointCloud2, g('cloud_topic'), self._cb, 1)
        self.pub = self.create_publisher(MarkerArray, g('markers_topic'), 1)

        self.get_logger().info(
            f'obstacle_extractor up: voxel={self.voxel} ground={self.remove_ground}'
            f'(thr={self.gthresh},vert={self.gminvert}) '
            f'dbscan(eps={self.eps},min={self.min_samples}) '
            f'max_range={self.max_range}')

    # ── attitude ─────────────────────────────────────────────────────────────
    def _att_cb(self, msg):
        # PX4 q is body<-NED Hamilton (w,x,y,z); R_ned_body rotates body->NED.
        w, x, y, z = (float(msg.q[0]), float(msg.q[1]),
                      float(msg.q[2]), float(msg.q[3]))
        R_ned_body = _quat_to_R(w, x, y, z)
        # world<-cam = (NED<-body)(body<-cam); we treat NED as the gravity frame
        # (its z axis is DOWN, so 'height up' = -z_ned below).
        self.R_world_cam = R_ned_body @ R_BODY_CAM
        self._att_stamp = time.time()

    def _gravity_ground_mask(self, xyz):
        """Ground inliers via a height cut in the gravity-aligned frame.

        Rotates points to NED, takes height = -z_ned (up positive), calls the
        low percentile the floor level, and marks everything within
        [floor, floor+margin] as ground. Returns None if attitude is unavailable
        or stale (caller falls back to RANSAC).
        """
        if self.R_world_cam is None:
            return None
        if (time.time() - self._att_stamp) > self.att_timeout:
            return None
        ned = xyz @ self.R_world_cam.T          # (N,3) in NED
        height = -ned[:, 2]                      # NED z is down -> up is -z
        floor = np.percentile(height, self.floor_pct)
        return height <= (floor + self.ground_margin)

    def _cb(self, cloud: PointCloud2):
        xyz = _read_xyz(cloud)
        if xyz.shape[0] == 0:
            self._publish([], cloud.header)
            return

        # range clip (drop far points; forward is +z in optical frame)
        rng = np.linalg.norm(xyz, axis=1)
        xyz = xyz[rng <= self.max_range]

        xyz = _voxel_downsample(xyz, self.voxel)
        if xyz.shape[0] < self.min_samples:
            self._publish([], cloud.header)
            return

        self._used_gravity = False
        if self.remove_ground:
            # prefer gravity-aligned height cut; fall back to RANSAC if no attitude
            ground = self._gravity_ground_mask(xyz)
            if ground is not None:
                self._used_gravity = True
            else:
                ground = _ransac_ground(xyz, self.gthresh, self.giters,
                                        self.up_axis, self.gminvert, self.rng)
            xyz = xyz[~ground]
        if xyz.shape[0] < self.min_samples:
            self._publish([], cloud.header)
            return

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xyz)

        boxes = []
        for lab in set(labels):
            if lab == -1:
                continue  # DBSCAN noise
            pts = xyz[labels == lab]
            if pts.shape[0] < self.min_cluster_pts:
                continue
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            center = (lo + hi) / 2.0
            size = np.maximum(hi - lo, 0.05)  # floor size so thin boxes are visible
            min_range = float(np.linalg.norm(pts, axis=1).min())
            boxes.append((center, size, min_range, int(pts.shape[0])))

        self._publish(boxes, cloud.header)
        self._tick(len(boxes))

    def _publish(self, boxes, header):
        arr = MarkerArray()
        # clear stale boxes first
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, (center, size, min_range, npts) in enumerate(boxes):
            m = Marker()
            m.header = header  # optical frame from the cloud
            m.ns = 'obstacles'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(size[0])
            m.scale.y = float(size[1])
            m.scale.z = float(size[2])
            # colour by nearest range: close = red, far = green
            t = min(min_range / self.max_range, 1.0)
            m.color.r = float(1.0 - t)
            m.color.g = float(t)
            m.color.b = 0.2
            m.color.a = 0.5
            arr.markers.append(m)
        self.pub.publish(arr)

    def _tick(self, n_boxes):
        self._n_since_log += 1
        now = time.time()
        dt = now - self._t_last_log
        if dt >= 1.0:
            method = 'gravity' if self._used_gravity else 'ransac'
            self.get_logger().info(
                f'obstacles @ {self._n_since_log / dt:.1f} Hz ({n_boxes} boxes, '
                f'ground={method})')
            self._t_last_log = now
            self._n_since_log = 0


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleExtractorNode()
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
