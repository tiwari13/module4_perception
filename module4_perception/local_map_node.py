#!/usr/bin/env python3
"""Local map node — Phase 6, persistent rolling voxel map.

Phase 5's obstacle_extractor_node re-detects obstacles fresh every frame and
emits a MarkerArray that a consumer must react to immediately -- there is no
memory. A single missed/occluded frame makes an obstacle vanish from the
picture entirely, and a planner reading /perception/obstacles directly has no
way to tell "genuinely gone" from "camera didn't see it this tick". This node
adds that memory: a rolling 3D voxel grid, in WORLD (PX4 NED local) frame,
that persists occupied cells across frames and decays them to unknown (not
free) if they go unconfirmed for too long.

Pipeline per /perception/obstacles frame:
  1. Read the drone's current PX4 NED position (/fmu/out/vehicle_local_position_v1).
  2. For each obstacle cluster marker (still in the camera's left-optical
     frame, per obstacle_extractor_node's docstring -- it does not transform),
     rotate optical -> body-FRD (fixed R_BODY_CAM, same convention as
     obstacle_extractor_node.py and smart_navigator.py) -> add the drone's NED
     position -> world-frame point.
  3. Mark the voxel(s) covering that cluster's AABB as OCCUPIED, stamped with
     the current time.
  4. On a timer tick, sweep all known voxels: any not re-confirmed within
     `decay_s` is dropped from the "occupied" set -- NOT marked free. A
     dropped voxel is simply absent, i.e. UNKNOWN. Treating unknown as unsafe
     is the CALLER's job (this node reports occupied vs not-currently-known;
     it does not claim anything is safe).
  5. Rolling window: voxels farther than `map_radius_m` from the drone's
     current position are pruned every tick, so memory stays bounded near the
     vehicle instead of growing over an entire flight.

Dependency-light by the same rule as obstacle_extractor_node.py: NumPy +
stdlib only, no PCL/Open3D/octomap. A voxel is just a dict key
(ix, iy, iz) -> last_seen_monotonic. Fine for a bounded local map (tens of
metres); not the right choice for city-scale mapping.

Output:
  /perception/local_map          (visualization_msgs/MarkerArray -- one CUBE
                                   per occupied voxel, RViz/debug-visible,
                                   same pragmatic contract obstacle_extractor
                                   uses ahead of a real map message type)
Input:
  /perception/obstacles          (from obstacle_extractor_node)
  /fmu/out/vehicle_local_position_v1  (PX4 NED position)

Run (stereo_depth_node + obstacle_extractor_node + PX4 up):
  ros2 run module4_perception local_map_node
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from px4_msgs.msg import VehicleLocalPosition
from visualization_msgs.msg import Marker, MarkerArray

# PX4 /fmu/out/* are best-effort.
_PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=1)

# Same fixed camera-optical -> body-FRD rotation as obstacle_extractor_node.py
# and smart_navigator.py's _R_BODY_CAM. Optical: x-right, y-down, z-forward;
# body FRD: x-forward, y-right, z-down.
R_BODY_CAM = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])


class LocalMapNode(Node):
    def __init__(self):
        super().__init__('local_map_node')

        self.declare_parameter('obstacles_topic', '/perception/obstacles')
        self.declare_parameter('position_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('map_topic', '/perception/local_map')
        self.declare_parameter('voxel_size', 0.5)       # m; map resolution
        self.declare_parameter('decay_s', 3.0)          # unconfirmed -> unknown
        self.declare_parameter('map_radius_m', 20.0)    # prune beyond this
        self.declare_parameter('sweep_period_s', 0.5)   # decay/prune tick rate
        self.declare_parameter('position_timeout_s', 1.0)

        g = lambda n: self.get_parameter(n).value
        self.voxel_size = float(g('voxel_size'))
        self.decay_s = float(g('decay_s'))
        self.map_radius_m = float(g('map_radius_m'))
        self.position_timeout = float(g('position_timeout_s'))

        # voxel key (ix, iy, iz) in world-frame metres/voxel_size -> last_seen
        # (time.monotonic()). Absence = unknown, not free -- see module docstring.
        self.voxels: dict[tuple[int, int, int], float] = {}

        self.drone_pos = None            # [x, y, z] NED, metres
        self._last_pos_update = 0.0

        self.create_subscription(VehicleLocalPosition, g('position_topic'),
                                 self._pos_cb, _PX4_QOS)
        self.create_subscription(MarkerArray, g('obstacles_topic'),
                                 self._obstacles_cb, 1)
        self.pub = self.create_publisher(MarkerArray, g('map_topic'), 1)

        self.create_timer(float(g('sweep_period_s')), self._sweep_and_publish)

        self._t_last_log = time.time()
        self._n_updates_since_log = 0

        self.get_logger().info(
            f'local_map_node up: voxel={self.voxel_size}m decay={self.decay_s}s '
            f'radius={self.map_radius_m}m')

    # ── inputs ──────────────────────────────────────────────────────────────
    def _pos_cb(self, msg: VehicleLocalPosition):
        if not (msg.xy_valid and msg.z_valid):
            return
        self.drone_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
        self._last_pos_update = time.monotonic()

    def _position_healthy(self) -> bool:
        return (self.drone_pos is not None and
                time.monotonic() - self._last_pos_update <= self.position_timeout)

    def _obstacles_cb(self, msg: MarkerArray):
        if not self._position_healthy():
            # Can't place obstacles in world frame without a current position;
            # drop this frame rather than guess. Existing voxels still decay
            # normally via the sweep timer.
            return

        now = time.monotonic()
        n_marked = 0
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            center_cam = np.array([m.pose.position.x, m.pose.position.y,
                                   m.pose.position.z])
            size_cam = np.array([m.scale.x, m.scale.y, m.scale.z])

            center_body = R_BODY_CAM @ center_cam
            # AABB half-extent doesn't need re-rotation for a coarse voxel
            # stamp (axes just permute under this rotation) -- reorder to match.
            half_body = np.abs(R_BODY_CAM @ size_cam) / 2.0

            center_world = self.drone_pos + center_body

            self._mark_voxels_in_aabb(center_world, half_body, now)
            n_marked += 1

        self._n_updates_since_log += n_marked

    def _mark_voxels_in_aabb(self, center_world, half_extent, stamp):
        lo = np.floor((center_world - half_extent) / self.voxel_size).astype(int)
        hi = np.floor((center_world + half_extent) / self.voxel_size).astype(int)
        # cap per-obstacle voxel count defensively -- a bad/huge cluster
        # shouldn't be able to blow up the map in one frame
        span = np.clip(hi - lo + 1, 1, 20)
        for ix in range(lo[0], lo[0] + span[0]):
            for iy in range(lo[1], lo[1] + span[1]):
                for iz in range(lo[2], lo[2] + span[2]):
                    self.voxels[(ix, iy, iz)] = stamp

    # ── decay + prune + publish ────────────────────────────────────────────
    def _sweep_and_publish(self):
        now = time.monotonic()

        # decay: drop anything unconfirmed for too long (-> unknown)
        stale = [k for k, t in self.voxels.items() if now - t > self.decay_s]
        for k in stale:
            del self.voxels[k]

        # prune: drop anything outside the rolling window around the drone
        if self.drone_pos is not None and self.voxels:
            center_idx = self.drone_pos / self.voxel_size
            radius_vox = self.map_radius_m / self.voxel_size
            far = [k for k in self.voxels
                  if np.linalg.norm(np.array(k) - center_idx) > radius_vox]
            for k in far:
                del self.voxels[k]

        self._publish()
        self._tick()

    def _publish(self):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        now_wall = self.get_clock().now().to_msg()
        for i, ((ix, iy, iz), _) in enumerate(self.voxels.items()):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now_wall
            m.ns = 'local_map'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = (ix + 0.5) * self.voxel_size
            m.pose.position.y = (iy + 0.5) * self.voxel_size
            m.pose.position.z = (iz + 0.5) * self.voxel_size
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = self.voxel_size
            m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.2, 0.8, 0.4
            arr.markers.append(m)

        self.pub.publish(arr)

    def _tick(self):
        now = time.time()
        dt = now - self._t_last_log
        if dt >= 2.0:
            self.get_logger().info(
                f'local_map: {len(self.voxels)} occupied voxels '
                f'({self._n_updates_since_log} cluster updates in last {dt:.1f}s)')
            self._t_last_log = now
            self._n_updates_since_log = 0


def main(args=None):
    rclpy.init(args=args)
    node = LocalMapNode()
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
