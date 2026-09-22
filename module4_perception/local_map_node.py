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

Phase 11 made this genuinely multi-camera: one obstacle_extractor_node
instance runs per physical unit (front/rear/left/right), each publishing on
its own /perception/obstacles_<unit> topic (still in that camera's own
left-optical frame -- extractor does not transform). This node subscribes to
all configured unit topics and, per source, rotates+translates optical ->
body-FRD -> world using that UNIT's own T_base_link_unit from
calibration.yaml (mount rotation differs per unit: front=0, rear=180deg,
left/right=+-90deg about z -- reusing one fixed front-only rotation for all
four would silently place side/rear obstacles at the wrong world location).

Pipeline per /perception/obstacles_<unit> frame:
  1. Read the drone's current PX4 NED position (/fmu/out/vehicle_local_position_v1).
  2. For each obstacle cluster marker, rotate optical -> body-FRD using the
     fixed per-camera-type convention (R_BODY_CAM_FWD, same convention as
     obstacle_extractor_node.py and smart_navigator.py for a forward-looking
     optical frame) composed with that unit's own body-mount rotation from
     calibration.yaml, then add the unit's body-frame mount translation and
     the drone's NED position -> world-frame point.
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
  /perception/obstacles_<unit>   (from one obstacle_extractor_node per unit;
                                   unit names come from the `units` param,
                                   default front/rear/left/right)
  /fmu/out/vehicle_local_position_v1  (PX4 NED position)

Run (stereo_depth_node + obstacle_extractor_node x N + PX4 up):
  ros2 run module4_perception local_map_node
"""

import functools
import os
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from visualization_msgs.msg import Marker, MarkerArray

# PX4 /fmu/out/* are best-effort.
_PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=1)

# Fixed camera-optical -> body-FRD rotation for a FORWARD-facing OAK-D, i.e.
# what the mount rotation in calibration.yaml is defined relative to. Optical
# frame: x-right, y-down, z-forward; body FRD: x-forward, y-right, z-down.
# Same convention as obstacle_extractor_node.py and smart_navigator.py.
R_BODY_CAM_FWD = np.array([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def _quat_to_R(w, x, y, z):
    """Hamilton quaternion (w,x,y,z) -> 3x3 rotation matrix. Same convention
    as obstacle_extractor_node.py's _quat_to_R (duplicated rather than
    cross-imported, matching how that file already duplicates it locally
    instead of sharing a common module across these two separate packages)."""
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = np.array([w, x, y, z]) / np.sqrt(n)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _load_unit_mounts(unit_names):
    """Per-unit (R_body_cam, t_body_mount) from calibration.yaml.

    R_body_cam includes both the unit's own body-yaw mount rotation and the
    fixed forward-optical->body-FRD convention, so callers just do
    R_body_cam @ point_optical + t_body_mount. Fails loud, same policy as
    step7_sensor_bridge._load_calibration -- never silently default a mount
    pose we don't actually have.
    """
    try:
        pkg_share = get_package_share_directory('sim_assets')
    except Exception as e:
        raise RuntimeError(
            f"sim_assets package not found ({e}). "
            f"Source the workspace and rebuild sim_assets.")

    path = os.path.join(pkg_share, 'config', 'calibration.yaml')
    if not os.path.isfile(path):
        raise RuntimeError(f"calibration.yaml not found at {path}")

    with open(path) as f:
        cal = yaml.safe_load(f)

    if 'units' not in cal:
        raise RuntimeError("calibration.yaml missing 'units' block")

    mounts = {}
    for unit in unit_names:
        if unit not in cal['units']:
            raise RuntimeError(
                f"calibration.yaml units missing '{unit}' "
                f"(configured in local_map_node's 'units' param)")
        mount = cal['units'][unit].get('T_base_link_unit')
        if mount is None:
            raise RuntimeError(
                f"calibration.yaml units.{unit} missing T_base_link_unit")
        roll, pitch, yaw = mount['rotation_rpy']
        if abs(roll) > 1e-9 or abs(pitch) > 1e-9:
            raise RuntimeError(
                f"calibration.yaml units.{unit}.T_base_link_unit has "
                f"nonzero roll/pitch ({roll},{pitch}) -- _rotz-only mount "
                f"composition below assumes yaw-only mounts (true for the "
                f"current front/rear/left/right rig); extend _load_unit_mounts "
                f"before adding a tilted or upward/downward unit.")
        R_body_cam = _rotz(yaw) @ R_BODY_CAM_FWD
        t_body_mount = np.array(mount['translation'], dtype=np.float64)
        mounts[unit] = (R_body_cam, t_body_mount)
    return mounts


class LocalMapNode(Node):
    def __init__(self):
        super().__init__('local_map_node')

        self.declare_parameter('units', ['front', 'rear', 'left', 'right'])
        self.declare_parameter('obstacles_topic_template', '/perception/obstacles_{unit}')
        self.declare_parameter('position_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('attitude_topic', '/fmu/out/vehicle_attitude')
        self.declare_parameter('map_topic', '/perception/local_map')
        self.declare_parameter('voxel_size', 0.5)       # m; map resolution
        self.declare_parameter('decay_s', 3.0)          # unconfirmed -> unknown
        self.declare_parameter('map_radius_m', 20.0)    # prune beyond this
        self.declare_parameter('sweep_period_s', 0.5)   # decay/prune tick rate
        self.declare_parameter('position_timeout_s', 1.0)
        self.declare_parameter('attitude_timeout_s', 1.0)

        g = lambda n: self.get_parameter(n).value
        self.voxel_size = float(g('voxel_size'))
        self.decay_s = float(g('decay_s'))
        self.map_radius_m = float(g('map_radius_m'))
        self.position_timeout = float(g('position_timeout_s'))
        self.attitude_timeout = float(g('attitude_timeout_s'))
        units = list(g('units'))
        topic_template = g('obstacles_topic_template')

        # voxel key (ix, iy, iz) in world-frame metres/voxel_size -> last_seen
        # (time.monotonic()). Absence = unknown, not free -- see module docstring.
        self.voxels: dict[tuple[int, int, int], float] = {}

        self.drone_pos = None            # [x, y, z] NED, metres
        self._last_pos_update = 0.0

        # body(FRD)->NED rotation from PX4 attitude. Needed because obstacles
        # are placed in BODY frame first (R_body_cam @ point + t_body_mount)
        # and body axes only equal world/NED axes when the drone has zero
        # roll/pitch/yaw -- under any real attitude (confirmed live: this rig
        # sat at 90.4deg yaw during Phase 11 validation) skipping this
        # rotation silently misplaces every obstacle in world XY.
        self.R_ned_body = None
        self._last_att_update = 0.0

        # calibration.yaml keys are 'cam_<unit>'; params use the bare unit
        # name ('front'/'rear'/'left'/'right') to keep topic names short.
        mounts = _load_unit_mounts([f'cam_{u}' for u in units])

        self.create_subscription(VehicleLocalPosition, g('position_topic'),
                                 self._pos_cb, _PX4_QOS)
        self.create_subscription(VehicleAttitude, g('attitude_topic'),
                                 self._att_cb, _PX4_QOS)
        for unit in units:
            R_body_cam, t_body_mount = mounts[f'cam_{unit}']
            topic = topic_template.format(unit=unit)
            self.create_subscription(
                MarkerArray,
                topic,
                functools.partial(self._obstacles_cb,
                                  R_body_cam=R_body_cam,
                                  t_body_mount=t_body_mount,
                                  unit=unit),
                1)
            self.get_logger().info(f'local_map_node: subscribed {topic} (unit={unit})')

        self.pub = self.create_publisher(MarkerArray, g('map_topic'), 1)

        self.create_timer(float(g('sweep_period_s')), self._sweep_and_publish)

        self._t_last_log = time.time()
        self._n_updates_since_log = 0

        self.get_logger().info(
            f'local_map_node up: voxel={self.voxel_size}m decay={self.decay_s}s '
            f'radius={self.map_radius_m}m units={units}')

    # ── inputs ──────────────────────────────────────────────────────────────
    def _pos_cb(self, msg: VehicleLocalPosition):
        if not (msg.xy_valid and msg.z_valid):
            return
        self.drone_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
        self._last_pos_update = time.monotonic()

    def _position_healthy(self) -> bool:
        return (self.drone_pos is not None and
                time.monotonic() - self._last_pos_update <= self.position_timeout)

    def _att_cb(self, msg: VehicleAttitude):
        # PX4 q is body<-NED Hamilton (w,x,y,z) -- same convention and same
        # _quat_to_R as obstacle_extractor_node.py's _att_cb.
        w, x, y, z = (float(msg.q[0]), float(msg.q[1]),
                      float(msg.q[2]), float(msg.q[3]))
        self.R_ned_body = _quat_to_R(w, x, y, z)
        self._last_att_update = time.monotonic()

    def _attitude_healthy(self) -> bool:
        return (self.R_ned_body is not None and
                time.monotonic() - self._last_att_update <= self.attitude_timeout)

    def _obstacles_cb(self, msg: MarkerArray, R_body_cam, t_body_mount, unit):
        if not self._position_healthy():
            # Can't place obstacles in world frame without a current position;
            # drop this frame rather than guess. Existing voxels still decay
            # normally via the sweep timer.
            return
        if not self._attitude_healthy():
            # Same policy as the position check: without a current attitude
            # we cannot correctly rotate body-frame obstacles into world/NED
            # (body axes == world axes only at zero roll/pitch/yaw) -- drop
            # the frame rather than silently place obstacles as if the drone
            # were pointed north.
            return

        now = time.monotonic()
        n_marked = 0
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            center_cam = np.array([m.pose.position.x, m.pose.position.y,
                                   m.pose.position.z])
            size_cam = np.array([m.scale.x, m.scale.y, m.scale.z])

            # this unit's own mount rotation (fixed forward-optical->body-FRD
            # composed with the unit's body-yaw), plus its mount offset --
            # NOT the single front-only rotation the pre-Phase-11 version used.
            center_body = R_body_cam @ center_cam + t_body_mount
            # AABB half-extent doesn't need re-rotation for a coarse voxel
            # stamp (axes just permute under this rotation) -- reorder to match.
            half_body = np.abs(R_body_cam @ size_cam) / 2.0

            # rotate body(FRD) -> world(NED) using the drone's CURRENT
            # attitude before adding drone_pos -- body axes only line up
            # with world axes at zero roll/pitch/yaw; skipping this silently
            # misplaces every obstacle in world XY under any real attitude.
            center_world = self.drone_pos + self.R_ned_body @ center_body

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
