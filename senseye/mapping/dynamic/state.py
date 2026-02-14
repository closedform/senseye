"""Combined world state: static map + dynamic overlay."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from senseye.mapping.dynamic.devices import TrackedDevice
from senseye.mapping.dynamic.motion import MotionState, update_motion
from senseye.mapping.static.floorplan import FloorPlan
from senseye.node.belief import Belief


@dataclass
class NodeInfo:
    node_id: str
    name: str
    role: str
    online: bool
    last_seen: float


@dataclass
class WorldState:
    floorplan: FloorPlan | None = None
    motion: MotionState = field(default_factory=MotionState)
    devices: dict[str, TrackedDevice] = field(default_factory=dict)
    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    map_age: float = 0.0  # seconds since calibration
    timestamp: float = field(default_factory=time.time)


def _nearest_room(
    floorplan: FloorPlan | None,
    position: tuple[float, float] | None,
) -> str | None:
    if floorplan is None or position is None:
        return None
    best_name: str | None = None
    best_dist = float("inf")
    px, py = position
    for room in floorplan.rooms.rooms:
        if room.center is None:
            continue
        dx = px - room.center[0]
        dy = py - room.center[1]
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_name = room.name
    return best_name


def update_world(
    state: WorldState,
    belief: Belief,
    dt: float,
    device_positions: dict[str, tuple[float, float]] | None = None,
    device_signal_types: dict[str, str] | None = None,
    online_nodes: set[str] | None = None,
) -> WorldState:
    """Single entry point for updating the dynamic layer from a fused belief.

    Updates motion state from zone beliefs, updates device info,
    upserts the reporting node, and recomputes map age.
    """
    now = time.time()

    # Update motion from zone beliefs
    state.motion = update_motion(state.motion, belief.zones, dt)

    # Update devices from belief
    for device_id, dev_state in belief.devices.items():
        existing = state.devices.get(device_id)
        if existing is None:
            existing = TrackedDevice(device_id=device_id)
            state.devices[device_id] = existing
        existing.moving = dev_state.moving
        existing.last_seen = now
        if device_positions and device_id in device_positions:
            existing.position = device_positions[device_id]
        if device_signal_types and device_id in device_signal_types:
            existing.signal_type = device_signal_types[device_id]
        label = state.floorplan.labels.get(device_id) if state.floorplan is not None else None
        if label:
            existing.name = label
        zone = _nearest_room(state.floorplan, existing.position)
        if zone is not None:
            existing.zone = zone

    # Upsert reporting nodes (local + peers seen in recent fusion window).
    active_nodes = online_nodes or {belief.node_id}
    for node_id in active_nodes:
        node = state.nodes.get(node_id)
        if node is None:
            state.nodes[node_id] = NodeInfo(
                node_id=node_id,
                name=node_id,
                role="fixed",
                online=True,
                last_seen=now,
            )
            continue
        node.online = True
        node.last_seen = now

    # Mark stale nodes as offline.
    for node in state.nodes.values():
        if now - node.last_seen > 15.0:
            node.online = False

    # Update map age
    if state.floorplan is not None:
        state.map_age = now - state.floorplan.calibrated_at
    else:
        state.map_age = 0.0

    state.timestamp = now
    return state
