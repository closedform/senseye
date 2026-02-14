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


def update_world(state: WorldState, belief: Belief, dt: float) -> WorldState:
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

    # Upsert the reporting node
    node = state.nodes.get(belief.node_id)
    if node is None:
        state.nodes[belief.node_id] = NodeInfo(
            node_id=belief.node_id,
            name=belief.node_id,
            role="fixed",
            online=True,
            last_seen=now,
        )
    else:
        node.online = True
        node.last_seen = now

    # Update map age
    if state.floorplan is not None:
        state.map_age = now - state.floorplan.calibrated_at
    else:
        state.map_age = 0.0

    state.timestamp = now
    return state
