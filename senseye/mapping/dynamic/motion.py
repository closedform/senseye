"""Per-zone motion state tracking with exponential decay."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from senseye.node.belief import ZoneBelief


@dataclass
class MotionState:
    zone_motion: dict[str, float] = field(default_factory=dict)  # zone -> intensity 0-1
    last_updated: dict[str, float] = field(default_factory=dict)  # zone -> timestamp


def update_motion(
    state: MotionState,
    zone_beliefs: dict[str, ZoneBelief],
    dt: float,
    decay: float = 0.3,
) -> MotionState:
    """Update motion intensities from zone beliefs, applying exponential decay.

    New observations set intensity to max(current, observed).
    Without new observations, intensity decays: intensity *= exp(-decay * dt).
    """
    now = time.time()

    # Apply decay to all existing zones
    for zone in list(state.zone_motion):
        state.zone_motion[zone] *= math.exp(-decay * dt)
        # Clamp near-zero to zero
        if state.zone_motion[zone] < 0.01:
            state.zone_motion[zone] = 0.0

    # Merge in new observations
    for zone, belief in zone_beliefs.items():
        current = state.zone_motion.get(zone, 0.0)
        state.zone_motion[zone] = max(current, belief.motion)
        state.last_updated[zone] = now

    return state
