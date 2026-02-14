"""Local inference: motion detection, attenuation estimation, zone beliefs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from senseye.node.belief import Belief, DeviceState, LinkState, ZoneBelief
from senseye.node.scanner import Observation


@dataclass
class ZoneConfig:
    name: str
    # Links that cross this zone, as (source_id, target_id) pairs
    links: list[tuple[str, str]]


# RSSI history buffer: device_id -> list of recent raw RSSI values.
# Used for variance-based motion detection (separate from Kalman filtering).
RssiHistory = dict[str, list[float]]


def _rssi_variance(samples: list[float], window: int = 10) -> float:
    """Variance of the last `window` samples."""
    tail = samples[-window:]
    if len(tail) < 2:
        return 0.0
    mean = sum(tail) / len(tail)
    return sum((x - mean) ** 2 for x in tail) / len(tail)


def _free_space_rssi(distance: float, n: float = 2.5, a: float = 45.0) -> float:
    """Expected RSSI at given distance under free-space path loss model.

    Returns a negative value (dBm-like).
    rssi_expected = -(10 * n * log10(d) + A)
    """
    if distance <= 0:
        return 0.0
    return -(10.0 * n * math.log10(distance) + a)


def _rssi_to_distance(rssi: float, n: float = 2.5, a: float = 45.0) -> float:
    """Invert the free-space model to estimate distance from RSSI.

    distance = 10 ^ ((-rssi - A) / (10 * n))
    """
    exponent = (-rssi - a) / (10.0 * n)
    return 10.0 ** exponent


def infer(
    observations: list[Observation],
    rssi_history: RssiHistory,
    node_positions: dict[str, tuple[float, float]] | None = None,
    zone_config: list[ZoneConfig] | None = None,
    my_node_id: str = "",
    motion_window: int = 10,
    motion_threshold: float = 2.0,
) -> Belief:
    """Produce a Belief from current observations and accumulated filter state.

    Args:
        observations: latest scan results
        rssi_history: device_id -> list of recent RSSI values (caller maintains)
        node_positions: known (x,y) positions of fixed nodes, keyed by node_id
        zone_config: zone definitions for zone-level inference
        my_node_id: this node's ID (used as Belief.node_id)
        motion_window: sliding window size for variance calculation
        motion_threshold: variance threshold (dB^2) for motion detection
    """
    if node_positions is None:
        node_positions = {}

    # Update filter bank with new observations
    for obs in observations:
        fb = rssi_history.setdefault(obs.device_id, [])
        fb.append(obs.rssi)

    # --- Per-link inference ---
    links: dict[str, LinkState] = {}
    devices: dict[str, DeviceState] = {}

    for device_id, samples in rssi_history.items():
        if not samples:
            continue

        # Motion: high variance in recent samples
        var = _rssi_variance(samples, motion_window)
        has_motion = var > motion_threshold

        # Current RSSI: latest sample
        current_rssi = samples[-1]

        # Attenuation: compare to free-space model if we know positions
        my_pos = node_positions.get(my_node_id)
        peer_pos = node_positions.get(device_id)

        if my_pos is not None and peer_pos is not None:
            dx = my_pos[0] - peer_pos[0]
            dy = my_pos[1] - peer_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                expected = _free_space_rssi(dist)
                # Excess attenuation: how much weaker than free-space
                attenuation = expected - current_rssi
            else:
                attenuation = 0.0

            # Confidence: more samples = more confident, cap at 1.0
            confidence = min(len(samples) / motion_window, 1.0)
            links[device_id] = LinkState(
                attenuation=max(attenuation, 0.0),
                motion=has_motion,
                confidence=confidence,
            )
        else:
            # No position info: still record link with zero attenuation
            confidence = min(len(samples) / motion_window, 1.0)
            links[device_id] = LinkState(
                attenuation=0.0,
                motion=has_motion,
                confidence=confidence,
            )

        # Device tracking: estimate distance from RSSI
        est_distance = _rssi_to_distance(current_rssi)
        devices[device_id] = DeviceState(
            rssi=current_rssi,
            estimated_distance=est_distance,
            moving=has_motion,
        )

    # --- Zone inference ---
    zones: dict[str, ZoneBelief] = {}
    if zone_config:
        for zone in zone_config:
            zone_motion_votes = 0
            zone_attenuation_sum = 0.0
            zone_link_count = 0

            for src, tgt in zone.links:
                # Check if we have a link state for either direction
                link = links.get(tgt) if src == my_node_id else links.get(src)
                if link is None:
                    continue
                zone_link_count += 1
                if link.motion:
                    zone_motion_votes += 1
                zone_attenuation_sum += link.attenuation

            if zone_link_count > 0:
                motion_prob = zone_motion_votes / zone_link_count
                # Occupied heuristic: high attenuation suggests something is there
                avg_atten = zone_attenuation_sum / zone_link_count
                occupied_prob = min(avg_atten / 20.0, 1.0)  # 20 dB = fully occluded
            else:
                motion_prob = 0.0
                occupied_prob = 0.0

            zones[zone.name] = ZoneBelief(
                occupied=occupied_prob,
                motion=motion_prob,
            )

    return Belief(
        node_id=my_node_id,
        links=links,
        devices=devices,
        zones=zones,
    )
