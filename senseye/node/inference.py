"""Local inference: motion detection, attenuation estimation, zone beliefs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from senseye.node.belief import Belief, DeviceState, LinkState, ZoneBelief
from senseye.node.scanner import Observation, SignalType


@dataclass
class ZoneConfig:
    name: str
    # Links that cross this zone, as (source_id, target_id) pairs
    links: list[tuple[str, str]]


# RSSI history buffer: device_id -> list of recent raw RSSI values.
# Used for variance-based motion detection (separate from Kalman filtering).
RssiHistory = dict[str, list[float]]
_MAX_RF_DISTANCE_M = 40.0
_MIN_RF_DISTANCE_M = 0.2

# Log-distance path-loss model parameters (indoor)
PATHLOSS_N = 2.5   # path-loss exponent
PATHLOSS_A = 45.0  # 1-meter intercept (dBm magnitude)


def _rssi_variance(samples: list[float], window: int = 10) -> float:
    """Variance of the last `window` samples."""
    tail = samples[-window:]
    if len(tail) < 2:
        return 0.0
    mean = sum(tail) / len(tail)
    return sum((x - mean) ** 2 for x in tail) / len(tail)


def _free_space_rssi(distance: float, n: float = PATHLOSS_N, a: float = PATHLOSS_A) -> float:
    """Expected RSSI at given distance under free-space path loss model.

    Returns a negative value (dBm-like).
    rssi_expected = -(10 * n * log10(d) + A)
    """
    if distance <= 0:
        return 0.0
    return -(10.0 * n * math.log10(distance) + a)


def _rssi_to_distance(rssi: float, n: float = PATHLOSS_N, a: float = PATHLOSS_A) -> float:
    """Invert the free-space model to estimate distance from RSSI.

    distance = 10 ^ ((-rssi - A) / (10 * n))
    """
    exponent = (-rssi - a) / (10.0 * n)
    distance = 10.0 ** exponent
    return min(max(distance, _MIN_RF_DISTANCE_M), _MAX_RF_DISTANCE_M)


def _raw_rssi_sample(obs: Observation) -> float:
    raw = obs.metadata.get("raw_rssi")
    if isinstance(raw, int | float):
        return float(raw)
    return obs.rssi


def _sample_confidence(sample_count: int, motion_window: int) -> float:
    window = max(motion_window, 1)
    return min(sample_count / window, 1.0)


def _innovation_penalty(observation: Observation | None) -> float:
    if observation is None:
        return 1.0
    innovation = observation.metadata.get("innovation")
    if isinstance(innovation, int | float):
        return 1.0 / (1.0 + (abs(float(innovation)) / 8.0))
    return 1.0


def _acoustic_confidence(observation: Observation | None, sample_confidence: float) -> float:
    if observation is None:
        return sample_confidence
    snr = observation.metadata.get("peak_snr")
    if not isinstance(snr, int | float):
        return sample_confidence
    snr_confidence = min(max((float(snr) - 1.0) / 8.0, 0.05), 1.0)
    return min(max((0.4 * sample_confidence) + (0.6 * snr_confidence), 0.05), 1.0)


def infer(
    observations: list[Observation],
    rssi_history: RssiHistory,
    node_positions: dict[str, tuple[float, float]] | None = None,
    zone_config: list[ZoneConfig] | None = None,
    my_node_id: str = "",
    motion_window: int = 10,
    motion_threshold: float = 2.0,
    history_limit: int = 120,
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
        history_limit: max history length retained per device
    """
    if node_positions is None:
        node_positions = {}

    # Motion should track raw RSSI variance, while attenuation and distance
    # should use the latest filtered RSSI.
    current_rssi_by_device: dict[str, float] = {}
    latest_observation_by_device: dict[str, Observation] = {}
    for obs in observations:
        fb = rssi_history.setdefault(obs.device_id, [])
        fb.append(_raw_rssi_sample(obs))
        if len(fb) > history_limit:
            del fb[:-history_limit]
        current_rssi_by_device[obs.device_id] = obs.rssi
        latest_observation_by_device[obs.device_id] = obs

    # --- Per-link inference ---
    links: dict[str, LinkState] = {}
    devices: dict[str, DeviceState] = {}

    for device_id, samples in rssi_history.items():
        if not samples:
            continue

        # Motion: high variance in recent samples
        var = _rssi_variance(samples, motion_window)
        has_motion = var > motion_threshold

        # Use filtered RSSI when available this cycle, otherwise last sample.
        current_rssi = current_rssi_by_device.get(device_id, samples[-1])
        latest_observation = latest_observation_by_device.get(device_id)
        is_acoustic = (
            latest_observation is not None
            and latest_observation.signal_type == SignalType.ACOUSTIC
        )
        base_confidence = _sample_confidence(len(samples), motion_window)
        confidence = base_confidence * _innovation_penalty(latest_observation)

        # Attenuation: compare to free-space model if we know positions
        my_pos = node_positions.get(my_node_id)
        peer_pos = node_positions.get(device_id)

        if is_acoustic:
            confidence = _acoustic_confidence(latest_observation, base_confidence)
            links[device_id] = LinkState(
                attenuation=0.0,
                motion=has_motion,
                confidence=confidence,
            )
        elif my_pos is not None and peer_pos is not None:
            dx = my_pos[0] - peer_pos[0]
            dy = my_pos[1] - peer_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                expected = _free_space_rssi(dist)
                # Excess attenuation: how much weaker than free-space
                attenuation = expected - current_rssi
            else:
                attenuation = 0.0

            links[device_id] = LinkState(
                attenuation=max(attenuation, 0.0),
                motion=has_motion,
                confidence=confidence,
            )
        else:
            # No position info: still record link with zero attenuation
            links[device_id] = LinkState(
                attenuation=0.0,
                motion=has_motion,
                confidence=confidence,
            )

        # Device tracking: estimate distance from RSSI, or use direct acoustic distance.
        distance_meta: float | None = None
        if is_acoustic and latest_observation is not None:
            raw_distance = latest_observation.metadata.get("distance_m")
            if isinstance(raw_distance, int | float):
                distance_meta = float(raw_distance)
        est_distance = (
            distance_meta
            if distance_meta is not None
            else _rssi_to_distance(current_rssi)
        )
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
