"""Runtime fusion helpers built on top of core fusion modules."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from senseye.fusion.tomography import LinkMeasurement, reconstruct
from senseye.fusion.trilateration import trilaterate
from senseye.node.belief import Belief

_MAX_DEVICE_RANGE = 40.0


def estimate_device_positions(
    beliefs: list[Belief],
    node_positions: dict[str, tuple[float, float]],
    min_anchors: int = 3,
) -> dict[str, tuple[tuple[float, float], float]]:
    """Estimate device positions from per-node distance beliefs via trilateration."""
    by_device: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for belief in sorted(beliefs, key=lambda item: item.timestamp):
        anchor_id = belief.node_id
        if anchor_id not in node_positions:
            continue
        for device_id, device in belief.devices.items():
            if device_id in node_positions:
                continue
            if device_id.startswith("acoustic:echo:"):
                continue
            distance = device.estimated_distance
            if distance is None or distance <= 0 or distance > _MAX_DEVICE_RANGE:
                continue
            by_device[device_id][anchor_id].append(float(distance))

    estimates: dict[str, tuple[tuple[float, float], float]] = {}
    for device_id, anchor_to_samples in by_device.items():
        if len(anchor_to_samples) < min_anchors:
            continue
        anchor_distances: dict[str, float] = {}
        for anchor_id, samples in anchor_to_samples.items():
            if not samples:
                continue
            # Median is robust to occasional RSSI distance outliers.
            anchor_distances[anchor_id] = float(np.median(np.array(samples)))

        observations = [
            (node_positions[anchor_id], distance)
            for anchor_id, distance in anchor_distances.items()
            if anchor_id in node_positions
        ]
        if len(observations) < min_anchors:
            continue
        result = trilaterate(observations)
        if result is not None:
            estimates[device_id] = result
    return estimates


def reconstruct_attenuation_grid(
    beliefs: list[Belief],
    node_positions: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float = 0.5,
) -> np.ndarray:
    """Reconstruct an attenuation image from link beliefs with known endpoints."""
    pair_to_stats: dict[tuple[str, str], list[float]] = {}
    for belief in beliefs:
        src = belief.node_id
        if src not in node_positions:
            continue
        for target, link in belief.links.items():
            if target not in node_positions:
                continue
            if link.attenuation <= 0:
                continue
            pair = (src, target) if src <= target else (target, src)
            weight = max(link.confidence, 0.05)
            stats = pair_to_stats.setdefault(pair, [0.0, 0.0, 0.0, 0.0])
            # [weighted attenuation sum, weight sum, confidence sum, count]
            stats[0] += link.attenuation * weight
            stats[1] += weight
            stats[2] += link.confidence
            stats[3] += 1.0

    if not pair_to_stats:
        return np.array([[]])

    measurements = [
        LinkMeasurement(
            p1=node_positions[src],
            p2=node_positions[target],
            excess_attenuation=(stats[0] / max(stats[1], 1e-6)),
            confidence=min(max(stats[2] / max(stats[3], 1.0), 0.01), 1.0),
        )
        for (src, target), stats in pair_to_stats.items()
    ]
    return reconstruct(
        links=measurements,
        bounds=bounds,
        resolution=resolution,
    )
