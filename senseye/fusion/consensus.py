"""Weighted belief fusion across peers."""

from __future__ import annotations

from senseye.node.belief import Belief, DeviceState, LinkState, ZoneBelief

_EPS = 1e-6


def fuse_beliefs(local: Belief, peer_beliefs: dict[str, Belief]) -> Belief:
    """Fuse local and peer beliefs with inverse-variance weighting."""
    all_beliefs = [local] + list(peer_beliefs.values())
    timestamp = max((belief.timestamp for belief in all_beliefs), default=local.timestamp)
    return Belief(
        node_id=local.node_id,
        timestamp=timestamp,
        links=_fuse_links(all_beliefs),
        devices=_fuse_devices(all_beliefs),
        zones=_fuse_zones(all_beliefs),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _variance_from_confidence(confidence: float) -> float:
    """Map confidence in [0, 1] to a positive variance."""
    c = max(0.01, min(confidence, 0.99))
    return ((1.0 - c) / c) + _EPS


def _precision_from_confidence(confidence: float) -> float:
    return 1.0 / _variance_from_confidence(confidence)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= _EPS:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def _weighted_variance(values: list[float], weights: list[float], mean: float) -> float:
    total_weight = sum(weights)
    if total_weight <= _EPS:
        return 0.0
    return (
        sum(weight * ((value - mean) ** 2) for value, weight in zip(values, weights))
        / total_weight
    )


def _agreement_penalty(values: list[float], weights: list[float], scale: float) -> float:
    if len(values) < 2:
        return 1.0
    variance = _weighted_variance(values, weights, _weighted_mean(values, weights))
    return 1.0 / (1.0 + scale * variance)


def _fuse_links(beliefs: list[Belief]) -> dict[str, LinkState]:
    all_link_ids: set[str] = set()
    for belief in beliefs:
        all_link_ids.update(belief.links.keys())

    fused: dict[str, LinkState] = {}
    for link_id in all_link_ids:
        attenuations: list[float] = []
        motion_probs: list[float] = []
        precisions: list[float] = []

        for belief in beliefs:
            link = belief.links.get(link_id)
            if link is None:
                continue
            precision = _precision_from_confidence(link.confidence)
            attenuations.append(link.attenuation)
            motion_probs.append(1.0 if link.motion else 0.0)
            precisions.append(precision)

        if not precisions:
            continue

        total_precision = sum(precisions)
        avg_attenuation = _weighted_mean(attenuations, precisions)
        avg_motion_prob = _weighted_mean(motion_probs, precisions)
        base_confidence = total_precision / (1.0 + total_precision)
        confidence = base_confidence * _agreement_penalty(attenuations, precisions, scale=2.5)

        fused[link_id] = LinkState(
            attenuation=max(avg_attenuation, 0.0),
            motion=avg_motion_prob >= 0.5,
            confidence=_clamp01(confidence),
        )
    return fused


def _device_confidence(belief: Belief, device_id: str, device: DeviceState) -> float:
    link = belief.links.get(device_id)
    if link is not None:
        confidence = link.confidence
    else:
        confidence = 0.35

    distance = device.estimated_distance
    if distance is not None and distance > 0:
        distance_confidence = 1.0 / (1.0 + (distance / 15.0))
        confidence = 0.6 * confidence + 0.4 * distance_confidence

    if device.moving:
        confidence *= 0.9

    return min(max(confidence, 0.05), 0.99)


def _fuse_devices(beliefs: list[Belief]) -> dict[str, DeviceState]:
    all_device_ids: set[str] = set()
    for belief in beliefs:
        all_device_ids.update(belief.devices.keys())

    fused: dict[str, DeviceState] = {}
    for device_id in all_device_ids:
        rssi_values: list[float] = []
        rssi_weights: list[float] = []
        distance_values: list[float] = []
        distance_weights: list[float] = []
        motion_values: list[float] = []
        motion_weights: list[float] = []

        for belief in beliefs:
            device = belief.devices.get(device_id)
            if device is None:
                continue
            precision = _precision_from_confidence(
                _device_confidence(belief, device_id, device),
            )
            rssi_values.append(device.rssi)
            rssi_weights.append(precision)
            motion_values.append(1.0 if device.moving else 0.0)
            motion_weights.append(precision)

            if device.estimated_distance is not None and device.estimated_distance > 0:
                # Long-range RF distances are less reliable; down-weight by squared range.
                range_scale = max(device.estimated_distance, 1.0) ** 2
                distance_values.append(device.estimated_distance)
                distance_weights.append(precision / range_scale)

        if not rssi_weights:
            continue

        avg_rssi = _weighted_mean(rssi_values, rssi_weights)
        avg_motion_prob = _weighted_mean(motion_values, motion_weights)
        avg_distance = (
            _weighted_mean(distance_values, distance_weights) if distance_weights else None
        )

        fused[device_id] = DeviceState(
            rssi=avg_rssi,
            estimated_distance=avg_distance,
            moving=avg_motion_prob >= 0.5,
        )

    return fused


def _zone_confidence(zone: ZoneBelief) -> float:
    certainty = max(abs(zone.occupied - 0.5), abs(zone.motion - 0.5)) * 2.0
    return min(max(0.2 + (0.8 * certainty), 0.05), 0.99)


def _fuse_zones(beliefs: list[Belief]) -> dict[str, ZoneBelief]:
    all_zone_ids: set[str] = set()
    for belief in beliefs:
        all_zone_ids.update(belief.zones.keys())

    fused: dict[str, ZoneBelief] = {}
    for zone_id in all_zone_ids:
        occupied_values: list[float] = []
        motion_values: list[float] = []
        weights: list[float] = []

        for belief in beliefs:
            zone = belief.zones.get(zone_id)
            if zone is None:
                continue
            precision = _precision_from_confidence(_zone_confidence(zone))
            occupied_values.append(zone.occupied)
            motion_values.append(zone.motion)
            weights.append(precision)

        if not weights:
            continue

        fused[zone_id] = ZoneBelief(
            occupied=_clamp01(_weighted_mean(occupied_values, weights)),
            motion=_clamp01(_weighted_mean(motion_values, weights)),
        )

    return fused
