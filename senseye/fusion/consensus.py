"""Weighted belief averaging across peers."""

from __future__ import annotations

from senseye.node.belief import Belief, DeviceState, LinkState, ZoneBelief


def fuse_beliefs(local: Belief, peer_beliefs: list[Belief]) -> Belief:
    """Fuse local belief with peer beliefs using confidence-weighted averaging.

    Handles partial overlap: not all peers see the same links/devices/zones.
    Agreement increases confidence; disagreement decreases it.
    """
    if not peer_beliefs:
        return local

    all_beliefs = [local] + peer_beliefs

    links = _fuse_links(all_beliefs)
    devices = _fuse_devices(all_beliefs)
    zones = _fuse_zones(all_beliefs)

    return Belief(
        node_id=local.node_id,
        links=links,
        devices=devices,
        zones=zones,
    )


def _fuse_links(beliefs: list[Belief]) -> dict[str, LinkState]:
    """Weighted average of link states across all beliefs."""
    # Collect all link IDs
    all_link_ids: set[str] = set()
    for b in beliefs:
        all_link_ids.update(b.links.keys())

    fused: dict[str, LinkState] = {}
    for link_id in all_link_ids:
        contributors: list[LinkState] = []
        for b in beliefs:
            if link_id in b.links:
                contributors.append(b.links[link_id])

        total_weight = sum(c.confidence for c in contributors)
        if total_weight == 0:
            # All zero-confidence: just take the first
            fused[link_id] = contributors[0]
            continue

        # Weighted average of attenuation
        avg_attenuation = sum(
            c.attenuation * c.confidence for c in contributors
        ) / total_weight

        # Weighted average of motion (as probability)
        motion_prob = sum(
            (1.0 if c.motion else 0.0) * c.confidence for c in contributors
        ) / total_weight

        # Confidence: agreement boosts, disagreement penalizes
        motion_values = [1.0 if c.motion else 0.0 for c in contributors]
        if len(motion_values) > 1:
            motion_mean = sum(motion_values) / len(motion_values)
            # Variance of motion votes: 0 = full agreement, 0.25 = max disagreement
            motion_var = sum((v - motion_mean) ** 2 for v in motion_values) / len(motion_values)
            # Scale: agreement (var=0) -> multiply by 1.2, disagreement (var=0.25) -> multiply by 0.6
            agreement_factor = 1.2 - 2.4 * motion_var
        else:
            agreement_factor = 1.0

        avg_confidence = total_weight / len(contributors)
        fused_confidence = min(max(avg_confidence * agreement_factor, 0.0), 1.0)

        fused[link_id] = LinkState(
            attenuation=avg_attenuation,
            motion=motion_prob > 0.5,
            confidence=fused_confidence,
        )

    return fused


def _fuse_devices(beliefs: list[Belief]) -> dict[str, DeviceState]:
    """Weighted average of device states (weight = number of samples, uniform)."""
    all_device_ids: set[str] = set()
    for b in beliefs:
        all_device_ids.update(b.devices.keys())

    fused: dict[str, DeviceState] = {}
    for dev_id in all_device_ids:
        contributors: list[DeviceState] = []
        for b in beliefs:
            if dev_id in b.devices:
                contributors.append(b.devices[dev_id])

        n = len(contributors)
        avg_rssi = sum(c.rssi for c in contributors) / n

        # Average distance, skipping None values
        dist_values = [c.estimated_distance for c in contributors if c.estimated_distance is not None]
        avg_distance = sum(dist_values) / len(dist_values) if dist_values else None

        # Motion: majority vote
        motion_votes = sum(1 for c in contributors if c.moving)
        moving = motion_votes > n / 2

        fused[dev_id] = DeviceState(
            rssi=avg_rssi,
            estimated_distance=avg_distance,
            moving=moving,
        )

    return fused


def _fuse_zones(beliefs: list[Belief]) -> dict[str, ZoneBelief]:
    """Weighted average of zone beliefs."""
    all_zone_ids: set[str] = set()
    for b in beliefs:
        all_zone_ids.update(b.zones.keys())

    fused: dict[str, ZoneBelief] = {}
    for zone_id in all_zone_ids:
        contributors: list[ZoneBelief] = []
        for b in beliefs:
            if zone_id in b.zones:
                contributors.append(b.zones[zone_id])

        n = len(contributors)
        avg_occupied = sum(c.occupied for c in contributors) / n
        avg_motion = sum(c.motion for c in contributors) / n

        fused[zone_id] = ZoneBelief(
            occupied=avg_occupied,
            motion=avg_motion,
        )

    return fused
