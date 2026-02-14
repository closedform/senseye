"""Map calibration helpers: scan observations -> static floorplan."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from senseye.config import AcousticMode, SenseyeConfig
from senseye.fusion.acoustic_range import SPEED_OF_SOUND, build_distance_matrix, merge_distances
from senseye.fusion.tomography import LinkMeasurement
from senseye.fusion.tomography import reconstruct as reconstruct_tomography
from senseye.mapping.static.floorplan import FloorPlan
from senseye.mapping.static.layout import anchor_positions, mds_positions
from senseye.mapping.static.topology import Room, RoomGraph, infer_rooms_from_nodes
from senseye.mapping.static.walls import WallSegment, classify_material, detect_walls
from senseye.node.acoustic import echo_profile, generate_chirp
from senseye.node.inference import PATHLOSS_A, PATHLOSS_N
from senseye.node.scanner import Observation, SignalType, scan_all

log = logging.getLogger(__name__)

_MIN_DISTANCE = 0.5
_MAX_DISTANCE = 25.0

# Free-space exponent for wall detection: using n=2 (theoretical free-space)
# makes all indoor propagation effects visible as excess attenuation,
# maximizing sensitivity to structural obstructions during calibration.
_PATHLOSS_N_FREESPACE = 2.0
_TOMOGRAPHY_RESOLUTION = 0.5
_MAX_TOMOGRAPHY_WALLS = 40


@dataclass
class _DeviceSummary:
    rssi_sum: float = 0.0
    count: int = 0
    label: str | None = None

    @property
    def avg_rssi(self) -> float:
        if self.count == 0:
            return -90.0
        return self.rssi_sum / self.count

    def add(self, obs: Observation) -> None:
        self.rssi_sum += obs.rssi
        self.count += 1
        if self.label:
            return
        for key in ("name", "ssid"):
            value = obs.metadata.get(key)
            if isinstance(value, str) and value.strip():
                self.label = value.strip()
                return


def _estimate_distance_from_rssi(
    rssi: float,
    n: float = PATHLOSS_N,
    a: float = PATHLOSS_A,
) -> float:
    distance = 10.0 ** ((-rssi - a) / (10.0 * n))
    return float(min(max(distance, _MIN_DISTANCE), _MAX_DISTANCE))


def _summarize_observations(observations: list[Observation]) -> dict[str, _DeviceSummary]:
    summary: dict[str, _DeviceSummary] = {}
    for obs in observations:
        if obs.signal_type == SignalType.ACOUSTIC:
            continue
        item = summary.setdefault(obs.device_id, _DeviceSummary())
        item.add(obs)
    return summary


def _acoustic_distances(observations: list[Observation]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for obs in observations:
        if obs.signal_type != SignalType.ACOUSTIC:
            continue
        raw_distance = obs.metadata.get("distance_m")
        if not isinstance(raw_distance, int | float):
            continue
        if raw_distance <= 0:
            continue
        sums[obs.device_id] = sums.get(obs.device_id, 0.0) + float(raw_distance)
        counts[obs.device_id] = counts.get(obs.device_id, 0) + 1
    return {device_id: sums[device_id] / counts[device_id] for device_id in sums}


def _rf_distance_matrix(
    node_id: str,
    node_ids: list[str],
    dist_to_self: dict[str, float],
) -> np.ndarray:
    n = len(node_ids)
    matrix = np.zeros((n, n))
    for idx, current_id in enumerate(node_ids):
        if current_id == node_id:
            continue
        dist = dist_to_self[current_id]
        matrix[0, idx] = dist
        matrix[idx, 0] = dist

    for i in range(1, n):
        for j in range(i + 1, n):
            di = float(matrix[0, i])
            dj = float(matrix[0, j])
            # Expected distance under uniform angular prior:
            # E[dij^2] = di^2 + dj^2 when angle is uniform over [0, 2pi]
            dij = min(math.sqrt(di * di + dj * dj), _MAX_DISTANCE)
            matrix[i, j] = dij
            matrix[j, i] = dij
    return matrix


def _extract_tomography_walls(
    grid: np.ndarray,
    bounds: tuple[float, float, float, float],
    resolution: float,
    max_segments: int = _MAX_TOMOGRAPHY_WALLS,
) -> list[WallSegment]:
    if grid.size == 0:
        return []

    active_values = grid[grid > 0]
    if active_values.size == 0:
        return []

    threshold = max(3.0, float(np.quantile(active_values, 0.8)))
    x_min, y_min, _, _ = bounds
    half = resolution * 0.45

    ranked_cells: list[tuple[float, int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = float(grid[r, c])
            if value >= threshold:
                ranked_cells.append((value, r, c))
    ranked_cells.sort(reverse=True)

    walls: list[WallSegment] = []
    for value, r, c in ranked_cells[:max_segments]:
        cx = x_min + (c + 0.5) * resolution
        cy = y_min + (r + 0.5) * resolution
        walls.append(WallSegment(
            start=(cx - half, cy),
            end=(cx + half, cy),
            attenuation_db=value,
            material=classify_material(value),
        ))
    return walls


def _dedupe_walls(walls: list[WallSegment]) -> list[WallSegment]:
    unique: list[WallSegment] = []
    seen: set[tuple[tuple[float, float], tuple[float, float], str]] = set()
    for wall in walls:
        start = (round(wall.start[0], 2), round(wall.start[1], 2))
        end = (round(wall.end[0], 2), round(wall.end[1], 2))
        ordered = (start, end) if start <= end else (end, start)
        key = (ordered[0], ordered[1], wall.material)
        if key in seen:
            continue
        seen.add(key)
        unique.append(wall)
    return unique


def _derive_bounds(
    node_positions: dict[str, tuple[float, float]],
    walls: list[WallSegment],
    acoustic_extent: float | None,
) -> tuple[float, float, float, float]:
    xs = [p[0] for p in node_positions.values()]
    ys = [p[1] for p in node_positions.values()]
    for wall in walls:
        xs.extend([wall.start[0], wall.end[0]])
        ys.extend([wall.start[1], wall.end[1]])

    if not xs or not ys:
        return (-2.0, -2.0, 2.0, 2.0)

    margin = 1.5
    if acoustic_extent is not None and acoustic_extent > 0:
        margin = max(margin, min(acoustic_extent, 6.0))

    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin

    if (x_max - x_min) < 2.0:
        x_min -= 1.0
        x_max += 1.0
    if (y_max - y_min) < 2.0:
        y_min -= 1.0
        y_max += 1.0

    return (x_min, y_min, x_max, y_max)


def build_floorplan_from_observations(
    node_id: str,
    node_name: str,
    observations: list[Observation],
    peer_ids: Iterable[str] = (),
    max_devices: int = 8,
    acoustic_extent: float | None = None,
) -> tuple[FloorPlan, dict[str, float]]:
    """Build a best-effort floorplan from current RF observations."""
    summaries = _summarize_observations(observations)
    acoustic_distance_by_device = _acoustic_distances(observations)
    ordered_devices = sorted(
        summaries.keys(),
        key=lambda device_id: summaries[device_id].avg_rssi,
        reverse=True,
    )

    known_peers = sorted({peer_id for peer_id in peer_ids if peer_id and peer_id != node_id})
    selected_devices: list[str] = []
    for device_id in ordered_devices:
        if device_id in known_peers:
            continue
        selected_devices.append(device_id)
        if len(selected_devices) >= max_devices:
            break

    acoustic_only_devices = [
        device_id
        for device_id in acoustic_distance_by_device
        if device_id not in known_peers and device_id not in selected_devices
    ]
    candidates = known_peers + selected_devices + acoustic_only_devices
    node_ids = [node_id] + candidates

    dist_to_self: dict[str, float] = {}
    for idx, peer_id in enumerate(known_peers):
        # Peer node distances are unknown here; keep a stable seed spacing.
        dist_to_self[peer_id] = 2.5 + idx * 0.5
    for device_id in selected_devices:
        dist_to_self[device_id] = _estimate_distance_from_rssi(summaries[device_id].avg_rssi)
    for device_id, acoustic_distance in acoustic_distance_by_device.items():
        if device_id in dist_to_self:
            dist_to_self[device_id] = float(
                min(max(acoustic_distance, _MIN_DISTANCE), _MAX_DISTANCE)
            )

    distances_rf = _rf_distance_matrix(
        node_id=node_id,
        node_ids=node_ids,
        dist_to_self=dist_to_self,
    )
    acoustic_tof = {
        (node_id, device_id): distance / SPEED_OF_SOUND
        for device_id, distance in acoustic_distance_by_device.items()
        if device_id in node_ids
    }
    distances_acoustic = build_distance_matrix(acoustic_tof, node_ids)
    distances = merge_distances(acoustic=distances_acoustic, rf=distances_rf)

    positions = anchor_positions(mds_positions(distances), anchors={0: (0.0, 0.0)})
    node_positions: dict[str, tuple[float, float]] = {}
    for idx, current_node_id in enumerate(node_ids):
        x, y = positions[idx]
        node_positions[current_node_id] = (float(x), float(y))

    link_attenuations: dict[tuple[str, str], float] = {}
    for device_id in selected_devices:
        rssi = summaries[device_id].avg_rssi
        est_distance = dist_to_self[device_id]
        expected = -(
            10.0
            * _PATHLOSS_N_FREESPACE
            * math.log10(max(est_distance, _MIN_DISTANCE))
            + PATHLOSS_A
        )
        attenuation = max(0.0, expected - rssi)
        if attenuation > 0:
            link_attenuations[(node_id, device_id)] = attenuation

    walls = detect_walls(node_positions=node_positions, link_attenuations=link_attenuations)
    provisional_bounds = _derive_bounds(
        node_positions=node_positions,
        walls=walls,
        acoustic_extent=acoustic_extent,
    )
    tomography_links: list[LinkMeasurement] = []
    for (src, tgt), attenuation in link_attenuations.items():
        src_pos = node_positions.get(src)
        tgt_pos = node_positions.get(tgt)
        if src_pos is None or tgt_pos is None:
            continue
        confidence = min(max(0.4 + (attenuation / 20.0), 0.05), 1.0)
        tomography_links.append(LinkMeasurement(
            p1=src_pos,
            p2=tgt_pos,
            excess_attenuation=attenuation,
            confidence=confidence,
        ))
    attenuation_grid = reconstruct_tomography(
        links=tomography_links,
        bounds=provisional_bounds,
        resolution=_TOMOGRAPHY_RESOLUTION,
    )
    tomo_walls = _extract_tomography_walls(
        grid=attenuation_grid,
        bounds=provisional_bounds,
        resolution=_TOMOGRAPHY_RESOLUTION,
    )
    walls = _dedupe_walls(walls + tomo_walls)
    rooms = infer_rooms_from_nodes(node_positions=node_positions, wall_segments=walls)
    if not rooms.rooms:
        points = np.array(list(node_positions.values()))
        center = points.mean(axis=0) if len(points) else np.array([0.0, 0.0])
        rooms = RoomGraph(
            rooms=[
                Room(
                    name="room_0",
                    center=(float(center[0]), float(center[1])),
                    node_ids=list(node_positions.keys()),
                ),
            ],
            connections=[],
        )

    labels: dict[str, str] = {node_id: node_name}
    for peer_id in known_peers:
        labels[peer_id] = f"peer-{peer_id[:6]}"
    for device_id in selected_devices:
        summary = summaries[device_id]
        labels[device_id] = summary.label or device_id[:8]
    for room in rooms.rooms:
        labels.setdefault(room.name, room.name.replace("_", " "))

    bounds = _derive_bounds(
        node_positions=node_positions,
        walls=walls,
        acoustic_extent=acoustic_extent,
    )
    baseline_rssi = {
        device_id: summaries[device_id].avg_rssi for device_id in selected_devices
    }

    plan = FloorPlan(
        node_positions=node_positions,
        wall_segments=walls,
        rooms=rooms,
        bounds=bounds,
        labels=labels,
        attenuation_grid=attenuation_grid.tolist(),
        attenuation_resolution=_TOMOGRAPHY_RESOLUTION,
    )
    return plan, baseline_rssi


async def calibrate_floorplan(
    config: SenseyeConfig,
    peer_ids: Iterable[str] = (),
    scans: int = 3,
    force_acoustic: bool = False,
) -> tuple[FloorPlan, dict[str, float]]:
    """Run active calibration using local scans and optional acoustic echo."""
    scan_count = max(scans, 1)
    observations: list[Observation] = []
    for _ in range(scan_count):
        observations.extend(
            await scan_all(
                wifi=config.wifi_enabled,
                ble=config.ble_enabled,
                ble_duration=config.ble_duration,
            )
        )

    acoustic_extent: float | None = None
    should_ping = force_acoustic or config.acoustic_mode == AcousticMode.INTERVAL
    if should_ping:
        try:
            chirp = generate_chirp(
                freq_start=config.chirp_freq_start,
                freq_end=config.chirp_freq_end,
                duration=config.chirp_duration,
            )
            profile = await echo_profile(chirp=chirp)
            if profile is not None and profile.distance is not None:
                acoustic_extent = profile.distance
                observations.append(Observation(
                    device_id=f"acoustic:echo:{config.node_id}",
                    rssi=float(profile.distance),
                    timestamp=profile.timestamp,
                    signal_type=SignalType.ACOUSTIC,
                    metadata={"distance_m": profile.distance},
                ))
        except Exception:
            log.debug("acoustic calibration unavailable", exc_info=True)

    return build_floorplan_from_observations(
        node_id=config.node_id,
        node_name=config.node_name,
        observations=observations,
        peer_ids=peer_ids,
        acoustic_extent=acoustic_extent,
    )
