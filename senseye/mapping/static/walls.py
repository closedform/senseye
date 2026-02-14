"""Wall inference from excess RF attenuation between node pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WallSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    attenuation_db: float
    material: str


def classify_material(attenuation_db: float) -> str:
    """Classify wall material from attenuation in dB."""
    if attenuation_db < 3.0:
        return "open"
    elif attenuation_db < 5.0:
        return "drywall"
    elif attenuation_db < 8.0:
        return "wood"
    elif attenuation_db < 12.0:
        return "brick"
    else:
        return "concrete"


# Minimum attenuation (dB) to consider a link as having a wall
_WALL_THRESHOLD = 3.0

# Wall segment half-length per dB of attenuation (meters)
_LENGTH_PER_DB = 0.15


def detect_walls(
    node_positions: dict[str, tuple[float, float]],
    link_attenuations: dict[tuple[str, str], float],
) -> list[WallSegment]:
    """Detect wall segments from node positions and link attenuations.

    For each link with attenuation >= threshold, places a wall segment
    perpendicular to the link at its midpoint. Segment length is proportional
    to attenuation.
    """
    walls: list[WallSegment] = []

    for (id_a, id_b), att_db in link_attenuations.items():
        if att_db < _WALL_THRESHOLD:
            continue
        if id_a not in node_positions or id_b not in node_positions:
            continue

        pa = np.array(node_positions[id_a])
        pb = np.array(node_positions[id_b])
        midpoint = (pa + pb) / 2.0

        # Direction from a to b
        direction = pb - pa
        link_len = np.linalg.norm(direction)
        if link_len < 1e-12:
            continue

        # Perpendicular (rotate 90 degrees)
        perp = np.array([-direction[1], direction[0]]) / link_len

        # Wall half-length proportional to attenuation
        half_len = att_db * _LENGTH_PER_DB
        start = midpoint - perp * half_len
        end = midpoint + perp * half_len

        material = classify_material(att_db)
        walls.append(WallSegment(
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            attenuation_db=att_db,
            material=material,
        ))

    return walls
