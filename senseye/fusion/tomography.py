"""Radio Tomographic Imaging: spatial attenuation field from link measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinkMeasurement:
    p1: tuple[float, float]  # endpoint 1 position
    p2: tuple[float, float]  # endpoint 2 position
    excess_attenuation: float  # dB above free-space expectation


def _point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Distance from point (px,py) to the line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        # Degenerate segment (zero length)
        return np.hypot(px - x1, py - y1)

    # Project point onto line, clamp to [0,1]
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.hypot(px - proj_x, py - proj_y)


def reconstruct(
    links: list[LinkMeasurement],
    bounds: tuple[float, float, float, float],
    resolution: float = 0.5,
    influence_radius: float = 0.5,
) -> np.ndarray:
    """Build a 2D attenuation image via radio tomographic imaging.

    Args:
        links: list of link measurements with endpoint positions and excess attenuation
        bounds: (x_min, y_min, x_max, y_max) of the imaging area
        resolution: grid cell size in meters
        influence_radius: distance from link within which a cell is affected (meters).
                         Conceptually lambda/2 of the carrier frequency.

    Returns:
        2D numpy array where each cell contains estimated attenuation.
        Shape is (n_rows, n_cols) with rows along Y and columns along X.
    """
    x_min, y_min, x_max, y_max = bounds

    if x_max <= x_min or y_max <= y_min or resolution <= 0:
        return np.array([[]])

    if not links:
        n_cols = max(1, int((x_max - x_min) / resolution))
        n_rows = max(1, int((y_max - y_min) / resolution))
        return np.zeros((n_rows, n_cols))

    # Build grid of cell centers
    xs = np.arange(x_min + resolution / 2, x_max, resolution)
    ys = np.arange(y_min + resolution / 2, y_max, resolution)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((max(1, len(ys)), max(1, len(xs))))

    n_rows = len(ys)
    n_cols = len(xs)

    accum = np.zeros((n_rows, n_cols))
    count = np.zeros((n_rows, n_cols))

    for link in links:
        x1, y1 = link.p1
        x2, y2 = link.p2

        for r in range(n_rows):
            for c in range(n_cols):
                dist = _point_to_segment_distance(
                    xs[c], ys[r], x1, y1, x2, y2,
                )
                if dist <= influence_radius:
                    accum[r, c] += link.excess_attenuation
                    count[r, c] += 1.0

    # Normalize: average attenuation per cell
    mask = count > 0
    accum[mask] /= count[mask]

    return accum
