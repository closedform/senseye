"""Radio tomographic imaging from attenuation link measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinkMeasurement:
    p1: tuple[float, float]
    p2: tuple[float, float]
    excess_attenuation: float
    confidence: float = 1.0


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Distance from point (px, py) to segment (x1, y1) -> (x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = (dx * dx) + (dy * dy)
    if seg_len_sq <= 0:
        return float(np.hypot(px - x1, py - y1))

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + (t * dx)
    proj_y = y1 + (t * dy)
    return float(np.hypot(px - proj_x, py - proj_y))


def _adaptive_alpha(A: np.ndarray, row_weights: np.ndarray, n_cells: int) -> float:
    if A.size == 0:
        return 1.0
    weighted_A = A * np.sqrt(row_weights)[:, None]
    ata = weighted_A.T @ weighted_A
    ata_reg = ata + (1e-6 * np.eye(ata.shape[0]))
    try:
        cond = float(np.linalg.cond(ata_reg))
    except np.linalg.LinAlgError:
        cond = 1e8
    if not np.isfinite(cond):
        cond = 1e8
    sample_ratio = n_cells / max(A.shape[0], 1)
    alpha = 0.05 * sample_ratio * (1.0 + np.log10(max(cond, 1.0)))
    return float(min(max(alpha, 0.05), 5.0))


def reconstruct(
    links: list[LinkMeasurement],
    bounds: tuple[float, float, float, float],
    resolution: float = 0.5,
    influence_radius: float = 0.5,
    regularization: float | None = None,
) -> np.ndarray:
    """Reconstruct a 2D attenuation field with confidence-weighted ridge regression."""
    x_min, y_min, x_max, y_max = bounds
    if x_max <= x_min or y_max <= y_min or resolution <= 0:
        return np.array([[]])

    n_cols = max(1, int(np.ceil((x_max - x_min) / resolution)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / resolution)))

    if not links:
        return np.zeros((n_rows, n_cols))

    xs = x_min + ((np.arange(n_cols) + 0.5) * resolution)
    ys = y_min + ((np.arange(n_rows) + 0.5) * resolution)
    n_cells = n_rows * n_cols

    cell_coords = [(float(ys[r]), float(xs[c])) for r in range(n_rows) for c in range(n_cols)]

    rows: list[np.ndarray] = []
    targets: list[float] = []
    row_weights: list[float] = []

    kernel_sigma = max(influence_radius / 2.0, 1e-3)
    for link in links:
        x1, y1 = link.p1
        x2, y2 = link.p2

        row = np.zeros(n_cells, dtype=np.float64)
        lx_min = min(x1, x2) - influence_radius
        lx_max = max(x1, x2) + influence_radius
        ly_min = min(y1, y2) - influence_radius
        ly_max = max(y1, y2) + influence_radius

        for idx, (cy, cx) in enumerate(cell_coords):
            if not (lx_min <= cx <= lx_max and ly_min <= cy <= ly_max):
                continue
            dist = _point_to_segment_distance(cx, cy, x1, y1, x2, y2)
            if dist > influence_radius:
                continue
            row[idx] = np.exp(-(dist**2) / (2.0 * (kernel_sigma**2)))

        row_sum = float(np.sum(row))
        if row_sum <= 1e-6:
            continue
        row /= row_sum
        rows.append(row)
        targets.append(float(link.excess_attenuation))
        # Inverse-variance weighting consistent with consensus fusion:
        # variance = (1-c)/c, so precision (weight) = c/(1-c)
        c = min(max(float(link.confidence), 0.01), 0.99)
        row_weights.append(c / (1.0 - c))

    if not rows:
        return np.zeros((n_rows, n_cols))

    A = np.vstack(rows)
    b = np.array(targets, dtype=np.float64)
    w = np.array(row_weights, dtype=np.float64)

    alpha = regularization if regularization is not None else _adaptive_alpha(A, w, n_cells)
    weighted_A = A * np.sqrt(w)[:, None]
    weighted_b = b * np.sqrt(w)

    lhs = weighted_A.T @ weighted_A + (alpha * np.eye(n_cells))
    rhs = weighted_A.T @ weighted_b

    try:
        x = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)

    grid = x.reshape((n_rows, n_cols))
    grid[grid < 0] = 0.0
    return grid
