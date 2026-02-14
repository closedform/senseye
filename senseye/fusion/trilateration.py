"""Device positioning via robust weighted Gauss-Newton trilateration."""

from __future__ import annotations

from itertools import combinations

import numpy as np

RangeObservation = tuple[tuple[float, float], float]
_EPS = 1e-9


def _initial_guess(observations: list[RangeObservation]) -> tuple[float, float]:
    """Linearized least-squares initialization."""
    anchors = np.array([obs[0] for obs in observations], dtype=np.float64)
    ranges = np.array([obs[1] for obs in observations], dtype=np.float64)

    x0, y0 = anchors[0]
    d0 = ranges[0]
    a_rows: list[list[float]] = []
    b_vals: list[float] = []

    for idx in range(1, len(observations)):
        xi, yi = anchors[idx]
        di = ranges[idx]
        a_rows.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b_vals.append((d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2))

    if len(a_rows) >= 2:
        A = np.array(a_rows, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)
        try:
            solution, *_ = np.linalg.lstsq(A, b, rcond=None)
            return float(solution[0]), float(solution[1])
        except np.linalg.LinAlgError:
            pass

    centroid = anchors.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def _range_sigma(distance: float) -> float:
    """Range uncertainty model (meters)."""
    # Error typically grows with distance in RSSI-derived ranging.
    return max(0.35, 0.08 * distance + 0.2)


def _solve_position(
    observations: list[RangeObservation],
    x: float,
    y: float,
    max_iters: int,
    tolerance: float,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    for _ in range(max_iters):
        jacobian_rows: list[list[float]] = []
        residuals: list[float] = []
        row_weights: list[float] = []
        base_weights: list[float] = []

        for (ax, ay), measured_distance in observations:
            dx = x - ax
            dy = y - ay
            predicted_distance = max(float(np.hypot(dx, dy)), _EPS)
            residual = predicted_distance - measured_distance

            sigma = _range_sigma(measured_distance)
            base_weight = 1.0 / (sigma * sigma)
            base_weights.append(base_weight)

            # Tukey biweight strongly suppresses gross outliers.
            cutoff = 2.5 * sigma
            abs_residual = abs(residual)
            if abs_residual >= cutoff:
                robust_weight = 0.0
            else:
                ratio = abs_residual / cutoff
                robust_weight = (1.0 - (ratio * ratio)) ** 2
            weight = base_weight * robust_weight

            jacobian_rows.append([dx / predicted_distance, dy / predicted_distance])
            residuals.append(residual)
            row_weights.append(weight)

        J = np.array(jacobian_rows, dtype=np.float64)
        r = np.array(residuals, dtype=np.float64)
        w = np.array(row_weights, dtype=np.float64)
        if np.max(w) <= 1e-12:
            w = np.array(base_weights, dtype=np.float64)
        sqrt_w = np.sqrt(w)

        Jw = J * sqrt_w[:, None]
        rw = r * sqrt_w

        lhs = Jw.T @ Jw
        rhs = Jw.T @ rw

        try:
            delta = np.linalg.solve(lhs + (1e-6 * np.eye(2)), rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(lhs) @ rhs

        x -= float(delta[0])
        y -= float(delta[1])

        if float(np.linalg.norm(delta)) < tolerance:
            break

    final_residuals = []
    normalized_residuals = []
    for (ax, ay), measured_distance in observations:
        predicted_distance = float(np.hypot(x - ax, y - ay))
        residual = predicted_distance - measured_distance
        final_residuals.append(residual)
        normalized_residuals.append(abs(residual) / _range_sigma(measured_distance))

    return (
        (x, y),
        np.array(final_residuals, dtype=np.float64),
        np.array(normalized_residuals, dtype=np.float64),
    )


def _residual_arrays(
    position: tuple[float, float],
    observations: list[RangeObservation],
) -> tuple[np.ndarray, np.ndarray]:
    x, y = position
    residuals: list[float] = []
    normalized_residuals: list[float] = []
    for (ax, ay), measured_distance in observations:
        predicted_distance = float(np.hypot(x - ax, y - ay))
        residual = predicted_distance - measured_distance
        residuals.append(residual)
        normalized_residuals.append(abs(residual) / _range_sigma(measured_distance))
    return np.array(residuals, dtype=np.float64), np.array(normalized_residuals, dtype=np.float64)


def trilaterate(
    observations: list[RangeObservation],
    max_iters: int = 12,
    tolerance: float = 1e-4,
) -> tuple[tuple[float, float], float] | None:
    """Estimate position from range observations.

    Returns:
        ((x, y), uncertainty_rmse) or None when underdetermined/unstable.
    """
    valid = [
        (anchor, distance)
        for anchor, distance in observations
        if np.isfinite(distance) and distance > 0
    ]
    if len(valid) < 3:
        return None

    candidate_index_sets: set[tuple[int, ...]] = {tuple(range(len(valid)))}
    if len(valid) > 3:
        indices = tuple(range(len(valid)))
        candidate_index_sets.update(combinations(indices, len(valid) - 1))
        if len(valid) <= 6:
            candidate_index_sets.update(combinations(indices, 3))

    best_position: tuple[float, float] | None = None
    best_norm_residuals: np.ndarray | None = None
    best_score = float("inf")
    best_inlier_count = -1

    for index_set in sorted(candidate_index_sets):
        subset = [valid[idx] for idx in index_set]
        anchors = np.array([obs[0] for obs in subset], dtype=np.float64)
        centroid = anchors.mean(axis=0)
        seeds = [
            _initial_guess(subset),
            (float(centroid[0]), float(centroid[1])),
        ]

        for seed_x, seed_y in seeds:
            (x, y), _, _ = _solve_position(
                subset,
                seed_x,
                seed_y,
                max_iters=max_iters,
                tolerance=tolerance,
            )
            _, norm_residuals = _residual_arrays((x, y), valid)
            inlier_count = int(np.sum(norm_residuals <= 2.5))
            score = float(np.mean(np.minimum(norm_residuals**2, 9.0)))
            if (
                inlier_count > best_inlier_count
                or (inlier_count == best_inlier_count and score < best_score)
            ):
                best_inlier_count = inlier_count
                best_score = score
                best_position = (x, y)
                best_norm_residuals = norm_residuals

    if best_position is None or best_norm_residuals is None:
        return None

    inliers = [obs for obs, keep in zip(valid, best_norm_residuals <= 2.5) if bool(keep)]
    solve_set = inliers if len(inliers) >= 3 else valid
    (x, y), _, _ = _solve_position(
        solve_set,
        best_position[0],
        best_position[1],
        max_iters=max_iters,
        tolerance=tolerance,
    )
    residuals, _ = _residual_arrays((x, y), solve_set)

    weights = np.array([1.0 / (_range_sigma(distance) ** 2) for _, distance in solve_set])
    rmse = float(np.sqrt(np.sum(weights * (residuals**2)) / max(np.sum(weights), _EPS)))

    # Sanity gate for divergent solutions.
    if not np.isfinite(rmse) or rmse > 8.0:
        return None

    return (float(x), float(y)), rmse
