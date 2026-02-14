"""Device positioning via trilateration (weighted least squares)."""

from __future__ import annotations

import numpy as np


def trilaterate(
    observations: list[tuple[tuple[float, float], float]],
) -> tuple[tuple[float, float], float] | None:
    """Estimate a device's position from distance observations to known anchors.

    Args:
        observations: list of ((x, y), distance) pairs where (x,y) is
                      the anchor position and distance is estimated range.

    Returns:
        ((est_x, est_y), uncertainty_radius) or None if fewer than 3 observations
        or the system is degenerate (collinear anchors).
    """
    if len(observations) < 3:
        return None

    # Linearize: subtract first observation's equation from all others.
    # Circle eq: (x - xi)^2 + (y - yi)^2 = di^2
    # Expanding and subtracting eq_0 from eq_i gives a linear system:
    #   2*(x0-xi)*x + 2*(y0-yi)*y = d_i^2 - d_0^2 - xi^2 + x0^2 - yi^2 + y0^2
    x0, y0 = observations[0][0]
    d0 = observations[0][1]

    n = len(observations) - 1
    A = np.zeros((n, 2))
    b = np.zeros(n)

    for i in range(n):
        xi, yi = observations[i + 1][0]
        di = observations[i + 1][1]

        A[i, 0] = 2.0 * (x0 - xi)
        A[i, 1] = 2.0 * (y0 - yi)
        b[i] = di * di - d0 * d0 - xi * xi + x0 * x0 - yi * yi + y0 * y0

    # Check for degeneracy (collinear points -> rank < 2)
    if np.linalg.matrix_rank(A) < 2:
        return None

    # Least squares solution
    result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    est_x, est_y = float(result[0]), float(result[1])

    # Uncertainty: RMS of residual distances
    errors = []
    for (xi, yi), di in observations:
        est_dist = np.hypot(est_x - xi, est_y - yi)
        errors.append(abs(est_dist - di))
    uncertainty = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    return (est_x, est_y), uncertainty
