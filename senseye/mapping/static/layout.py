"""MDS self-localization of fixed nodes from pairwise distances."""

from __future__ import annotations

import numpy as np


def mds_positions(distances: np.ndarray) -> np.ndarray:
    """Classical MDS: squared distance matrix -> 2D positions.

    Returns Nx2 array of (x, y) coordinates. Positions are relative
    (centered at origin, arbitrary rotation/reflection).
    """
    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    if n == 2:
        d = distances[0, 1]
        return np.array([[0.0, 0.0], [d, 0.0]])

    # Squared distance matrix
    D2 = distances**2

    # Double centering: B = -0.5 * J * D2 * J, where J = I - (1/n)*11'
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Symmetrize to handle floating-point asymmetry
    B = (B + B.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # eigh returns ascending order; take the two largest
    idx = np.argsort(eigenvalues)[::-1]
    top2_vals = eigenvalues[idx[:2]]
    top2_vecs = eigenvectors[:, idx[:2]]

    # Clamp negative eigenvalues to zero (can happen with noisy distances)
    top2_vals = np.maximum(top2_vals, 0.0)

    positions = top2_vecs * np.sqrt(top2_vals)[np.newaxis, :]
    return positions


def anchor_positions(
    positions: np.ndarray,
    anchors: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Transform MDS positions to match known anchor coordinates.

    anchors: {node_index: (x, y)} — 1 or 2 anchors supported.
      - 1 anchor: translate so that node sits at given (x, y).
      - 2 anchors: translate + rotate + optional reflect to best fit both.
    """
    n = positions.shape[0]
    if n == 0 or not anchors:
        return positions.copy()

    result = positions.copy()
    anchor_indices = sorted(anchors.keys())

    if len(anchor_indices) >= 2:
        i, j = anchor_indices[0], anchor_indices[1]
        # Current vector from anchor i to anchor j
        src = result[j] - result[i]
        # Target vector
        tgt = np.array(anchors[j]) - np.array(anchors[i])

        src_len = np.linalg.norm(src)
        tgt_len = np.linalg.norm(tgt)

        if src_len > 1e-12 and tgt_len > 1e-12:
            # Compute rotation angle
            angle_src = np.arctan2(src[1], src[0])
            angle_tgt = np.arctan2(tgt[1], tgt[0])
            theta = angle_tgt - angle_src

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            # Rotate around anchor i
            result = (result - result[i]) @ rot.T

            # Check if reflection is needed: compare the rotated j with target j
            # relative to anchor i
            rotated_vec = result[j] - result[i]
            target_vec = np.array(anchors[j]) - np.array(anchors[i])

            # If dot product with perpendicular is large, try reflecting across the axis
            # defined by the two anchors
            error_no_flip = np.linalg.norm(rotated_vec - target_vec)
            # Reflect across x-axis (flip y), then re-rotate — simpler: just flip y
            reflected = result.copy()
            reflected[:, 1] = -reflected[:, 1]
            error_flip = np.linalg.norm(reflected[j] - reflected[i] - target_vec)

            if error_flip < error_no_flip:
                result = reflected

            # Translate to anchor i's target position
            result = result - result[i] + np.array(anchors[i])
        else:
            # Degenerate: coincident points, just translate
            i = anchor_indices[0]
            offset = np.array(anchors[i]) - result[i]
            result += offset
    else:
        # Single anchor: translate only
        i = anchor_indices[0]
        offset = np.array(anchors[i]) - result[i]
        result += offset

    return result
