"""Distance matrix from acoustic time-of-flight measurements."""

from __future__ import annotations

import numpy as np

from senseye.node.acoustic import SPEED_OF_SOUND  # single source of truth


def build_distance_matrix(
    tof_measurements: dict[tuple[str, str], float],
    node_ids: list[str],
) -> np.ndarray:
    """Convert ToF measurements into a symmetric distance matrix.

    Args:
        tof_measurements: mapping of (source_id, target_id) -> time_of_flight in seconds
        node_ids: ordered list of node identifiers (defines row/col order)

    Returns:
        NxN numpy array of distances in meters. Unmeasured pairs are 0.0.
    """
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    matrix = np.zeros((n, n))

    for (src, tgt), tof in tof_measurements.items():
        i = idx.get(src)
        j = idx.get(tgt)
        if i is None or j is None:
            continue
        dist = tof * SPEED_OF_SOUND
        matrix[i, j] = dist
        matrix[j, i] = dist  # symmetric

    return matrix


def merge_distances(
    acoustic: np.ndarray,
    rf: np.ndarray,
    has_acoustic: np.ndarray | None = None,
) -> np.ndarray:
    """Merge acoustic and RF distance matrices, preferring acoustic where available.

    Args:
        acoustic: NxN distance matrix from acoustic ranging
        rf: NxN distance matrix from RF-based estimation
        has_acoustic: NxN boolean mask. True where acoustic measurement exists.
                     If None, inferred from non-zero entries in acoustic matrix.

    Returns:
        NxN merged distance matrix.
    """
    if has_acoustic is None:
        has_acoustic = acoustic > 0

    return np.where(has_acoustic, acoustic, rf)
