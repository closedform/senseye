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


def propagate_distances(
    acoustic: np.ndarray,
    max_hops: int = 3,
) -> np.ndarray:
    """Fill missing distances with shortest multi-hop acoustic paths.

    Daisy-chaining is only used for pairs without a direct acoustic measurement.
    """
    if acoustic.size == 0:
        return acoustic.copy()

    n = acoustic.shape[0]
    if acoustic.shape != (n, n):
        raise ValueError("acoustic matrix must be square")

    hop_limit = max(int(max_hops), 2)

    # Best known path length and hop count for each pair.
    dist = np.full((n, n), np.inf, dtype=np.float64)
    hops = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(hops, 0.0)

    direct_mask = acoustic > 0
    dist[direct_mask] = acoustic[direct_mask]
    hops[direct_mask] = 1.0

    for k in range(n):
        for i in range(n):
            dik = dist[i, k]
            hik = hops[i, k]
            if not np.isfinite(dik) or not np.isfinite(hik):
                continue
            for j in range(n):
                dkj = dist[k, j]
                hkj = hops[k, j]
                if not np.isfinite(dkj) or not np.isfinite(hkj):
                    continue
                cand_hops = hik + hkj
                if cand_hops <= 1 or cand_hops > hop_limit:
                    continue
                cand_dist = dik + dkj
                if cand_dist < dist[i, j]:
                    dist[i, j] = cand_dist
                    hops[i, j] = cand_hops

    propagated = acoustic.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if propagated[i, j] > 0:
                continue
            if not np.isfinite(dist[i, j]) or not np.isfinite(hops[i, j]):
                continue
            if hops[i, j] < 2 or hops[i, j] > hop_limit:
                continue
            propagated[i, j] = dist[i, j]
            propagated[j, i] = dist[i, j]

    return propagated
