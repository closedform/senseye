from __future__ import annotations

import numpy as np

from senseye.fusion.acoustic_range import propagate_distances


def test_propagate_distances_fills_two_hop_paths() -> None:
    direct = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    propagated = propagate_distances(direct, max_hops=3)
    assert abs(float(propagated[0, 2]) - 5.0) < 1e-6
    assert abs(float(propagated[2, 0]) - 5.0) < 1e-6


def test_propagate_distances_preserves_direct_edges() -> None:
    direct = np.array(
        [
            [0.0, 1.2, 0.0],
            [1.2, 0.0, 0.8],
            [0.0, 0.8, 0.0],
        ],
        dtype=np.float64,
    )

    propagated = propagate_distances(direct, max_hops=3)
    assert abs(float(propagated[0, 1]) - 1.2) < 1e-6
    assert abs(float(propagated[1, 2]) - 0.8) < 1e-6
