from __future__ import annotations

from senseye.fusion.trilateration import trilaterate


def test_trilaterate_handles_single_outlier() -> None:
    # True point is near (1, 1)
    observations = [
        ((0.0, 0.0), 2**0.5),
        ((4.0, 0.0), 10**0.5),
        ((0.0, 4.0), 10**0.5),
        ((4.0, 4.0), 8.0),  # Outlier (true would be ~4.24)
    ]

    result = trilaterate(observations)
    assert result is not None
    (x, y), uncertainty = result
    assert abs(x - 1.0) < 0.6
    assert abs(y - 1.0) < 0.6
    assert uncertainty < 2.0


def test_trilaterate_requires_three_anchors() -> None:
    assert trilaterate([((0.0, 0.0), 1.0), ((1.0, 0.0), 1.0)]) is None
