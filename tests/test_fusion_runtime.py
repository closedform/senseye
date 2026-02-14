from __future__ import annotations

import numpy as np

from senseye.fusion.runtime import estimate_device_positions, reconstruct_attenuation_grid
from senseye.node.belief import Belief, DeviceState, LinkState


def test_estimate_device_positions_trilaterates_from_three_anchors() -> None:
    node_positions = {
        "a": (0.0, 0.0),
        "b": (3.0, 0.0),
        "c": (0.0, 4.0),
    }
    beliefs = [
        Belief(
            node_id="a",
            devices={"phone": DeviceState(rssi=-55.0, estimated_distance=2**0.5, moving=False)},
        ),
        Belief(
            node_id="b",
            devices={"phone": DeviceState(rssi=-57.0, estimated_distance=5**0.5, moving=False)},
        ),
        Belief(
            node_id="c",
            devices={"phone": DeviceState(rssi=-58.0, estimated_distance=10**0.5, moving=False)},
        ),
    ]

    estimates = estimate_device_positions(beliefs, node_positions)
    assert "phone" in estimates
    (x, y), uncertainty = estimates["phone"]
    assert abs(x - 1.0) < 0.2
    assert abs(y - 1.0) < 0.2
    assert uncertainty < 0.5


def test_reconstruct_attenuation_grid_uses_node_links() -> None:
    node_positions = {
        "a": (0.0, 0.0),
        "b": (3.0, 0.0),
        "c": (0.0, 3.0),
    }
    beliefs = [
        Belief(
            node_id="a",
            links={
                "b": LinkState(attenuation=6.0, motion=True, confidence=1.0),
                "c": LinkState(attenuation=4.0, motion=False, confidence=1.0),
            },
        ),
        Belief(
            node_id="b",
            links={"c": LinkState(attenuation=5.0, motion=True, confidence=1.0)},
        ),
    ]

    grid = reconstruct_attenuation_grid(
        beliefs=beliefs,
        node_positions=node_positions,
        bounds=(-1.0, -1.0, 4.0, 4.0),
        resolution=0.5,
    )
    assert grid.size > 0
    assert float(np.max(grid)) > 0.0
