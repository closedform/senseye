from __future__ import annotations

from senseye.node.belief import Belief


def test_belief_serializes_acoustic_ranges() -> None:
    belief = Belief(
        node_id="node-a",
        acoustic_ranges={"node-b": 2.75},
    )

    payload = belief.to_dict()
    restored = Belief.from_dict(payload)
    assert restored.acoustic_ranges["node-b"] == 2.75
