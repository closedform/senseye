from __future__ import annotations

from senseye.fusion.consensus import fuse_beliefs
from senseye.node.belief import Belief, DeviceState, LinkState


def test_consensus_prefers_high_confidence_link_estimate() -> None:
    local = Belief(
        node_id="local",
        links={"peer-x": LinkState(attenuation=9.0, motion=True, confidence=0.9)},
    )
    peer = Belief(
        node_id="peer",
        links={"peer-x": LinkState(attenuation=2.0, motion=False, confidence=0.2)},
    )

    fused = fuse_beliefs(local, {"peer": peer})
    assert fused.links["peer-x"].attenuation > 6.0
    assert fused.links["peer-x"].motion is True


def test_consensus_weights_devices_using_link_confidence() -> None:
    local = Belief(
        node_id="local",
        links={"phone": LinkState(attenuation=3.0, motion=False, confidence=0.9)},
        devices={"phone": DeviceState(rssi=-52.0, estimated_distance=2.0, moving=False)},
    )
    peer = Belief(
        node_id="peer",
        links={"phone": LinkState(attenuation=3.0, motion=True, confidence=0.1)},
        devices={"phone": DeviceState(rssi=-80.0, estimated_distance=12.0, moving=True)},
    )

    fused = fuse_beliefs(local, {"peer": peer})
    assert fused.devices["phone"].rssi > -60.0
    assert fused.devices["phone"].estimated_distance is not None
    assert fused.devices["phone"].estimated_distance < 6.0
