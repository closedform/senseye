from __future__ import annotations

from senseye.calibration import build_floorplan_from_observations
from senseye.node.scanner import Observation, SignalType


def _obs(device_id: str, rssi: float, **metadata: str) -> Observation:
    return Observation(
        device_id=device_id,
        rssi=rssi,
        timestamp=0.0,
        signal_type=SignalType.WIFI,
        metadata=metadata,
    )


def test_build_floorplan_includes_node_bounds_labels_and_baseline() -> None:
    observations = [
        _obs("aa:aa:aa:aa:aa:aa", -45.0, ssid="router"),
        _obs("bb:bb:bb:bb:bb:bb", -62.0, name="watch"),
        _obs("cc:cc:cc:cc:cc:cc", -70.0),
    ]
    plan, baseline = build_floorplan_from_observations(
        node_id="node-local",
        node_name="living-pi",
        observations=observations,
        peer_ids=["peer-1"],
    )

    assert "node-local" in plan.node_positions
    assert "peer-1" in plan.node_positions
    assert len(plan.rooms.rooms) >= 1
    assert baseline
    assert plan.labels["node-local"] == "living-pi"
    assert plan.labels["aa:aa:aa:aa:aa:aa"] == "router"

    x_min, y_min, x_max, y_max = plan.bounds
    assert x_max > x_min
    assert y_max > y_min


def test_build_floorplan_handles_empty_observations() -> None:
    plan, baseline = build_floorplan_from_observations(
        node_id="node-local",
        node_name="living-pi",
        observations=[],
    )

    assert set(plan.node_positions) == {"node-local"}
    assert baseline == {}
    assert len(plan.rooms.rooms) == 1


def test_build_floorplan_prefers_acoustic_distance_when_present() -> None:
    observations = [
        _obs("device-a", -80.0, ssid="far-ap"),
        Observation(
            device_id="device-a",
            rssi=2.0,
            timestamp=0.0,
            signal_type=SignalType.ACOUSTIC,
            metadata={"distance_m": 2.0},
        ),
    ]
    plan, _baseline = build_floorplan_from_observations(
        node_id="node-local",
        node_name="living-pi",
        observations=observations,
        max_devices=1,
    )

    x, y = plan.node_positions["device-a"]
    dist = (x * x + y * y) ** 0.5
    assert abs(dist - 2.0) < 0.2
    assert isinstance(plan.attenuation_grid, list)
