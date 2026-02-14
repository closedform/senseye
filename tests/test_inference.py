from __future__ import annotations

from senseye.node.inference import infer
from senseye.node.scanner import Observation, SignalType


def _observation(rssi: float, raw_rssi: float) -> Observation:
    return Observation(
        device_id="device-1",
        rssi=rssi,
        timestamp=0.0,
        signal_type=SignalType.WIFI,
        metadata={"raw_rssi": raw_rssi},
    )


def test_infer_uses_raw_history_and_filtered_current_rssi() -> None:
    history: dict[str, list[float]] = {}
    belief = infer(
        observations=[_observation(rssi=-60.0, raw_rssi=-50.0)],
        rssi_history=history,
        my_node_id="node-a",
    )

    assert history["device-1"][-1] == -50.0
    assert belief.devices["device-1"].rssi == -60.0


def test_infer_caps_rssi_history() -> None:
    history: dict[str, list[float]] = {}
    for idx in range(20):
        infer(
            observations=[_observation(rssi=float(idx), raw_rssi=float(idx))],
            rssi_history=history,
            my_node_id="node-a",
            history_limit=5,
        )

    assert history["device-1"] == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_infer_uses_acoustic_distance_metadata() -> None:
    history: dict[str, list[float]] = {}
    belief = infer(
        observations=[
            Observation(
                device_id="acoustic:echo:node-a",
                rssi=2.0,
                timestamp=0.0,
                signal_type=SignalType.ACOUSTIC,
                metadata={"distance_m": 2.75, "raw_rssi": 2.0},
            ),
        ],
        rssi_history=history,
        my_node_id="node-a",
    )

    assert belief.devices["acoustic:echo:node-a"].estimated_distance == 2.75
    assert belief.links["acoustic:echo:node-a"].attenuation == 0.0
