from __future__ import annotations

from senseye.node.filter import FilterBank, KalmanFilter1D


def test_filter_bank_tracks_rssi_rate() -> None:
    bank = FilterBank(process_noise=0.2, measurement_noise=1.0, dt=0.5)
    for value in (-70.0, -68.0, -66.0, -64.0, -62.0):
        bank.update("node-a", "device-1", value)

    state = bank.get_state("node-a", "device-1")
    assert state is not None
    filtered_rssi, rssi_rate = state
    assert filtered_rssi > -66.0
    assert rssi_rate > 0.0


def test_adaptive_kalman_reacts_to_jump() -> None:
    kf = KalmanFilter1D(
        process_noise=0.1,
        measurement_noise=1.0,
        adaptive_threshold=2.0,
        scaling_factor=200.0,
        dt=1.0,
    )
    for _ in range(5):
        kf.update(-70.0)

    filtered, innovation = kf.update(-50.0)
    assert filtered > -60.0
    assert innovation > 0.0
