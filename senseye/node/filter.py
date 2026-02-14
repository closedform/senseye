"""Kalman filter per signal path."""

from __future__ import annotations

import numpy as np


class KalmanFilter1D:
    """1D RSSI Kalman filter with constant-velocity state.

    State vector: x = [rssi, d(rssi)/dt]
    Measurement: z = rssi reading
    """

    def __init__(
        self,
        process_noise: float = 1.0,  # Q
        measurement_noise: float = 4.0,  # R
        adaptive_threshold: float = 3.0,  # Innovation Z-score threshold for jump detection
        scaling_factor: float = 100.0,  # Multiplier for Q when jump is detected
        dt: float = 1.0,
    ) -> None:
        self.dt = max(float(dt), 1e-3)
        # State: [rssi, rate]
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        # Covariance
        self.P = np.eye(2, dtype=np.float64) * 100.0
        # Constant-velocity transition.
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=np.float64)
        # Observe RSSI directly.
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.base_Q = self._build_process_covariance(process_noise)
        self.R = np.array([[measurement_noise]], dtype=np.float64)

        # Adaptive parameters
        self.adaptive_threshold = adaptive_threshold
        self.scaling_factor = scaling_factor

        self._initialized = False

    def _build_process_covariance(self, process_noise: float) -> np.ndarray:
        q = float(max(process_noise, 1e-6))
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        return q * np.array(
            [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]],
            dtype=np.float64,
        )

    def predict(self, Q_scale: float = 1.0) -> None:
        Q = self.base_Q * Q_scale
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, measurement: float) -> tuple[float, float]:
        """Incorporate a new RSSI measurement.

        Uses adaptive logic: check innovation. If the measurement is "impossible"
        given the current tracking state (high Z-score), assume a maneuver/jump occurred
        and massively boost process noise (Q) for this step to allow the filter to catch up.
        """
        if not self._initialized:
            self.x[0] = measurement
            self.x[1] = 0.0
            self._initialized = True
            return (measurement, 0.0)

        # 1. Calc innovation with standard prediction
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.base_Q

        y = measurement - float((self.H @ x_pred)[0])
        S = self.H @ P_pred @ self.H.T + self.R

        # 2. Check Z-score (std deviations from expectation)
        innovation_std = float(np.sqrt(max(S[0, 0], 1e-12)))
        z_score = abs(y) / innovation_std

        # 3. If jump detected, scale Q to allow rapid transition
        q_scale = 1.0
        if z_score > self.adaptive_threshold:
            # Boost uncertainty to believe measurement more than history.
            q_scale = self.scaling_factor

        # 4. Real update
        self.predict(Q_scale=q_scale)

        # Re-calc innovation with potentially new P
        y = measurement - float((self.H @ self.x)[0])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ np.array([y], dtype=np.float64)).ravel()
        # Joseph form: guarantees P stays symmetric positive-definite
        IKH = np.eye(2, dtype=np.float64) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        return (float(self.x[0]), y)


class FilterBank:
    """Collection of Kalman filters keyed by (source_id, target_id)."""

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 4.0,
        dt: float = 1.0,
    ) -> None:
        self._filters: dict[tuple[str, str], KalmanFilter1D] = {}
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._dt = dt

    def update(self, source_id: str, target_id: str, rssi: float) -> tuple[float, float]:
        key = (source_id, target_id)
        if key not in self._filters:
            self._filters[key] = KalmanFilter1D(
                process_noise=self._process_noise,
                measurement_noise=self._measurement_noise,
                dt=self._dt,
            )
        return self._filters[key].update(rssi)

    def get_state(self, source_id: str, target_id: str) -> tuple[float, float] | None:
        """Return (filtered_rssi, rssi_rate) for a signal path, or None if unseen."""
        key = (source_id, target_id)
        kf = self._filters.get(key)
        if kf is None or not kf._initialized:
            return None
        return (float(kf.x[0]), float(kf.x[1]))
