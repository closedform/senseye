"""Kalman filter per signal path."""

from __future__ import annotations

import numpy as np


class KalmanFilter1D:
    """1D Kalman filter tracking [rssi, rssi_rate].

    State vector: x = [rssi, d(rssi)/dt]
    Measurement: z = rssi reading
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 4.0,
        dt: float = 1.0,
    ) -> None:
        # State: [rssi, rssi_rate]
        self.x = np.zeros(2)
        # Covariance
        self.P = np.eye(2) * 100.0  # high initial uncertainty
        # State transition: rssi(t+1) = rssi(t) + dt * rssi_rate(t)
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        # Measurement matrix: we observe rssi directly
        self.H = np.array([[1.0, 0.0]])
        # Process noise
        self.Q = np.array([
            [dt**2 / 4, dt / 2],
            [dt / 2,    1.0],
        ]) * process_noise
        # Measurement noise
        self.R = np.array([[measurement_noise]])
        self._initialized = False

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: float) -> tuple[float, float]:
        """Incorporate a new RSSI measurement.

        Returns (filtered_rssi, innovation).
        Innovation = measurement - predicted measurement. Large values indicate change.
        """
        if not self._initialized:
            self.x[0] = measurement
            self.x[1] = 0.0
            self._initialized = True
            return (measurement, 0.0)

        self.predict()

        # Innovation
        y = measurement - float(self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ np.array([[y]])).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P

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
