"""Mobile device position tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TrackedDevice:
    device_id: str
    name: str | None = None
    position: tuple[float, float] | None = None
    zone: str | None = None
    moving: bool = False
    last_seen: float = 0.0
    signal_type: str = "ble"


class DeviceTracker:
    def __init__(self) -> None:
        self._devices: dict[str, TrackedDevice] = {}

    def update(
        self,
        device_id: str,
        rssi: float | None = None,
        position: tuple[float, float] | None = None,
        zone: str | None = None,
        moving: bool = False,
        signal_type: str = "ble",
        name: str | None = None,
    ) -> TrackedDevice:
        """Upsert a device with latest observation."""
        now = time.time()
        existing = self._devices.get(device_id)
        if existing is None:
            device = TrackedDevice(
                device_id=device_id,
                name=name,
                position=position,
                zone=zone,
                moving=moving,
                last_seen=now,
                signal_type=signal_type,
            )
            self._devices[device_id] = device
            return device

        # Update fields, keeping previous values where new data is None
        if name is not None:
            existing.name = name
        if position is not None:
            existing.position = position
        if zone is not None:
            existing.zone = zone
        existing.moving = moving
        existing.last_seen = now
        existing.signal_type = signal_type
        return existing

    def get_active(self, max_age: float = 60.0) -> dict[str, TrackedDevice]:
        """Return devices seen within max_age seconds."""
        cutoff = time.time() - max_age
        return {
            did: dev for did, dev in self._devices.items()
            if dev.last_seen >= cutoff
        }

    def cleanup(self, max_age: float = 60.0) -> None:
        """Remove devices not seen within max_age seconds."""
        cutoff = time.time() - max_age
        stale = [did for did, dev in self._devices.items() if dev.last_seen < cutoff]
        for did in stale:
            del self._devices[did]
