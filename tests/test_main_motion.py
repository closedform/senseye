from __future__ import annotations

from senseye.main import _extract_motion_events
from senseye.mapping.dynamic.devices import TrackedDevice
from senseye.mapping.dynamic.state import WorldState


def test_extract_motion_events_tracks_zone_transitions() -> None:
    state = WorldState()
    state.devices["phone"] = TrackedDevice(
        device_id="phone",
        zone="kitchen",
        moving=True,
    )
    last_zone: dict[str, str] = {}

    first = _extract_motion_events(state, last_zone, timestamp=1.0)
    assert first == []

    state.devices["phone"].zone = "hallway"
    second = _extract_motion_events(state, last_zone, timestamp=2.0)
    assert second == [("kitchen", "hallway", 2.0)]


def test_extract_motion_events_requires_moving_device() -> None:
    state = WorldState()
    state.devices["phone"] = TrackedDevice(
        device_id="phone",
        zone="kitchen",
        moving=False,
    )
    last_zone = {"phone": "bedroom"}

    events = _extract_motion_events(state, last_zone, timestamp=3.0)
    assert events == []
