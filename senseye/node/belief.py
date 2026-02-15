from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class LinkState:
    attenuation: float
    motion: bool
    confidence: float


@dataclass
class DeviceState:
    rssi: float
    estimated_distance: float | None
    moving: bool


@dataclass
class ZoneBelief:
    occupied: float  # 0-1
    motion: float  # 0-1


@dataclass
class Belief:
    node_id: str
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    hop_count: int = 3
    links: dict[str, LinkState] = field(default_factory=dict)
    devices: dict[str, DeviceState] = field(default_factory=dict)
    zones: dict[str, ZoneBelief] = field(default_factory=dict)
    acoustic_ranges: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "hop_count": self.hop_count,
            "links": {
                k: {"attenuation": v.attenuation, "motion": v.motion, "confidence": v.confidence}
                for k, v in self.links.items()
            },
            "devices": {
                k: {
                    "rssi": v.rssi,
                    "estimated_distance": v.estimated_distance,
                    "moving": v.moving,
                }
                for k, v in self.devices.items()
            },
            "zones": {
                k: {"occupied": v.occupied, "motion": v.motion}
                for k, v in self.zones.items()
            },
            "acoustic_ranges": dict(self.acoustic_ranges),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Belief:
        return cls(
            node_id=d["node_id"],
            timestamp=d["timestamp"],
            sequence_number=d.get("sequence_number", 0),
            hop_count=d.get("hop_count", 3),
            links={
                k: LinkState(
                    attenuation=v["attenuation"],
                    motion=v["motion"],
                    confidence=v["confidence"],
                )
                for k, v in d.get("links", {}).items()
            },
            devices={
                k: DeviceState(
                    rssi=v["rssi"],
                    estimated_distance=v["estimated_distance"],
                    moving=v["moving"],
                )
                for k, v in d.get("devices", {}).items()
            },
            zones={
                k: ZoneBelief(occupied=v["occupied"], motion=v["motion"])
                for k, v in d.get("zones", {}).items()
            },
            acoustic_ranges={
                k: float(v) for k, v in d.get("acoustic_ranges", {}).items()
                if isinstance(v, int | float)
            },
        )
