"""Runtime configuration for senseye."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NodeRole(Enum):
    FIXED = "fixed"
    MOBILE = "mobile"


class AcousticMode(Enum):
    OFF = "off"
    ON_DEMAND = "on-demand"
    INTERVAL = "interval"


@dataclass
class Position:
    x: float
    y: float


@dataclass
class SenseyeConfig:
    # Node identity
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_name: str = ""
    node_role: NodeRole = NodeRole.FIXED
    position: Position | None = None

    # Scanning
    wifi_enabled: bool = True
    ble_enabled: bool = True
    scan_interval: float = 1.0
    ble_duration: float = 2.0

    # Acoustic
    acoustic_mode: AcousticMode = AcousticMode.OFF
    acoustic_interval: float = 600.0  # seconds, for INTERVAL mode
    chirp_freq_start: int = 18000
    chirp_freq_end: int = 22000
    chirp_duration: float = 0.01

    # Networking
    mesh_port: int = 5483
    belief_rate: float = 1.0

    # UI
    ui_enabled: bool = True
    ui_refresh: float = 1.0

    # Paths
    data_dir: Path = field(default_factory=lambda: Path.home() / ".senseye")

    def __post_init__(self):
        if not self.node_name:
            self.node_name = self.node_id


def load_config_file(path: Path) -> dict:
    """Load config overrides from a TOML file. Returns empty dict if not found."""
    if not path.exists():
        return {}
    import tomllib
    return tomllib.loads(path.read_text())


def apply_overrides(config: SenseyeConfig, overrides: dict) -> SenseyeConfig:
    """Apply dict overrides (from TOML or CLI) onto a config."""
    for key, value in overrides.items():
        if key == "node_role" and isinstance(value, str):
            config.node_role = NodeRole(value)
        elif key == "acoustic_mode" and isinstance(value, str):
            config.acoustic_mode = _parse_acoustic_mode(value)
        elif key == "acoustic_interval":
            if isinstance(value, str):
                config.acoustic_interval = parse_acoustic_interval(value)
            elif isinstance(value, int | float):
                config.acoustic_interval = float(value)
        elif key == "position" and isinstance(value, dict):
            config.position = Position(x=value["x"], y=value["y"])
        elif key == "data_dir" and isinstance(value, str):
            config.data_dir = Path(value)
        elif hasattr(config, key):
            setattr(config, key, value)
    return config


def _parse_acoustic_mode(s: str) -> AcousticMode:
    """Parse acoustic mode string: 'off', 'on-demand', '10m', '1h', etc."""
    s = s.lower().strip()
    if s == "off":
        return AcousticMode.OFF
    if s == "on-demand":
        return AcousticMode.ON_DEMAND
    return AcousticMode.INTERVAL


def parse_acoustic_interval(s: str) -> float:
    """Parse interval string like '10m' or '1h' into seconds."""
    s = s.lower().strip()
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)
