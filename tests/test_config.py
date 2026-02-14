from __future__ import annotations

from senseye.config import (
    AcousticMode,
    NodeRole,
    SenseyeConfig,
    apply_overrides,
    parse_acoustic_interval,
)


def test_parse_acoustic_interval_units() -> None:
    assert parse_acoustic_interval("10m") == 600.0
    assert parse_acoustic_interval("1h") == 3600.0
    assert parse_acoustic_interval("30s") == 30.0
    assert parse_acoustic_interval("15") == 15.0


def test_apply_overrides_parses_acoustic_interval_string() -> None:
    config = SenseyeConfig()
    apply_overrides(config, {"acoustic_interval": "10m"})
    assert config.acoustic_interval == 600.0


def test_apply_overrides_accepts_numeric_interval() -> None:
    config = SenseyeConfig()
    apply_overrides(config, {"acoustic_interval": 90})
    assert config.acoustic_interval == 90.0


def test_apply_overrides_parses_enums() -> None:
    config = SenseyeConfig()
    apply_overrides(config, {"acoustic_mode": "on-demand", "node_role": "mobile"})
    assert config.acoustic_mode is AcousticMode.ON_DEMAND
    assert config.node_role is NodeRole.MOBILE
