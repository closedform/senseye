from __future__ import annotations

import sys

from senseye.config import AcousticMode
from senseye.main import build_config


def test_headless_does_not_force_disable_acoustic(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["senseye", "--headless", "--acoustic", "10m"])
    config = build_config()

    assert config.ui_enabled is False
    assert config.acoustic_mode is AcousticMode.INTERVAL
    assert config.acoustic_interval == 600.0
