"""Main entry point: orchestrates scan → filter → infer → share → fuse → render."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

from senseye.config import (
    AcousticMode,
    NodeRole,
    SenseyeConfig,
    apply_overrides,
    load_config_file,
    parse_acoustic_interval,
    _parse_acoustic_mode,
)
from senseye.mapping.dynamic.state import WorldState, update_world
from senseye.mapping.static.floorplan import FloorPlan, load as load_floorplan
from senseye.node.belief import Belief
from senseye.node.filter import FilterBank
from senseye.node.inference import RssiHistory, infer
from senseye.node.peer import PeerMesh
from senseye.node.scanner import scan_all
from senseye.fusion.consensus import fuse_beliefs

log = logging.getLogger("senseye")


def build_config() -> SenseyeConfig:
    parser = argparse.ArgumentParser(prog="senseye", description="Distributed RF sensing")
    parser.add_argument("command", nargs="?", default=None, help="Subcommand: calibrate")
    parser.add_argument("--headless", action="store_true", help="No UI, sensor only")
    parser.add_argument("--ui-only", action="store_true", help="UI only, no scanning")
    parser.add_argument("--name", type=str, default="", help="Node name")
    parser.add_argument("--role", choices=["fixed", "mobile"], default="fixed")
    parser.add_argument("--acoustic", type=str, default="off", help="off|on-demand|10m|1h")
    parser.add_argument("--port", type=int, default=5483, help="Mesh TCP port")
    parser.add_argument("--no-wifi", action="store_true", help="Disable WiFi scanning")
    parser.add_argument("--no-ble", action="store_true", help="Disable BLE scanning")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = SenseyeConfig()

    # Load from config file
    config_path = config.data_dir / "config.toml"
    file_overrides = load_config_file(config_path)
    apply_overrides(config, file_overrides)

    # Apply CLI overrides
    if args.name:
        config.node_name = args.name
    config.node_role = NodeRole(args.role)
    config.mesh_port = args.port
    config.wifi_enabled = not args.no_wifi
    config.ble_enabled = not args.no_ble
    config.acoustic_mode = _parse_acoustic_mode(args.acoustic)
    if config.acoustic_mode == AcousticMode.INTERVAL:
        config.acoustic_interval = parse_acoustic_interval(args.acoustic)

    if args.headless:
        config.ui_enabled = False
    if args.ui_only:
        config.wifi_enabled = False
        config.ble_enabled = False

    config._command = args.command  # type: ignore[attr-defined]
    return config


async def run(config: SenseyeConfig) -> None:
    """Main async loop."""
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Load existing floorplan
    floorplan_path = config.data_dir / "floorplan.json"
    floorplan: FloorPlan | None = None
    if floorplan_path.exists():
        try:
            floorplan = load_floorplan(floorplan_path)
            log.info("loaded floorplan from %s", floorplan_path)
        except Exception:
            log.warning("failed to load floorplan, starting fresh")

    # State
    world = WorldState(floorplan=floorplan)
    filter_bank = FilterBank()
    rssi_history: RssiHistory = {}
    peer_beliefs: list[Belief] = []
    lock = asyncio.Lock()

    # Peer mesh
    mesh = PeerMesh(node_id=config.node_id, port=config.mesh_port)

    def on_peer_belief(belief: Belief) -> None:
        peer_beliefs.append(belief)
        # Keep bounded
        if len(peer_beliefs) > 100:
            peer_beliefs.pop(0)

    mesh.on_belief(on_peer_belief)

    shutdown = asyncio.Event()

    def handle_signal() -> None:
        log.info("shutting down...")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await mesh.start()
    log.info("node %s (%s) started, role=%s", config.node_name, config.node_id, config.node_role.value)

    # Scan → filter → infer → share → fuse loop
    async def sense_loop() -> None:
        nonlocal world
        last_time = time.time()

        while not shutdown.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now

            # Scan
            if config.wifi_enabled or config.ble_enabled:
                observations = await scan_all(
                    wifi=config.wifi_enabled,
                    ble=config.ble_enabled,
                    ble_duration=config.ble_duration,
                )
            else:
                observations = []

            # Filter (Kalman) each observation
            for obs in observations:
                filter_bank.update(config.node_id, obs.device_id, obs.rssi)

            # Infer locally
            node_positions = {}
            if floorplan is not None:
                node_positions = floorplan.node_positions
            local_belief = infer(
                observations=observations,
                rssi_history=rssi_history,
                node_positions=node_positions,
                my_node_id=config.node_id,
            )

            # Share with peers
            await mesh.broadcast_belief(local_belief)

            # Fuse with peer beliefs
            async with lock:
                recent_peers = list(peer_beliefs[-10:])  # last 10 peer beliefs
            fused = fuse_beliefs(local_belief, recent_peers)

            # Update world state
            world = update_world(world, fused, dt)

            # Wait for next cycle
            elapsed = time.time() - now
            sleep_time = max(0, config.scan_interval - elapsed)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=sleep_time)
                break  # shutdown was set
            except asyncio.TimeoutError:
                pass

    # Dashboard
    async def ui_loop() -> None:
        if not config.ui_enabled:
            await shutdown.wait()
            return

        from senseye.ui.dashboard import Dashboard

        dashboard = Dashboard(config)

        async def state_stream():
            while not shutdown.is_set():
                yield world
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=config.ui_refresh)
                    return
                except asyncio.TimeoutError:
                    pass

        await dashboard.run(state_stream())

    # Run loops
    tasks = [asyncio.create_task(sense_loop())]
    if config.ui_enabled:
        tasks.append(asyncio.create_task(ui_loop()))
    else:
        tasks.append(asyncio.create_task(shutdown.wait()))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await mesh.stop()
        log.info("senseye stopped")


def main() -> None:
    config = build_config()
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
