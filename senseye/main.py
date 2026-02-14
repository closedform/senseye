"""Main entry point: orchestrates scan → filter → infer → share → fuse → render."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from collections.abc import Mapping
from pathlib import Path

from senseye.calibration import calibrate_floorplan
from senseye.config import (
    AcousticMode,
    NodeRole,
    SenseyeConfig,
    _parse_acoustic_mode,
    apply_overrides,
    load_config_file,
    parse_acoustic_interval,
)
from senseye.fusion.consensus import fuse_beliefs
from senseye.fusion.runtime import estimate_device_positions, reconstruct_attenuation_grid
from senseye.mapping.dynamic.state import WorldState, update_world
from senseye.mapping.static.floorplan import FloorPlan
from senseye.mapping.static.floorplan import load as load_floorplan
from senseye.mapping.static.floorplan import save as save_floorplan
from senseye.mapping.static.topology import update_topology
from senseye.node.acoustic import (
    DEFAULT_SAMPLE_RATE,
    SPEED_OF_SOUND,
    echo_profile,
    generate_chirp,
    get_chirp_params,
    listen_for_chirp,
    play_chirp,
)
from senseye.node.belief import Belief
from senseye.node.filter import FilterBank
from senseye.node.inference import RssiHistory, infer
from senseye.node.peer import PeerMesh
from senseye.node.scanner import Observation, SignalType, scan_all

log = logging.getLogger("senseye")

_CALIBRATION_SCANS = 3
_MIN_CALIBRATION_GAP = 30.0
_RSSI_DRIFT_THRESHOLD = 8.0
_MIN_BELIEF_RATE = 0.1
_ACOUSTIC_OBSERVATION_ID = "acoustic:echo"
_TOMOGRAPHY_RESOLUTION = 0.5
_ACOUSTIC_PING_DELAY_S = 0.2
_ACOUSTIC_MIN_SNR = 3.0
_MAX_ACOUSTIC_TOF_S = 0.2
_MAX_MOTION_EVENTS = 500


def _command(config: SenseyeConfig) -> str | None:
    return getattr(config, "_command", None)


def _apply_kalman(
    observations: list[Observation],
    filter_bank: FilterBank,
    node_id: str,
) -> list[Observation]:
    filtered: list[Observation] = []
    for obs in observations:
        filtered_rssi, innovation = filter_bank.update(node_id, obs.device_id, obs.rssi)
        metadata = dict(obs.metadata)
        metadata["raw_rssi"] = obs.rssi
        metadata["innovation"] = innovation
        filtered.append(Observation(
            device_id=obs.device_id,
            rssi=filtered_rssi,
            timestamp=obs.timestamp,
            signal_type=obs.signal_type,
            metadata=metadata,
        ))
    return filtered


async def _run_calibrate_command(config: SenseyeConfig, floorplan_path: Path) -> None:
    log.info("starting calibration")
    floorplan, _ = await calibrate_floorplan(
        config,
        scans=_CALIBRATION_SCANS,
        force_acoustic=config.acoustic_mode != AcousticMode.OFF,
    )
    save_floorplan(floorplan, floorplan_path)
    log.info(
        "saved floorplan to %s (%d nodes, %d walls, %d rooms)",
        floorplan_path,
        len(floorplan.node_positions),
        len(floorplan.wall_segments),
        len(floorplan.rooms.rooms),
    )


async def _sample_acoustic_observation(config: SenseyeConfig) -> Observation | None:
    try:
        chirp = generate_chirp(
            freq_start=config.chirp_freq_start,
            freq_end=config.chirp_freq_end,
            duration=config.chirp_duration,
        )
        profile = await echo_profile(chirp=chirp)
    except Exception:
        log.debug("acoustic sample failed", exc_info=True)
        return None

    if profile is None:
        return None

    distance = float(profile.distance) if profile.distance is not None else 0.0
    return Observation(
        device_id=f"{_ACOUSTIC_OBSERVATION_ID}:{config.node_id}",
        rssi=distance,
        timestamp=profile.timestamp,
        signal_type=SignalType.ACOUSTIC,
        metadata={
            "distance_m": profile.distance,
            "peak_snr": profile.peak_snr,
            "raw_rssi": distance,
        },
    )


async def _handle_acoustic_ping_request(
    peer_id: str, msg: Mapping[str, object], config: SenseyeConfig
) -> bool:
    """Handle a request from a peer to play a chirp."""
    del peer_id
    if config.acoustic_mode == AcousticMode.OFF:
        return False

    # Always emit on this node's deterministic channel for passive ID.
    f_start, f_end = get_chirp_params(config.node_id)

    requested_duration = msg.get("chirp_duration", config.chirp_duration)
    requested_sample_rate = msg.get("sample_rate", DEFAULT_SAMPLE_RATE)
    requested_delay = msg.get("delay_s", _ACOUSTIC_PING_DELAY_S)

    duration = (
        float(requested_duration)
        if isinstance(requested_duration, int | float) and requested_duration > 0
        else config.chirp_duration
    )
    sample_rate = (
        int(requested_sample_rate)
        if isinstance(requested_sample_rate, int | float) and requested_sample_rate > 0
        else DEFAULT_SAMPLE_RATE
    )
    delay = (
        float(requested_delay)
        if isinstance(requested_delay, int | float) and requested_delay >= 0
        else _ACOUSTIC_PING_DELAY_S
    )

    chirp = generate_chirp(
        freq_start=f_start,
        freq_end=f_end,
        duration=duration,
        sample_rate=sample_rate,
    )
    return await play_chirp(chirp=chirp, sample_rate=sample_rate, delay=delay)


async def _measure_peer_acoustic_tof(
    mesh: PeerMesh,
    config: SenseyeConfig,
    peer_id: str,
) -> float | None:
    if peer_id == config.node_id:
        return None

    # We expect the peer to chirp on THEIR channel
    f_start, f_end = get_chirp_params(peer_id)
    expected_chirp = generate_chirp(
        freq_start=f_start,
        freq_end=f_end,
        duration=config.chirp_duration,
        sample_rate=DEFAULT_SAMPLE_RATE,
    )

    record_duration = max(config.chirp_duration + _ACOUSTIC_PING_DELAY_S + 0.5, 0.8)
    listen_task = asyncio.create_task(
        listen_for_chirp(
            chirp=expected_chirp,
            sample_rate=DEFAULT_SAMPLE_RATE,
            record_duration=record_duration,
            template_length=0,
        )
    )
    await asyncio.sleep(0.02)
    loop = asyncio.get_running_loop()
    request_sent_at = loop.time()
    try:
        response = await mesh.request_acoustic_ping(
            peer_id,
            delay_s=_ACOUSTIC_PING_DELAY_S,
            sample_rate=DEFAULT_SAMPLE_RATE,
            freq_start=f_start,
            freq_end=f_end,
            chirp_duration=config.chirp_duration,
            timeout=record_duration + 1.0,
        )
    except ConnectionError:
        response = None
    response_received_at = loop.time()
    listen_result = await listen_task

    if response is None or not bool(response.get("ok")):
        return None
    if listen_result is None or listen_result.tof is None:
        return None
    if listen_result.peak_snr < _ACOUSTIC_MIN_SNR:
        return None

    one_way_network = max((response_received_at - request_sent_at) / 2.0, 0.0)
    expected_emit_at = request_sent_at + one_way_network + _ACOUSTIC_PING_DELAY_S
    arrival_at = listen_result.record_started_at + listen_result.tof
    estimated_tof = arrival_at - expected_emit_at
    if estimated_tof <= 0 or estimated_tof > _MAX_ACOUSTIC_TOF_S:
        return None
    return estimated_tof


async def _collect_peer_acoustic_observations(
    mesh: PeerMesh,
    config: SenseyeConfig,
    peer_ids: set[str],
) -> list[Observation]:
    observations: list[Observation] = []
    for peer_id in sorted(peer_ids):
        tof = await _measure_peer_acoustic_tof(mesh=mesh, config=config, peer_id=peer_id)
        if tof is None:
            continue
        distance = tof * SPEED_OF_SOUND
        observations.append(Observation(
            device_id=peer_id,
            rssi=distance,
            timestamp=time.time(),
            signal_type=SignalType.ACOUSTIC,
            metadata={
                "distance_m": distance,
                "tof_s": tof,
                "raw_rssi": distance,
            },
        ))
    return observations


def _extract_motion_events(
    world: WorldState,
    last_zone_by_device: dict[str, str],
    timestamp: float,
) -> list[tuple[str, str, float]]:
    events: list[tuple[str, str, float]] = []
    for device_id, device in world.devices.items():
        current_zone = device.zone
        if current_zone is None:
            continue
        previous_zone = last_zone_by_device.get(device_id)
        if (
            previous_zone is not None
            and previous_zone != current_zone
            and device.moving
        ):
            events.append((previous_zone, current_zone, timestamp))
        last_zone_by_device[device_id] = current_zone
    return events


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

    if args.command not in (None, "calibrate"):
        parser.error(f"unknown command: {args.command}")

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
    floorplan_path = config.data_dir / "floorplan.json"

    if _command(config) == "calibrate":
        await _run_calibrate_command(config, floorplan_path)
        return

    # Load existing floorplan
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
    # peer_id -> latest belief (snapshot)
    peer_beliefs: dict[str, Belief] = {}
    # Track latest sequence number seen for each peer to prevent loops/re-processing
    peer_seq_numbers: dict[str, int] = {}

    lock = asyncio.Lock()
    baseline_rssi: dict[str, float] = {}
    known_peers: set[str] = set()

    local_sequence_number = 0

    # Peer mesh
    mesh = PeerMesh(node_id=config.node_id, port=config.mesh_port)

    def on_peer_belief(belief: Belief) -> None:
        asyncio.create_task(process_peer_belief(belief))

    async def process_peer_belief(belief: Belief):
        # 1. Dedup: Check if we've already seen this sequence number from this node
        last_seq = peer_seq_numbers.get(belief.node_id, -1)
        if belief.sequence_number <= last_seq:
            return

        # Update tracker
        peer_seq_numbers[belief.node_id] = belief.sequence_number

        # 2. Store: Update our knowledge of this peer
        async with lock:
            peer_beliefs[belief.node_id] = belief

        # 3. Relay (Gossip): If TTL allows, re-broadcast to neighbors
        # We only relay if hop_count > 0. We decrement before sending.
        # Note: We do NOT modify the original belief object in storage,
        # but we create a copy or modify on the fly for sending.
        # Actually, since Belief is a dataclass, it's mutable. Let's be careful.

        if belief.hop_count > 0:
            # Create a relay copy with decremented TTL
            # We don't change the node_id (it's still *their* belief)
            # We rely on the mesh to not send it back to the sender if possible,
            # but the sequence number check protects us anyway.
            relay_belief = Belief(
                node_id=belief.node_id,
                timestamp=belief.timestamp,
                sequence_number=belief.sequence_number,
                hop_count=belief.hop_count - 1,
                links=belief.links,
                devices=belief.devices,
                zones=belief.zones,
            )
            # Fire and forget relay
            asyncio.create_task(mesh.broadcast_belief(relay_belief))

    async def on_acoustic_ping(peer_id: str, msg: Mapping[str, object]) -> bool:
        return await _handle_acoustic_ping_request(peer_id, msg, config)

    mesh.on_belief(on_peer_belief)
    mesh.on_acoustic_ping(on_acoustic_ping)

    shutdown = asyncio.Event()

    def handle_signal() -> None:
        log.info("shutting down...")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await mesh.start()
    log.info(
        "node %s (%s) started, role=%s",
        config.node_name,
        config.node_id,
        config.node_role.value,
    )
    known_peers = set(mesh.get_peers())

    # Scan → filter → infer → share → fuse loop
    async def sense_loop() -> None:
        nonlocal baseline_rssi
        nonlocal floorplan
        nonlocal known_peers
        nonlocal world
        nonlocal local_sequence_number

        last_time = time.time()
        last_belief_sent = 0.0
        last_calibration_at = 0.0
        motion_events: list[tuple[str, str, float]] = []
        last_zone_by_device: dict[str, str] = {}
        belief_period = 1.0 / max(config.belief_rate, _MIN_BELIEF_RATE)
        next_interval_calibration: float | None = None
        if config.acoustic_mode == AcousticMode.INTERVAL:
            next_interval_calibration = time.time() + config.acoustic_interval
        next_acoustic_sample: float | None = None
        if config.acoustic_mode == AcousticMode.INTERVAL:
            next_acoustic_sample = time.time() + config.acoustic_interval

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

            acoustic_due = (
                next_acoustic_sample is not None
                and now >= next_acoustic_sample
            )
            if acoustic_due:
                acoustic_observation = await _sample_acoustic_observation(config)
                if acoustic_observation is not None:
                    observations.append(acoustic_observation)
                peer_observations = await _collect_peer_acoustic_observations(
                    mesh=mesh,
                    config=config,
                    peer_ids=set(mesh.get_peers()),
                )
                observations.extend(peer_observations)
                next_acoustic_sample = now + config.acoustic_interval

            # Filter (Kalman) each observation
            filtered_observations = _apply_kalman(observations, filter_bank, config.node_id)

            if observations:
                log.debug("scanned %d signals", len(observations))

            # Infer locally
            node_positions = {}
            if floorplan is not None:
                node_positions = floorplan.node_positions
            local_belief = infer(
                observations=filtered_observations,
                rssi_history=rssi_history,
                node_positions=node_positions,
                my_node_id=config.node_id,
            )

            # Attach sequence number and reset hop count for OUR belief
            local_belief.sequence_number = local_sequence_number
            local_belief.hop_count = 3  # Default TTL
            local_sequence_number += 1

            # Share with peers
            if (now - last_belief_sent) >= belief_period:
                await mesh.broadcast_belief(local_belief)
                last_belief_sent = now

            # Fuse with peer beliefs
            async with lock:
                # Copy the dict to avoid mutation during fusion
                snapshot = dict(peer_beliefs)

            fused = fuse_beliefs(local_belief, snapshot)
            # Use current snapshot for downstream logic
            # (Note: using .values() here for list compatibility in visualization/logging if needed,
            # but fusion logic now takes the dict)
            fusion_window = [local_belief] + list(snapshot.values())

            # Update world state
            device_positions: dict[str, tuple[float, float]] = {}
            if floorplan is not None:
                estimates = estimate_device_positions(
                    beliefs=fusion_window,
                    node_positions=floorplan.node_positions,
                )
                device_positions = {
                    device_id: position
                    for device_id, (position, _uncertainty) in estimates.items()
                }
                attenuation_grid = reconstruct_attenuation_grid(
                    beliefs=fusion_window,
                    node_positions=floorplan.node_positions,
                    bounds=floorplan.bounds,
                    resolution=_TOMOGRAPHY_RESOLUTION,
                )
                if attenuation_grid.size > 0:
                    floorplan.attenuation_grid = attenuation_grid.tolist()
                    floorplan.attenuation_resolution = _TOMOGRAPHY_RESOLUTION

            signal_types = {
                obs.device_id: obs.signal_type.value
                for obs in filtered_observations
            }
            online_nodes = {belief.node_id for belief in fusion_window}
            world = update_world(
                world,
                fused,
                dt,
                device_positions=device_positions,
                device_signal_types=signal_types,
                online_nodes=online_nodes,
            )
            if floorplan is not None:
                new_events = _extract_motion_events(world, last_zone_by_device, now)
                if new_events:
                    motion_events.extend(new_events)
                    if len(motion_events) > _MAX_MOTION_EVENTS:
                        del motion_events[:-_MAX_MOTION_EVENTS]
                    updated_graph = update_topology(floorplan.rooms, motion_events)
                    if updated_graph != floorplan.rooms:
                        floorplan.rooms = updated_graph
                        world.floorplan = floorplan
                        save_floorplan(floorplan, floorplan_path)
                        log.info(
                            "updated topology from motion traces (%d connections)",
                            len(updated_graph.connections),
                        )

            # Rebuild static map when topology or acoustic schedule changes.
            peer_ids = set(mesh.get_peers())
            latest_rssi = {obs.device_id: obs.rssi for obs in filtered_observations}
            common = [device_id for device_id in baseline_rssi if device_id in latest_rssi]
            avg_drift = 0.0
            if common:
                avg_drift = sum(
                    abs(latest_rssi[device_id] - baseline_rssi[device_id])
                    for device_id in common
                ) / len(common)
            drift_due = len(common) >= 3 and avg_drift >= _RSSI_DRIFT_THRESHOLD

            interval_due = (
                next_interval_calibration is not None
                and now >= next_interval_calibration
            )

            reasons: list[str] = []
            if config.node_role == NodeRole.FIXED and (config.wifi_enabled or config.ble_enabled):
                if floorplan is None:
                    reasons.append("no-floorplan")
                elif peer_ids != known_peers:
                    reasons.append("peer-topology-change")
                if drift_due:
                    reasons.append(f"rssi-drift-{avg_drift:.1f}dB")
                if interval_due:
                    reasons.append("acoustic-interval")

            if reasons and (now - last_calibration_at) >= _MIN_CALIBRATION_GAP:
                log.info("recalibrating map (%s)", ", ".join(reasons))
                try:
                    calibrated, baseline = await calibrate_floorplan(
                        config,
                        peer_ids=peer_ids,
                        scans=_CALIBRATION_SCANS,
                        force_acoustic=interval_due,
                    )
                    save_floorplan(calibrated, floorplan_path)
                    floorplan = calibrated
                    world.floorplan = calibrated
                    baseline_rssi = baseline
                    known_peers = peer_ids
                    motion_events.clear()
                    last_zone_by_device.clear()
                    log.info(
                        "map updated (%d nodes, %d walls, %d rooms)",
                        len(calibrated.node_positions),
                        len(calibrated.wall_segments),
                        len(calibrated.rooms.rooms),
                    )
                except Exception:
                    log.exception("map recalibration failed")
                last_calibration_at = time.time()
                if next_interval_calibration is not None:
                    next_interval_calibration = last_calibration_at + config.acoustic_interval
            elif interval_due and next_interval_calibration is not None:
                next_interval_calibration = now + config.acoustic_interval
            elif peer_ids != known_peers:
                known_peers = peer_ids

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
