# Senseye Design Document

## Overview

Senseye is a distributed RF sensing system that maps an apartment and detects motion using commodity hardware. Every WiFi, BLE, and acoustic-capable device in the space acts as a sensor node. The system builds a static map of the apartment from signal attenuation patterns and overlays live motion detection on top.

## Core Principles

1. **Every node is equal** — same codebase runs on Mac, Pi, or any Python-capable device
2. **Local inference first** — each node processes its own data and shares beliefs, not raw readings
3. **Graph-based fusion** — the signal topology IS the data structure. More nodes = more edges = better resolution
4. **Static map, dynamic overlay** — the apartment layout is built once and refined rarely. Motion is ephemeral
5. **Zero config** — nodes discover each other via mDNS. No IP addresses, no config files
6. **Minimal dependencies** — `bleak` (BLE), `zeroconf` (mDNS), `numpy` (DSP/math), `rich` (terminal UI). That's it.

## Architecture

```
senseye/
    node/                    # The autonomous sensor agent
        scanner.py           # WiFi + BLE RSSI collection
        acoustic.py          # Ultrasonic chirp TX/RX + echo profiling
        filter.py            # Kalman filter per signal path
        inference.py         # Local motion/zone/attenuation inference
        peer.py              # mDNS discovery + TCP peer mesh
        belief.py            # Belief state dataclass + serialization
        __main__.py          # Entry point for headless nodes
    fusion/                  # Distributed inference
        consensus.py         # Weighted belief averaging across peers
        tomography.py        # Radio Tomographic Imaging (attenuation field)
        trilateration.py     # Device positioning when 3+ nodes observe it
        acoustic_range.py    # Precise distances from ultrasonic chirp ToF
        graph.py             # Signal graph: vertices (nodes/devices), edges (observations)
    mapping/
        static/              # Built once, refined rarely
            layout.py        # MDS self-localization of fixed nodes
            walls.py         # Wall inference from excess RF attenuation + echo
            topology.py      # Room connectivity from motion path traces
            floorplan.py     # Combined static map, serializable to disk
        dynamic/             # Updated every cycle
            motion.py        # Per-edge motion detection via RSSI variance
            devices.py       # Mobile device position tracking
            state.py         # Fused world state combining static + dynamic
    ui/
        renderer.py          # Static map -> character grid
        overlay.py           # Motion + device overlay on static background
        dashboard.py         # Live terminal dashboard (rich)
    config.py                # Runtime configuration
    protocol.py              # Wire format for peer communication
    main.py                  # Entry point: python -m senseye
```

## Node Architecture

Every node runs an identical pipeline:

```
SCAN → FILTER → INFER → SHARE → FUSE → (optional: RENDER)
```

### Scan (scanner.py, acoustic.py)

Collects raw signal observations from all available sensors:

- **WiFi RSSI**: Signal strength to all visible access points. On macOS, uses CoreWLAN via the `airport` utility. On Linux, uses `iwlist` or `/proc/net/wireless`.
- **BLE RSSI**: Signal strength to all BLE-advertising devices. Uses `bleak` (cross-platform). Captures manufacturer data and service UUIDs for device identification.
- **Acoustic**: Ultrasonic chirp time-of-flight for precise ranging. Uses `numpy` for chirp generation (18-22kHz FMCW sweep) and matched filtering. Optional echo profiling for room geometry.

All observations are timestamped and tagged with source type.

### Filter (filter.py)

Each signal path (node→device or node→node) gets an independent Kalman filter:

- State: estimated RSSI + rate of change
- Measurement: raw RSSI reading
- Output: smoothed RSSI, innovation (deviation from prediction)
- Innovation spikes indicate environmental changes (motion, door opening, etc.)

The Kalman filter is a simple 1D filter per path. No complex state estimation.

### Infer (inference.py)

Local inference produces beliefs about the environment:

- **Per-link motion**: RSSI variance in a sliding window exceeds threshold → motion on that path
- **Per-link attenuation**: Stable RSSI below free-space prediction → obstruction (wall) on that path
- **Zone estimation**: Bayesian update over room probabilities given all link observations
- **Device tracking**: RSSI-based distance estimate for each observed device

Output is a `Belief` dataclass (belief.py) containing:
- Link states: {peer_id → (attenuation, motion, confidence)}
- Device states: {device_id → (rssi, estimated_distance, moving)}
- Zone beliefs: {zone_name → (occupied_probability, motion_probability)}

### Share (peer.py)

Nodes communicate over a TCP mesh:

- **Discovery**: Each node registers a `_senseye._tcp.local.` mDNS service on startup
- **Connection**: Nodes connect to all discovered peers via TCP
- **Protocol**: Newline-delimited JSON (protocol.py). Each message is a serialized Belief
- **Heartbeat**: Nodes send beliefs at a configurable rate (default: 1/sec)
- **Reconnection**: Automatic reconnect on disconnect with exponential backoff

### Fuse (fusion/)

Each node independently fuses its own beliefs with received peer beliefs:

- **Consensus** (consensus.py): Weighted average of beliefs, weighted by confidence. Agreement between nodes amplifies confidence; disagreement flags uncertainty.
- **Tomography** (tomography.py): Builds a spatial attenuation field from all node-to-node link attenuations. Cells where many high-attenuation links intersect → walls.
- **Trilateration** (trilateration.py): When 3+ nodes observe the same device, estimate its (x,y) position via weighted least-squares.
- **Acoustic ranging** (acoustic_range.py): Precise inter-node distances from ultrasonic chirp time-of-flight. Feeds into MDS layout.

## Static Map Generation

The static map represents the physical apartment layout. It updates only on explicit triggers.

### Building the Map

```
1. DISTANCE MATRIX
   - RF: pairwise RSSI between all fixed nodes → rough distances (±2m)
   - Acoustic: pairwise chirp ToF → precise distances (±1cm)
   - Combined: acoustic when available, RF as fallback

2. NODE POSITIONS (layout.py)
   - Classical MDS on distance matrix → relative (x,y) for each fixed node
   - User anchors 1-2 nodes to fix rotation/reflection
   - Refines as more nodes join

3. WALL DETECTION (walls.py)
   - For each node pair: compare measured RSSI to free-space prediction at known distance
   - Excess attenuation = obstruction on that path
   - Tomographic reconstruction: project excess attenuation onto spatial grid
   - Cells with high cumulative attenuation → walls

4. ROOM CONNECTIVITY (topology.py)
   - Observe motion traces over time
   - Sequential link attenuations reveal traversal paths
   - Paths go through doorways, not walls
   - Builds graph: rooms as vertices, doorways as edges

5. FLOOR PLAN (floorplan.py)
   - Combines: node positions + wall grid + room connectivity + room labels
   - Serializes to JSON on disk (~/.senseye/floorplan.json)
   - Loaded on startup, recalculated only on trigger
```

### Map Update Triggers

- `senseye calibrate` — user-initiated full recalibration with acoustic pings
- New fixed node joins the mesh
- Fixed node RSSI to peers shifts significantly (node was moved)
- Scheduled acoustic ping (configurable: off / on-demand / 10m / 1h)

### Node Roles

- **Fixed**: Pi, router, Hue bulb, Apple TV — known positions, contributes to static map
- **Mobile**: laptop, phone, watch — tracked ON the map, never part of it

## Dynamic Overlay

Updated every scan cycle (~1 second):

- **Motion shading**: rooms/zones colored by motion intensity
- **Device markers**: mobile devices plotted at estimated (x,y) or zone
- **Trails**: optional motion trails showing recent device paths
- **Status**: node health, map age, signal quality

## Configuration (config.py)

```python
@dataclass
class SenseyeConfig:
    # Node identity
    node_id: str            # auto-generated or user-set
    node_role: NodeRole     # FIXED or MOBILE
    position: Position | None  # (x, y) if known, None for auto

    # Scanning
    wifi_enabled: bool = True
    ble_enabled: bool = True
    scan_interval: float = 1.0  # seconds

    # Acoustic
    acoustic_mode: str = "on-demand"  # off | on-demand | 10m | 1h
    chirp_freq_start: int = 18000     # Hz
    chirp_freq_end: int = 22000       # Hz
    chirp_duration: float = 0.01      # seconds

    # Networking
    mesh_port: int = 5483
    belief_rate: float = 1.0  # beliefs per second

    # UI
    ui_enabled: bool = True
    ui_refresh: float = 1.0  # seconds
```

## Wire Protocol (protocol.py)

Newline-delimited JSON over TCP. All messages have a type field:

```json
{"type": "belief", "node_id": "...", "timestamp": ..., "links": {...}, "devices": {...}, "zones": {...}}
{"type": "ping", "node_id": "...", "timestamp": ...}
{"type": "calibrate", "node_id": "...", "sequence": [...]}
{"type": "announce", "node_id": "...", "role": "fixed", "position": [x, y]}
```

No framing beyond newline. No versioning for now. Keep it simple.

## Dependencies

```
bleak       # BLE scanning (cross-platform)
zeroconf    # mDNS discovery (cross-platform)
numpy       # DSP (chirp generation, matched filter, MDS, Kalman)
rich        # Terminal UI (live dashboard)
```

Four packages. No frameworks, no message brokers, no databases.

## Platform Support

- **macOS**: WiFi via `airport` utility, BLE via `bleak` (CoreBluetooth), acoustic via `sounddevice` or `pyaudio`
- **Linux (Pi)**: WiFi via `iwlist`/`iw`, BLE via `bleak` (BlueZ), acoustic via ALSA
- **Headless**: `python -m senseye --headless` — no UI, no acoustic, just WiFi+BLE scanning and peer mesh

## Design Constraints

- No overengineering: if a simple approach works, use it. Three lines of repetition > premature abstraction.
- No unnecessary error handling: trust internal code. Validate at boundaries (network input, user config).
- No classes where a function suffices. No inheritance hierarchies. Dataclasses for data, functions for behavior.
- Type hints on public interfaces. No docstrings on obvious code.
- async throughout the node pipeline (scanning, networking, UI are all I/O-bound).
