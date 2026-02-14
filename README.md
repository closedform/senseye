# senseye

Weekend prototype for fun: a state-of-the-art sensor-fusion demo for room mapping and motion tracking. Built with help from Claude, Codex, and Antigravity. Please do not embed this in my thermostat or smoke detector.

Distributed RF/acoustic sensing for indoor floorplan mapping and live motion inference.

Senseye builds a static floorplan from WiFi/BLE/acoustic structure and overlays dynamic motion/device estimates in real time.

Senseye uses:

- Adaptive **2-state Kalman filtering** per link (`rssi`, `rssi_rate`) with Joseph-form covariance update and innovation-based process-noise scaling
- Confidence-aware local inference (sample support + innovation penalty + acoustic SNR)
- **Inverse-variance consensus** fusion across links, devices, and zones with consistent `c/(1-c)` precision weighting
- Robust weighted trilateration with Tukey biweight outlier suppression and inlier refit
- Inverse-variance-weighted ridge tomography with adaptive regularization
- Gossip relay with `sequence_number` dedup and `hop_count` TTL

For full derivations and equations, see `DESIGN.md`.

## How it works

Each node runs: **scan** → **filter** → **infer** → **share** → **fuse** → **render**

### PHASE 1: passive RF sensing (automatic, continuous)

All nodes scan WiFi + BLE. Kalman-filtered RSSI gives distance estimates, attenuation reveals walls.

Result: coarse zone graph (numbered candidates)

```mermaid
flowchart LR
  Z1["Zone 1 (candidate)"] ---|"strong link"| Z2["Zone 2 (candidate)"]
  Z1 ---|"high attenuation"| Z3["Zone 3 (candidate)"]
  Z2 -.->|"weak evidence"| Z3
```

### PHASE 2: acoustic calibration (on demand)

User triggers: `senseye calibrate`
All fixed nodes chirp in sequence (~30 seconds)
→ precise distance matrix (cm-accurate)
→ MDS gives accurate layout
→ echo profiles give room dimensions
→ wall positions snap into focus

Result: structured floorplan

```mermaid
flowchart TB
  K["Kitchen (n1)"] --- H["Hallway"]
  B["Bedroom (n2)"] --- L["Living (n3)"]
  K --- B
  H ---|"door"| L
```

### PHASE 3: motion-refined (passive, over hours)

Zone transitions from device tracking refine room connectivity.
Doorways discovered from repeated cross-room movement patterns.

```mermaid
flowchart LR
  K["Kitchen (motion: low)"] --- H["Hallway"]
  H ---|"doorway learned from transitions"| L["Living (motion: high)"]
  B["Bedroom (motion: idle)"] --- L
  P["Phone"] --> K
  W["Watch"] --> L
```

`motion: living high, kitchen low • devices: 2 tracked • nodes: 3 online`

### Node pipeline

Each node runs the same code. Beliefs flow through the gossip mesh into shared fusion.

```mermaid
flowchart TD
  S["WiFi/BLE/Acoustic Signals"] --> SCAN["Scan"]
  SCAN --> KF["Adaptive Kalman"]
  KF --> INF["Local Inference"]
  INF --> BLF["Belief (links/devices/zones + confidence)"]
  BLF --> MESH["Gossip Mesh (mDNS + TCP, seq dedup, hop TTL)"]
  MESH --> CONS["Inverse-Variance Consensus"]
  CONS --> TRI["Robust Trilateration"]
  CONS --> TOMO["Weighted Ridge Tomography"]
  TRI --> WORLD["World State (static map + dynamic overlay)"]
  TOMO --> WORLD
  WORLD --> UI["Dashboard"]
```

## Adding nodes

Any device that can run Python is a node. More nodes = more signal paths = better resolution.

Approximate pairwise link growth (complete graph assumption): `links = N(N-1)/2`

```mermaid
flowchart TB
  subgraph G1["1 fixed node"]
    A1["n1"]
  end

  subgraph G3["2-3 fixed nodes (all-connected mesh)"]
    B1["n1"] --- B2["n2"]
    B1 --- B3["n3"]
    B2 --- B3
  end

  subgraph G4["4-5 fixed nodes (all-connected mesh example)"]
    C1["n1"] --- C2["n2"]
    C1 --- C3["n3"]
    C1 --- C4["n4"]
    C2 --- C3
    C2 --- C4
    C3 --- C4
  end

  subgraph G8["8+ fixed nodes (all-connected pattern, shown with 5)"]
    D1["n1"] --- D2["n2"]
    D1 --- D3["n3"]
    D1 --- D4["n4"]
    D1 --- D5["n5"]
    D2 --- D3
    D2 --- D4
    D2 --- D5
    D3 --- D4
    D3 --- D5
    D4 --- D5
  end
```

| Nodes | Signal paths | What you get |
|-------|-------------|-------------|
| 1 | sparse | Motion detection to router + BLE devices |
| 2-3 | basic mesh | Zone-level tracking, room connectivity |
| 4-5 | 6-10 links | Rough floor plan, directional motion |
| 8+ | 28+ links | Detailed room shapes, sub-room localization |

## Quick start

```bash
# UI + sensing
uv run senseye

# Headless sensor node
uv run senseye --headless --role fixed --name kitchen

# Calibrate the map (acoustic ping sequence)
uv run senseye calibrate

# Interval acoustic mode
uv run senseye --acoustic 10m
```

## Acoustic mode

Nodes with speakers and microphones can ping each other ultrasonically (18-22kHz, inaudible) for centimeter-accurate distance measurement.

```bash
uv run senseye --acoustic off         # no chirps (default)
uv run senseye --acoustic on-demand   # only during calibration
uv run senseye --acoustic 10m         # ping every 10 minutes
uv run senseye --acoustic 1h          # ping every hour
```

## Architecture

```
senseye/
    node/           # scan, filter, infer, peer, belief
    fusion/         # consensus, trilateration, tomography
    mapping/
        static/     # walls, rooms, topology (built once, refined rarely)
        dynamic/    # device positions, motion (updated every second)
    ui/             # terminal dashboard
    calibration.py  # active map calibration
    config.py       # runtime configuration
    protocol.py     # wire format (newline-delimited JSON over TCP)
    main.py         # entry point
```

## Requirements

- Python 3.11+
- macOS or Linux
- WiFi and/or BLE hardware
- Optional: speaker + mic for acoustic mode
