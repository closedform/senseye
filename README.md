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

## Quick math summary

1. RF model (`n=2.5, A=45` indoor; `n=2.0` free-space baseline for calibration)

$$
\text{RSSI}_{\text{expected}}(d) = -(10n\log_{10}(d)+A),
\qquad
d = 10^{\frac{-\text{RSSI}-A}{10n}}
$$

2. Kalman model (per path)

$$
\mathbf{x}_k=[\text{rssi}_k,\dot{\text{rssi}}_k]^T,
\quad
\mathbf{x}_{k|k-1}=\mathbf{F}\mathbf{x}_{k-1|k-1},
\quad
\mathbf{K}_k=\mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T+R)^{-1}
$$

3. Consensus weighting

$$
\sigma^2(c)=\frac{1-c}{c}+\epsilon,
\quad
w=\sigma^{-2},
\quad
\hat{x}=\frac{\sum_i w_i x_i}{\sum_i w_i}
$$

4. Trilateration objective

$$
r_i(x)=||x-a_i||-d_i,
\quad
\Delta=(J^T W J+\lambda I)^{-1}J^TWr
$$

5. Tomography reconstruction (`W = diag(c_i/(1-c_i))`, consistent with consensus)

$$
\min_x ||W^{1/2}(Ax-b)||_2^2 + \alpha ||x||_2^2,
\quad
x^*=(A^TWA+\alpha I)^{-1}A^TWb
$$

## How it works

Each node runs: **scan** → **filter** → **infer** → **share** → **fuse** → **render**

### PHASE 1: passive RF sensing (automatic, continuous)

All nodes scan WiFi + BLE. Kalman-filtered RSSI gives distance estimates, attenuation reveals walls.

```
Result: blurry room graph

  +---+       +---+
  | ? |-------| ? |
  +---+       +---+
    |
    |wall
    |
  +---+
  | ? |
  +---+
```

### PHASE 2: acoustic calibration (on demand)

User triggers: `senseye calibrate`
All fixed nodes chirp in sequence (~30 seconds)
→ precise distance matrix (cm-accurate)
→ MDS gives accurate layout
→ echo profiles give room dimensions
→ wall positions snap into focus

```
Result: real floor plan

  +---------+---------+
  | kitchen | hallway |
  |  Pi1    |    *    |
  |         +--   ----+
  |              door |
  +---------+---------+
  | bedroom | living  |
  |  Pi2    |  Pi3    |
  +---------+---------+
```

### PHASE 3: motion-refined (passive, over hours)

Zone transitions from device tracking refine room connectivity.
Doorways discovered from repeated cross-room movement patterns.

```
Live dashboard:

  +---------+---------+
  | kitchen | hallway |
  |  *Pi1   |    *    |
  |   o phone  ------+
  |         |    door |
  +---------+---------+
  | bedroom | living  |
  |  *Pi2   |  *Pi3   |
  |         | oo watch|
  +---------+---------+

  motion: living ####  kitchen #
  devices: 2 tracked  nodes: 3 online
```

### Pipeline

```
  scan --> kalman --> infer --> share --> fuse --> render
                                 |         |
                            gossip mesh    |
                            (mDNS+TCP)     |
                                      +----+----+
                                      |         |
                                   trilat.   tomography
```

## Adding nodes

Any device that can run Python is a node. More nodes = more signal paths = better resolution.

```
1 node                    2-3 nodes

  Mac                     Mac-------Pi1
   |                       |  \   /  |
   |                       |   \ /   |
 router                    |    X    |
   |                       |   / \   |
   |                       |  /   \  |
 bulb                    router   bulb
                         1 link    3 links
                         per pair  per pair


4-5 nodes                 8+ nodes

 Pi1------Pi2             Pi1---Pi2---Pi3
  |\      /|               |\ / | \ /|
  | \    / |               | X  |  X |
  |  \  /  |               |/ \ | / \|
  |   \/   |              Pi4---Pi5--Pi6
  |   /\   |               |\ / | \ /|
  |  /  \  |               | X  |  X |
  | /    \ |               |/ \ | / \|
 Pi3-----Pi4              Pi7---Pi8---Pi9
  6 links                 every pair connected
  per pair                dense signal coverage
```

| Nodes | Signal paths | What you get |
|-------|-------------|-------------|
| 1 (Mac only) | sparse | Motion detection to router + BLE devices |
| 2-3 | basic mesh | Zone-level tracking, room connectivity |
| 4-5 | 6-10 links | Rough floor plan, directional motion |
| 8+ | 28+ links | Detailed room shapes, sub-room localization |

## Quick start

```bash
# UI + sensing
uv run senseye

# Headless sensor node
uv run senseye --headless --role fixed --name pi-kitchen

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
