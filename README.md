# senseye

Distributed RF sensing that maps your apartment and detects motion using the devices you already have.

Every WiFi access point, BLE device, smart bulb, and speaker in your space becomes a sensor node. Senseye builds a map of your apartment from signal attenuation patterns and shows live motion on top.

## How it works

1. **Scan** — each node measures WiFi and BLE signal strength to every device it can see
2. **Infer** — each node independently detects motion, estimates distances, and classifies zones
3. **Share** — nodes discover each other via mDNS and exchange beliefs over TCP
4. **Fuse** — each node combines its beliefs with peers to produce a shared map
5. **Render** — a live terminal dashboard shows the apartment with motion overlay

```
╔══════════════════════════════════════╗
║  senseye                    12:34   ║
║                                      ║
║   ┌──────────┬────────────┐          ║
║   │ kitchen  │  hallway ░░│          ║
║   │  ◈Pi1    │    ◆router │          ║
║   │          └────   ─────┤          ║
║   │     ●phone    door    │          ║
║   ├─────   ──┬────────────┤          ║
║   │ bedroom  │ living rm  │          ║
║   │  ◈Pi2    │  ◈AppleTV  │          ║
║   │          │  ░░●watch  │          ║
║   └──────────┴────────────┘          ║
║                                      ║
║  motion: living room ██░░  kitchen ░ ║
╚══════════════════════════════════════╝
```

## Quick start

```bash
# Install
pip install -e .

# Run on your Mac (sensor + UI)
python -m senseye

# Run on a Raspberry Pi (headless sensor)
python -m senseye --headless

# Calibrate the map (acoustic ping sequence)
python -m senseye calibrate

# Nodes discover each other automatically via mDNS
```

## Adding nodes

Any device that can run Python is a node. Place it somewhere, start senseye, and the mesh absorbs it:

```bash
# On a Raspberry Pi
python -m senseye --headless --role fixed --name pi-kitchen

# The node auto-discovers peers via mDNS
# The map recalibrates to incorporate the new sensor
# More nodes = more signal paths = better resolution
```

Resolution scales with the number of fixed nodes:

| Nodes | What you get |
|-------|-------------|
| 1 (Mac only) | Motion detection on signal paths to router + BLE devices |
| 2-3 | Zone-level motion tracking, room connectivity map |
| 4-5 | Rough floor plan from signal attenuation, directional motion |
| 8+ | Detailed room shapes, sub-room motion localization |

## Acoustic mode

Nodes with speakers and microphones can ping each other ultrasonically (18-22kHz, inaudible) for centimeter-accurate distance measurement.

```bash
python -m senseye --acoustic off         # no chirps
python -m senseye --acoustic on-demand   # only during calibration
python -m senseye --acoustic 10m         # ping every 10 minutes
python -m senseye --acoustic 1h          # ping every hour
```

## Architecture

```
senseye/
    node/           # autonomous sensor agent (scan → filter → infer → share)
    fusion/         # distributed inference (consensus, tomography, trilateration)
    mapping/
        static/     # apartment layout (walls, rooms — built once, refined rarely)
        dynamic/    # live motion + device positions (updated every second)
    ui/             # terminal dashboard
    config.py       # runtime configuration
    protocol.py     # wire format (newline-delimited JSON over TCP)
    main.py         # entry point
```

Every node runs the same code. The map is a static background built from signal data. Motion is drawn on top.

## Dependencies

```
bleak       # BLE scanning
zeroconf    # mDNS discovery
numpy       # signal processing + linear algebra
rich        # terminal UI
```

## Requirements

- Python 3.11+
- macOS or Linux
- WiFi and/or Bluetooth hardware
- Optional: speaker + microphone for acoustic sensing
