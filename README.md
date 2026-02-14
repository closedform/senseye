# senseye

Weekend prototype for fun: a state-of-the-art sensor-fusion demo for room mapping and motion tracking. Built with help from Claude, Codex, and Antigravity. Please do not embed this in my thermostat or smoke detector.

Distributed RF/acoustic sensing for apartment mapping and live motion inference.

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

1. **Scan**: WiFi/BLE RSSI (+ optional acoustic observations).
2. **Filter**: adaptive Kalman smoothing per source-target path.
3. **Infer**: local link/device/zone beliefs with confidence.
4. **Share**: peer mesh exchange via mDNS + TCP gossip relay.
5. **Fuse**: consensus + robust trilateration + tomography.
6. **Render**: terminal dashboard with static map + dynamic overlay.

## Quick start

```bash
# UI + sensing
uv run senseye

# Headless sensor node
uv run senseye --headless

# Calibrate floorplan
uv run senseye calibrate

# Interval acoustic mode
uv run senseye --acoustic 10m
```

## Architecture

```
senseye/
  node/         # scan, filter, infer, peer, belief
  fusion/       # consensus, trilateration, tomography
  mapping/      # static map + dynamic world state
  calibration.py
  main.py
```

## Requirements

- Python 3.11+
- macOS or Linux
- WiFi and/or BLE hardware
- Optional: speaker + mic for acoustic mode
