# Senseye Design Document

## 1. Overview

Senseye is a distributed RF/acoustic sensing system for indoor mapping and motion inference. Each node runs the same pipeline:

`SCAN -> FILTER -> INFER -> SHARE -> FUSE -> UPDATE WORLD -> RENDER`

The modeling stack is uncertainty-aware end-to-end:

- Adaptive 2-state Kalman filtering per link
- Confidence-aware local inference
- Inverse-variance consensus across peers
- Robust weighted trilateration with outlier rejection
- Confidence-weighted ridge tomography with adaptive regularization

## 2. Code Map

```
senseye/
  node/
    scanner.py       # WiFi/BLE/acoustic observations
    acoustic.py      # chirps, matched filter, ToF primitives
    filter.py        # adaptive CV Kalman per signal path
    inference.py     # local link/device/zone beliefs
    peer.py          # mDNS + TCP mesh + gossip relay
    belief.py        # belief dataclasses and serialization
  fusion/
    consensus.py     # uncertainty-weighted fusion
    trilateration.py # robust weighted Gauss-Newton solver
    tomography.py    # weighted ridge RTI reconstruction
    runtime.py       # runtime wrappers over fusion modules
    acoustic_range.py
    graph.py
  mapping/
    static/
      layout.py      # MDS localization + anchoring
      walls.py
      topology.py
      floorplan.py
    dynamic/
      motion.py
      devices.py
      state.py
  calibration.py     # active map calibration and reconstruction
  main.py            # runtime loop and orchestration
```

## 3. Measurement Models

### 3.0 Notation

Common symbols used throughout:

- $z_k$: measurement at step $k$
- $x_k$: latent state at step $k$
- $P_k$: state covariance
- $r_i$: residual for observation $i$
- $J$: Jacobian matrix of residuals
- $W$: diagonal weight matrix
- $A$: forward model / design matrix
- $b$: measured observation vector
- $x$: solved parameter vector (context-dependent)
- $\kappa(\cdot)$: condition number

### 3.1 RF Path-Loss Model

For RSSI-to-distance inversion and expected free-space RSSI, Senseye uses:

$$
\mathrm{RSSI}_{\mathrm{expected}}(d) = -\left(10 n \log_{10}(d) + A\right)
$$

$$
d = 10^{\frac{-\mathrm{RSSI} - A}{10 n}}
$$

where $n$ is the path-loss exponent and $A$ is the 1 m intercept (dBm magnitude). The canonical indoor parameters are defined in `inference.py`:

- `PATHLOSS_N = 2.5` (indoor propagation)
- `PATHLOSS_A = 45.0` (1 m reference level)

During calibration wall detection, the free-space exponent $n = 2.0$ is used instead so that all indoor propagation effects are visible as excess attenuation (see section 10).

This is the bridge from raw signal strength to a rough physical distance baseline, which is required before attenuation can be interpreted as likely obstruction or wall evidence.

### 3.2 Acoustic ToF Model

One-way distance from time-of-flight:

$$
d = c \cdot t_{\mathrm{tof}}
$$

with $c = 343\,\mathrm{m/s}$ (approximate speed of sound at ~20 °C), defined canonically in `node/acoustic.py` and re-exported by `fusion/acoustic_range.py`.

In peer ranging, ToF is estimated from matched-filter peak timing after accounting for scheduled chirp delay and approximate network latency compensation.

Acoustic ToF gives much tighter distance estimates than RSSI, so it is the key source of geometric accuracy during calibration.

### 3.3 Acoustic Signature Channelization

Each audio-capable node is assigned a deterministic chirp-band signature from its node identifier $i$:

$$
k = H(i) \bmod N_c
$$

$$
f_{\mathrm{start}} = f_0 + k\Delta f, \quad f_{\mathrm{end}} = f_{\mathrm{start}} + \Delta f
$$

where $H$ is SHA-256 interpreted as an integer, $N_c = 6$, $f_0 = 17{,}000\,\mathrm{Hz}$, and $\Delta f = 1{,}000\,\mathrm{Hz}$.

This yields channels:

- `17-18 kHz`
- `18-19 kHz`
- `19-20 kHz`
- `20-21 kHz`
- `21-22 kHz`
- `22-23 kHz`

During peer ranging, the requesting node expects the target peer chirp on that peer's deterministic band and performs matched filtering against that signature template.

Per-node acoustic signatures reduce emitter ambiguity, so simultaneous or nearby chirps are less likely to be misattributed.

## 4. Per-Link Adaptive Kalman Filter (`node/filter.py`)

Each `(source_id, target_id)` path has a constant-velocity state:

$$
\mathbf{x}_k = \begin{bmatrix} \mathrm{rssi}_k \\ \dot{\mathrm{rssi}}_k \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 1 & dt \\ 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}
$$

Continuous-acceleration-inspired process covariance:

$$
\mathbf{Q} = q \begin{bmatrix} \frac{dt^4}{4} & \frac{dt^3}{2} \\ \frac{dt^3}{2} & dt^2 \end{bmatrix}
$$

Prediction/update:

$$
\hat{\mathbf{x}}^-_k = \mathbf{F}\hat{\mathbf{x}}_{k-1}, \quad \mathbf{P}^-_k = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^T + \mathbf{Q}
$$

$$
\mathbf{y}_k = z_k - \mathbf{H}\hat{\mathbf{x}}^-_k, \quad \mathbf{S}_k = \mathbf{H}\mathbf{P}^-_k\mathbf{H}^T + R
$$

$$
\mathbf{K}_k = \mathbf{P}^-_k\mathbf{H}^T\mathbf{S}_k^{-1}, \quad \hat{\mathbf{x}}_k = \hat{\mathbf{x}}^-_k + \mathbf{K}_k\mathbf{y}_k
$$

Dimensions:

- $x_k \in \mathbb{R}^2$
- $F \in \mathbb{R}^{2\times 2}$
- $H \in \mathbb{R}^{1\times 2}$
- $P_k, Q \in \mathbb{R}^{2\times 2}$
- $S_k, R \in \mathbb{R}^{1\times 1}$

Covariance update uses the **Joseph form** for numerical stability, guaranteeing symmetry and positive-definiteness:

$$
\mathbf{P}_k = (\mathbf{I}-\mathbf{K}_k\mathbf{H})\mathbf{P}^-_k(\mathbf{I}-\mathbf{K}_k\mathbf{H})^T + \mathbf{K}_k R \mathbf{K}_k^T
$$

Adaptive jump handling:

$$
z_{\mathrm{score}} = \frac{|y_k|}{\sqrt{S_k}}
$$

If $z_{\mathrm{score}} > \mathrm{threshold}$, use $Q_{\mathrm{scaled}} = \mathrm{scaling\_factor}\,Q$ for that step to quickly track abrupt environment changes.

Interpretation: large normalized innovation means the constant-velocity prior is no longer adequate (e.g., abrupt path change), so process noise is temporarily increased to reduce lag.

This keeps link estimates stable when noise is small, but still lets the system react quickly when the environment actually changes.

## 5. Local Inference (`node/inference.py`)

### 5.1 Motion from Variance

For each device history window $W$:

$$
\mathrm{var}(W) = \frac{1}{|W|}\sum_{x \in W}(x-\bar{x})^2
$$

Motion is detected when $\mathrm{var}(W) > \mathrm{motion\_threshold}$.

### 5.2 Link Attenuation

When node and target positions are known:

$$
\mathrm{attenuation} = \max(0, \mathrm{RSSI}_{\mathrm{expected}}(d) - \mathrm{RSSI}_{\mathrm{filtered}})
$$

### 5.3 Confidence Model

Local link confidence combines sample support and measurement quality:

- Sample confidence:

$$
c_{\mathrm{samples}} = \min\left(\frac{N}{W}, 1\right)
$$

- Innovation penalty:

$$
p_{\mathrm{innov}} = \frac{1}{1 + |\mathrm{innovation}|/8}
$$

- RF confidence:

$$
c_{\mathrm{rf}} = c_{\mathrm{samples}} \cdot p_{\mathrm{innov}}
$$

- Acoustic confidence (uses SNR when available):

$$
c_{\mathrm{acoustic}} = 0.4 c_{\mathrm{samples}} + 0.6 c_{\mathrm{snr}}
$$

where $c_{\mathrm{snr}}$ is a clipped affine map of peak SNR.

### 5.4 Zone Beliefs

Given zone-crossing links, zone motion and occupancy use aggregate heuristics:

$$
P(\mathrm{motion}|\mathrm{zone}) = \frac{N_{\mathrm{moving\ links}}}{N_{\mathrm{links}}}
$$

$$
P(\mathrm{occupied}|\mathrm{zone}) = \min\left(\frac{\mathrm{avg\ attenuation}}{20\,\mathrm{dB}}, 1\right)
$$

This stage converts noisy per-device/per-link measurements into actionable local semantics like "motion" and "occupancy" that downstream fusion can combine.

## 6. Peer Mesh and Gossip (`node/peer.py`, `main.py`)

- Discovery: mDNS `_senseye._tcp.local.`
- Transport: newline-delimited JSON over TCP
- Relay control: `(node_id, sequence_number)` dedup + `hop_count` TTL

Beliefs are relayed iff `hop_count > 0`, with `hop_count := hop_count - 1` on relay.

Gossip plus TTL gives each node enough shared context for fusion without flooding the network or creating belief loops.

## 7. Consensus Fusion (`fusion/consensus.py`)

Let confidence $c \in (0,1)$ map to variance after implementation clamping:

$$
c_{\mathrm{eff}} = \min(\max(c, 0.01), 0.99)
$$

$$
\sigma^2(c) = \frac{1-c_{\mathrm{eff}}}{c_{\mathrm{eff}}} + \epsilon
$$

Precision:

$$
\pi(c) = \frac{1}{\sigma^2(c)}
$$

Weighted mean for scalar quantity $x$:

$$
\hat{x} = \frac{\sum_i \pi_i x_i}{\sum_i \pi_i}
$$

### 7.1 Link Fusion

- `attenuation` and `motion probability` are precision-weighted averages.
- Base fused confidence:

$$
c_{\mathrm{base}} = \frac{\sum_i \pi_i}{1 + \sum_i \pi_i}
$$

- Disagreement penalty from weighted variance $v$:

$$
\mathrm{penalty} = \frac{1}{1 + s v}
$$

$$
c_{\mathrm{fused}} = c_{\mathrm{base}} \cdot \mathrm{penalty}
$$

Weighted disagreement variance used in the penalty term:

$$
v = \frac{\sum_i \pi_i (x_i - \hat{x})^2}{\sum_i \pi_i}
$$

where $x_i$ is per-peer attenuation and $\hat{x}$ is its weighted mean.

### 7.2 Device Fusion

Device precision is derived from link confidence + range-based reliability. Distance estimates are additionally down-weighted by range squared:

$$
w_{d,i} = \frac{\pi_i}{\max(d_i,1)^2}
$$

This prevents long-range RSSI distance estimates from dominating.

Final device scalar estimates follow weighted means:

$$
\hat{rssi} = \frac{\sum_i w_i \, rssi_i}{\sum_i w_i}, \qquad \hat{d} = \frac{\sum_i w_{d,i} \, d_i}{\sum_i w_{d,i}}
$$

### 7.3 Zone Fusion

Zone confidence proxy is derived from certainty away from 0.5:

$$
\mathrm{certainty} = 2\max(|o-0.5|, |m-0.5|)
$$

Zone confidence used for fusion is:

$$
c_{\mathrm{zone}} = \min(\max(0.2 + 0.8 \cdot \mathrm{certainty}, 0.05), 0.99)
$$

Zone occupied/motion beliefs are fused with inverse-variance weighting using $c_{\mathrm{zone}}$.

Consensus lets multiple imperfect nodes agree on a single estimate while naturally trusting high-confidence peers more than low-confidence ones.

## 8. Robust Trilateration (`fusion/trilateration.py`)

Given anchors $a_i$ and measured distances $d_i$, estimate position $x$.

Residual:

$$
r_i(x) = \lVert x-a_i \rVert - d_i
$$

with $x = [x, y]^T$ and anchor $a_i = [a_{x,i}, a_{y,i}]^T$.

Range-dependent noise model:

$$
\sigma_i = \max(0.35, 0.08 d_i + 0.2)
$$

Base weight:

$$
w_i^{\mathrm{base}} = \frac{1}{\sigma_i^2}
$$

Tukey robust factor with cutoff $c_i = 2.5\,\sigma_i$:

$$
\omega_i = \begin{cases} \left(1 - (|r_i|/c_i)^2\right)^2 & |r_i| < c_i \\ 0 & |r_i| \ge c_i \end{cases}
$$

Final IRLS weight:

$$
w_i = w_i^{\mathrm{base}} \omega_i
$$

Gauss-Newton step:

$$
\Delta = (J^T W J + \lambda I)^{-1} J^T W r, \quad x \leftarrow x - \Delta
$$

Jacobian row for observation $i$:

$$
J_i = [ (x-a_{x,i})/\hat{d}_i,\ (y-a_{y,i})/\hat{d}_i ]
$$

$$
\hat{d}_i = \max(\sqrt{(x-a_{x,i})^2 + (y-a_{y,i})^2}, \epsilon)
$$

with a small epsilon floor in implementation to avoid singularity.

Outlier handling:

- Evaluate full set and selected subsets (leave-one-out and size-3 subsets when small).
- Score candidates by inlier count and clipped normalized residual score.
- Refit on inliers ($|r_i|/\sigma_i \le 2.5$) when possible.

Normalized residual and score:

$$
\rho_i = \frac{|r_i|}{\sigma_i}, \qquad \mathrm{score} = \frac{1}{N}\sum_{i=1}^{N}\min(\rho_i^2, 9)
$$

Robust trilateration keeps position estimates usable even when some range inputs are wrong, stale, or multipath-corrupted.

## 9. Tomographic Reconstruction (`fusion/tomography.py`)

Build linear model over grid cells:

$$
A x \approx b
$$

- $x$: per-cell attenuation field
- $b$: measured link excess attenuation
- $A$: link-to-cell influence matrix (Gaussian kernel around each link segment)

Each row is normalized so links contribute by spatial distribution, not raw row magnitude.

For link $i$ and cell $j$, unnormalized influence is:

$$
\tilde{A}_{ij} = \begin{cases} e^{-\frac{d_{ij}^2}{2\sigma_k^2}}, & d_{ij} \le r \\ 0, & d_{ij} > r \end{cases}
$$

where $d_{ij}$ is point-to-segment distance, $r$ is influence radius, and $\sigma_k = r/2$.

Row normalization:

$$
A_{ij} = \frac{\tilde{A}_{ij}}{\sum_j \tilde{A}_{ij}}
$$

Confidence weights use the same inverse-variance mapping as consensus fusion (section 7):

$$
c_i^{\mathrm{eff}} = \min(\max(c_i, 0.01), 0.99), \quad W = \mathrm{diag}\left(\frac{c_i^{\mathrm{eff}}}{1-c_i^{\mathrm{eff}}}\right)
$$

Weighted ridge objective:

$$
\min_x \lVert W^{1/2}(Ax-b) \rVert_2^2 + \alpha \lVert x \rVert_2^2
$$

Closed form:

$$
x^* = (A^T W A + \alpha I)^{-1} A^T W b
$$

Equivalent whitened least-squares form used numerically:

$$
\bar{A} = W^{1/2}A,\quad \bar{b}=W^{1/2}b,\quad x^* = (\bar{A}^T\bar{A} + \alpha I)^{-1}\bar{A}^T\bar{b}
$$

Adaptive regularization:

$$
\alpha \propto \frac{n_{\mathrm{cells}}}{n_{\mathrm{links}}} (1 + \log_{10}(\kappa(A^TWA)))
$$

clipped to $[0.05, 5.0]$.

Practical role: when the inverse problem is underdetermined or ill-conditioned (many cells, few links), larger $\alpha$ suppresses unstable high-frequency artifacts in the attenuation map.

Tomography turns many link-level attenuations into a spatial heatmap, which is what allows wall and obstruction structure to appear in map space.

## 10. Static Map Generation (`calibration.py`, `mapping/static/*`)

### 10.1 Distance Fusion

RF and acoustic distance matrices are merged with acoustic priority when available:

$$
D_{ij} = \begin{cases} D^{\mathrm{acoustic}}_{ij}, & \mathrm{if\ measured\ acoustically} \\ D^{\mathrm{rf}}_{ij}, & \mathrm{otherwise} \end{cases}
$$

For unknown pairwise distances between non-reference nodes (where only distances to the reference are known), the expected distance under a uniform angular prior is used:

$$
\hat{D}_{ij} = \sqrt{D_{0i}^2 + D_{0j}^2}
$$

This follows from the law of cosines with $E[\cos(\theta)] = 0$ over uniform $[0, 2\pi]$.

When peers share direct acoustic ranges, missing acoustic entries are daisy-chained over the peer graph with bounded-hop shortest paths:

$$
D^{\mathrm{chain}}_{ij} = \min_{p \in \mathcal{P}_{ij},\; h(p)\le H} \sum_{(u,v)\in p} D^{\mathrm{acoustic}}_{uv}
$$

with hop limit $H = 3$. Chained values are only used when a direct acoustic measurement for $(i,j)$ is unavailable.

### 10.2 Wall Detection Model

Calibration uses the free-space exponent $n = 2.0$ (rather than the indoor $n = 2.5$) as the attenuation baseline. This maximizes sensitivity to structural obstructions because the free-space model predicts stronger signal at any given distance; any signal loss beyond this baseline is attributed to walls:

$$
\mathrm{excess} = \max\left(0,\; \mathrm{RSSI}_{\mathrm{free-space}}(d) - \mathrm{RSSI}_{\mathrm{measured}}\right)
$$

The 1 m intercept $A = 45$ is shared with the indoor model since it reflects hardware characteristics, not propagation environment.

### 10.3 MDS Localization

From pairwise distances $D$, classical MDS:

$$
B = -\frac{1}{2} J D^{\circ 2} J, \quad J = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T
$$

Take top-2 eigenpairs of $B$ for 2D coordinates, then anchor local node at $(0,0)$.

If $B = V \Lambda V^T$ (eigendecomposition), coordinates are:

$$
X_{2D} = V_{:,1:2}\Lambda_{1:2,1:2}^{1/2}
$$

Negative eigenvalues from noisy/non-Euclidean distances are clamped to zero before square root.

### 10.4 Walls and Rooms

- Link attenuation produces midpoint-perpendicular wall candidates.
- Tomography peak cells also produce wall candidates.
- Rooms are inferred by connectivity with wall intersections.

This stage converts abstract signal evidence into a human-usable floorplan representation (walls, rooms, and topology).

## 11. Dynamic World Update (`mapping/dynamic/*`)

Zone motion intensity decays exponentially:

$$
I_t = I_{t-1} e^{-\lambda dt}
$$

Then merged with current belief using:

$$
I_t \leftarrow \max(I_t, P_{\mathrm{motion,zone}})
$$

Devices are assigned to nearest room center when position estimates exist.

Dynamic updates keep the map useful after calibration by continuously layering live motion and device activity onto static geometry.

## 12. Recalibration Policy (`main.py`)

Recalibration can be triggered by:

- No floorplan
- Peer set change
- Interval acoustic schedule
- RSSI drift

Average RSSI drift against calibration baseline:

$$
\mathrm{avg\ drift} = \frac{1}{|C|}\sum_{d \in C} |\mathrm{RSSI}_{\mathrm{now}}(d)-\mathrm{RSSI}_{\mathrm{baseline}}(d)|
$$

where $C$ is the set of devices present in both snapshots. Drift triggers recalibration when enough common devices exist and drift exceeds threshold.

Automatic recalibration prevents the map from silently degrading as the environment and peer set evolve over time.

## 13. Wire Protocol (`protocol.py`)

Newline-delimited JSON messages:

```json
{"type":"announce","node_id":"..."}
{"type":"belief","node_id":"...","timestamp":...,"sequence_number":...,"hop_count":...,"links":{...},"devices":{...},"zones":{...},"acoustic_ranges":{"peer-a":2.3}}
{"type":"acoustic_ping","request_id":"...","delay_s":0.2,"sample_rate":48000,"freq_start":18000,"freq_end":19000,"chirp_duration":0.01}
{"type":"acoustic_pong","request_id":"...","ok":true,"error":""}
```

A simple explicit wire protocol keeps nodes interoperable and debuggable as the sensing/fusion stack gets more sophisticated.
