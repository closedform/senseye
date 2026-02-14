"""Ultrasonic chirp TX/RX and echo profiling."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at ~20°C

# Default chirp parameters (from config spec)
DEFAULT_FREQ_START = 18_000  # Hz
DEFAULT_FREQ_END = 22_000    # Hz
DEFAULT_CHIRP_DURATION = 0.01  # seconds
DEFAULT_SAMPLE_RATE = 48_000  # Hz

# Channel definitions
CHANNEL_BASE_FREQ = 17_000
CHANNEL_WIDTH = 1_000
NUM_CHANNELS = 6


def get_chirp_params(node_id: str) -> tuple[int, int]:
    """Get frequency range for a node's deterministic channel.

    Channels:
      0: 17k - 18k
      1: 18k - 19k
      ...
      5: 22k - 23k
    """
    # Deterministic hash to channel index
    h = int(hashlib.sha256(node_id.encode()).hexdigest(), 16)
    channel_idx = h % NUM_CHANNELS

    f_start = CHANNEL_BASE_FREQ + (channel_idx * CHANNEL_WIDTH)
    f_end = f_start + CHANNEL_WIDTH
    return f_start, f_end


def generate_chirp(
    freq_start: int | None = None,
    freq_end: int | None = None,
    duration: float = DEFAULT_CHIRP_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Generate an FMCW chirp signal (linear frequency sweep)."""
    if freq_start is None:
        freq_start = DEFAULT_FREQ_START
    if freq_end is None:
        freq_end = DEFAULT_FREQ_END

    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Linear chirp: instantaneous frequency = freq_start + (freq_end - freq_start) * t / duration
    # Phase integral: 2*pi * (freq_start * t + 0.5 * (freq_end - freq_start) * t^2 / duration)
    sweep_rate = (freq_end - freq_start) / duration
    phase = 2.0 * np.pi * (freq_start * t + 0.5 * sweep_rate * t**2)
    return np.sin(phase).astype(np.float32)


def matched_filter(received: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cross-correlate received signal with chirp template. Returns correlation envelope."""
    # Normalize template
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    # Full cross-correlation via FFT
    n = len(received) + len(template_norm) - 1
    fft_size = 1 << (n - 1).bit_length()  # next power of 2
    R = np.fft.rfft(received, fft_size)
    T = np.fft.rfft(template_norm, fft_size)
    corr = np.fft.irfft(R * np.conj(T), fft_size)
    return np.abs(corr[:n])


def tof_to_distance(tof_seconds: float) -> float:
    """Convert one-way time of flight to distance in meters."""
    return tof_seconds * SPEED_OF_SOUND


def find_peak_tof(
    correlation: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    template_length: int | None = None,
) -> float | None:
    """Find time-of-flight from correlation peak.

    Returns ToF in seconds, or None if no clear peak is found.
    The template_length offset accounts for the matched filter delay.
    """
    if len(correlation) == 0:
        return None
    # Skip the direct/zero-lag region (template length) to find the first echo
    skip = template_length if template_length is not None else 0
    if skip >= len(correlation):
        return None
    search_region = correlation[skip:]
    peak_idx = int(np.argmax(search_region))
    peak_val = search_region[peak_idx]
    # Reject if peak is not significantly above noise floor
    noise_floor = np.median(search_region)
    if peak_val < noise_floor * 3.0:
        return None
    return (skip + peak_idx) / sample_rate


@dataclass(frozen=True, slots=True)
class EchoProfile:
    distance: float | None  # meters, None if no echo detected
    tof: float | None       # seconds
    peak_snr: float         # peak / noise floor ratio
    timestamp: float
    raw_correlation: np.ndarray


@dataclass(frozen=True, slots=True)
class ListenResult:
    tof: float | None
    peak_snr: float
    record_started_at: float
    timestamp: float


def _try_import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        return None


async def echo_profile(
    chirp: np.ndarray | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    record_duration: float = 0.1,
) -> EchoProfile | None:
    """Play chirp on speaker, record on mic, extract echo profile.

    Returns None if sounddevice is not available.
    """
    sd = _try_import_sounddevice()
    if sd is None:
        return None

    if chirp is None:
        chirp = generate_chirp(sample_rate=sample_rate)

    n_record = int(record_duration * sample_rate)
    # Pad chirp to full record length for simultaneous play+record
    padded = np.zeros(n_record, dtype=np.float32)
    padded[:len(chirp)] = chirp

    loop = asyncio.get_running_loop()
    recorded = await loop.run_in_executor(
        None,
        lambda: sd.playrec(
            padded.reshape(-1, 1),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocking=True,
        ),
    )
    recorded = recorded.flatten()

    corr = matched_filter(recorded, chirp)
    now = time.time()

    # Find echo peak (skip the direct chirp region)
    tof = find_peak_tof(corr, sample_rate, template_length=len(chirp))

    noise_floor = float(np.median(np.abs(corr))) + 1e-12
    peak_val = float(np.max(np.abs(corr[len(chirp):]))) if len(corr) > len(chirp) else 0.0
    snr = peak_val / noise_floor

    distance = tof_to_distance(tof) if tof is not None else None

    return EchoProfile(
        distance=distance,
        tof=tof,
        peak_snr=snr,
        timestamp=now,
        raw_correlation=corr,
    )


async def play_chirp(
    chirp: np.ndarray | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    delay: float = 0.0,
) -> bool:
    """Play a chirp without recording. Returns False if audio backend is unavailable."""
    sd = _try_import_sounddevice()
    if sd is None:
        return False

    if chirp is None:
        chirp = generate_chirp(sample_rate=sample_rate)

    if delay > 0:
        await asyncio.sleep(delay)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: sd.play(
            chirp,
            samplerate=sample_rate,
            blocking=True,
        ),
    )
    return True


async def listen_for_chirp(
    chirp: np.ndarray | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    record_duration: float = 0.5,
    template_length: int | None = None,
) -> ListenResult | None:
    """Record audio and detect chirp arrival via matched filter."""
    sd = _try_import_sounddevice()
    if sd is None:
        return None

    if chirp is None:
        chirp = generate_chirp(sample_rate=sample_rate)
    if template_length is None:
        template_length = 0

    n_record = int(record_duration * sample_rate)
    loop = asyncio.get_running_loop()
    record_started_at = loop.time()
    recorded = await loop.run_in_executor(
        None,
        lambda: sd.rec(
            n_record,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocking=True,
        ),
    )
    recorded = recorded.flatten()
    corr = matched_filter(recorded, chirp)
    tof = find_peak_tof(corr, sample_rate, template_length=template_length)

    noise_floor = float(np.median(np.abs(corr))) + 1e-12
    peak_val = float(np.max(np.abs(corr))) if len(corr) else 0.0
    peak_snr = peak_val / noise_floor

    return ListenResult(
        tof=tof,
        peak_snr=peak_snr,
        record_started_at=record_started_at,
        timestamp=time.time(),
    )


def identify_chirps(
    recording: np.ndarray,
    candidates: list[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, float]:
    """Identify which nodes are present in a recording.

    Args:
        recording: raw audio samples
        candidates: list of node_ids to check for
        sample_rate: audio sample rate

    Returns:
        dict of {node_id: peak_snr} for detected nodes (SNR > 3.0)
    """
    results = {}
    for node_id in candidates:
        f_start, f_end = get_chirp_params(node_id)
        # Generate template for this node
        template = generate_chirp(
            freq_start=f_start,
            freq_end=f_end,
            sample_rate=sample_rate
        )

        corr = matched_filter(recording, template)

        # Check for peak
        peak_val = float(np.max(np.abs(corr))) if len(corr) else 0.0
        noise_floor = float(np.median(np.abs(corr))) + 1e-12
        snr = peak_val / noise_floor

        if snr > 3.0:
            results[node_id] = snr

    return results
