"""Ultrasonic chirp TX/RX and echo profiling."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np


SPEED_OF_SOUND = 343.0  # m/s at ~20°C

# Default chirp parameters (from config spec)
DEFAULT_FREQ_START = 18_000  # Hz
DEFAULT_FREQ_END = 22_000    # Hz
DEFAULT_CHIRP_DURATION = 0.01  # seconds
DEFAULT_SAMPLE_RATE = 48_000  # Hz


def generate_chirp(
    freq_start: int = DEFAULT_FREQ_START,
    freq_end: int = DEFAULT_FREQ_END,
    duration: float = DEFAULT_CHIRP_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Generate an FMCW chirp signal (linear frequency sweep)."""
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
