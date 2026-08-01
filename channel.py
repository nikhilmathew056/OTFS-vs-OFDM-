"""
Delay-Doppler multipath wireless channel model.
Implements time-varying multipath channels with delay spread and Doppler shifts.
"""
from dataclasses import dataclass
import numpy as np
from config import MOBILITY_SCENARIOS, SAMPLING_RATE


@dataclass
class ChannelPath:
    gain: complex
    delay_samples: int
    doppler_hz: float


def generate_channel(scenario_name: str, seed: int | None = None) -> list[ChannelPath]:
    """Generates a list of ChannelPath objects according to the scenario."""
    rng = np.random.default_rng(seed)

    if scenario_name == "los":
        return [ChannelPath(gain=1.0 + 0.0j, delay_samples=0, doppler_hz=0.0)]

    delays = [0, 3, 7]                  # Tap delays in sample units
    power_db = [0.0, -3.0, -6.0]        # Relative power profile in dB
    linear_powers = 10.0 ** (np.array(power_db) / 10.0)

    # Normalize path gains so total channel power E[sum |g_i|^2] = 1.0
    linear_gains = np.sqrt(linear_powers / np.sum(linear_powers))

    # Apply random uniform phase offset
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(delays))
    complex_gains = linear_gains * np.exp(1j * phases)

    if scenario_name == "static_multipath":
        return [
            ChannelPath(gain=g, delay_samples=d, doppler_hz=0.0)
            for g, d in zip(complex_gains, delays)
        ]

    # Mobility scenarios
    f_d = MOBILITY_SCENARIOS[scenario_name]["f_d"]
    dopplers = [f_d, 0.5 * f_d, -0.5 * f_d]

    return [
        ChannelPath(gain=g, delay_samples=d, doppler_hz=nu)
        for g, d, nu in zip(complex_gains, delays, dopplers)
    ]


def apply_channel(x: np.ndarray, paths: list[ChannelPath], fs: float = SAMPLING_RATE,
                  snr_db: float | None = 30.0, seed: int | None = None) -> np.ndarray:
    """
    Applies time-varying channel response (Eq. 8) and adds complex AWGN.
    """
    rng = np.random.default_rng(seed)
    L = len(x)
    n_indices = np.arange(L)
    t = n_indices / fs

    y = np.zeros(L, dtype=np.complex128)
    for path in paths:
        x_delayed = np.roll(x, path.delay_samples)
        doppler_phase = np.exp(1j * 2.0 * np.pi * path.doppler_hz * t)
        y += path.gain * doppler_phase * x_delayed

    # Add AWGN if snr_db is provided and finite
    if snr_db is not None and not np.isinf(snr_db):
        p_signal = np.mean(np.abs(y) ** 2)
        if p_signal > 0:
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = p_signal / snr_linear
            noise_std = np.sqrt(noise_power / 2.0)
            noise = noise_std * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
            y += noise

    return y