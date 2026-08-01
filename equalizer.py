"""
Linear MMSE Channel Estimation and Equalization module.
Calculates continuous time-frequency channel response H_tf (Eq. 5) and equalizes received signals.
"""
import numpy as np
from config import M, N, DELTA_F, T_TOTAL
from channel import ChannelPath


def compute_H_tf_from_paths(paths: list[ChannelPath], m_subcarriers: int = M,
                            n_symbols: int = N, delta_f: float = DELTA_F,
                            t_total: float = T_TOTAL) -> np.ndarray:
    """
    Computes exact Time-Frequency channel matrix H_tf[m, n] according to Eq. (5).
    """
    H_tf = np.zeros((m_subcarriers, n_symbols), dtype=np.complex128)

    m_idx = np.arange(m_subcarriers)[:, np.newaxis]   # Subcarrier indices
    n_idx = np.arange(n_symbols)[np.newaxis, :]       # Symbol indices

    for path in paths:
        tau_i = path.delay_samples / (m_subcarriers * delta_f)
        nu_i = path.doppler_hz

        phase_const = -1j * 2.0 * np.pi * nu_i * tau_i
        phase_tf = -1j * 2.0 * np.pi * (m_idx * delta_f * tau_i - nu_i * n_idx * t_total)

        H_tf += path.gain * np.exp(phase_const) * np.exp(phase_tf)

    return H_tf


def get_lmmse_channel_estimate(H_tf: np.ndarray, snr_linear: float, scheme: str,
                              seed: int | None = None) -> np.ndarray:
    """
    Models Linear MMSE channel estimation from pilots.
    OFDM error variance = 1 / (1 + SNR)
    OTFS error variance = 1 / (N * (1 + SNR)) due to 2D DD Processing Gain
    """
    rng = np.random.default_rng(seed)

    if scheme == "OFDM":
        err_var = 1.0 / (1.0 + snr_linear)
    else:  # OTFS
        err_var = 1.0 / (N * (1.0 + snr_linear))

    err_std = np.sqrt(err_var / 2.0)
    h_noise = err_std * (rng.standard_normal(H_tf.shape) + 1j * rng.standard_normal(H_tf.shape))

    H_hat = H_tf + h_noise

    # Apply LMMSE shrinkage
    h_sq = np.abs(H_hat) ** 2
    return H_hat * (h_sq / (h_sq + err_var))


def mmse_equalize_tf(Y_tf: np.ndarray, H_est: np.ndarray, snr_linear: float) -> np.ndarray:
    """Applies point-wise LMMSE equalization in the Time-Frequency domain."""
    h_sq = np.abs(H_est) ** 2
    return Y_tf * np.conj(H_est) / (h_sq + 1.0 / snr_linear)