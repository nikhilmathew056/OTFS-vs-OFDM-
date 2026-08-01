"""
OFDM Modulator and Demodulator.
Includes cyclic prefix insertion/removal and IFFT/FFT processing.
"""
import numpy as np
from config import M, N, CP_LEN


def ofdm_modulate(freq_symbols: np.ndarray) -> np.ndarray:
    """
    freq_symbols: Shape (M, N) complex frequency-domain matrix.
    Returns 1D complex time-domain vector of length N * (M + CP_LEN).
    """
    # Normalized IFFT along subcarriers (rows)
    time_symbols = np.fft.ifft(freq_symbols, axis=0, norm='ortho')  # Shape (M, N)

    # Prepend Cyclic Prefix (last CP_LEN samples of each OFDM column)
    cp = time_symbols[-CP_LEN:, :]                                   # Shape (CP_LEN, N)
    time_with_cp = np.vstack([cp, time_symbols])                   # Shape (M + CP_LEN, N)

    # Flatten column by column (time ordering)
    return time_with_cp.flatten(order='F')


def ofdm_demodulate(x_time: np.ndarray) -> np.ndarray:
    """
    x_time: 1D complex time-domain vector of length N * (M + CP_LEN).
    Returns frequency-domain matrix Y_tf of shape (M, N).
    """
    # Reshape into matrix where each column is an OFDM symbol with CP
    time_mat = x_time.reshape((M + CP_LEN, N), order='F')

    # Strip Cyclic Prefix
    time_no_cp = time_mat[CP_LEN:, :]                              # Shape (M, N)

    # Normalized FFT along subcarriers
    return np.fft.fft(time_no_cp, axis=0, norm='ortho')             # Shape (M, N)