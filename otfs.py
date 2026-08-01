"""
OTFS Modulator and Demodulator.
Implements the Symplectic Fourier Transform (SFFT / ISFFT) and 2D precoding (Eqs. 19-26).
"""
import numpy as np
from config import M, N
from ofdm import ofdm_modulate, ofdm_demodulate


def isfft(x_dd: np.ndarray) -> np.ndarray:
    """
    Inverse Symplectic Fast Fourier Transform (ISFFT): Delay-Doppler -> Time-Frequency.
    Eq. (22): Xt_f = FFT_delay( IFFT_doppler( x_dd.T ).T )
    """
    step1 = np.fft.ifft(x_dd.T, axis=1, norm='ortho').T  # IFFT along Doppler axis
    return np.fft.fft(step1, axis=0, norm='ortho')       # FFT along Delay axis


def sfft(y_tf: np.ndarray) -> np.ndarray:
    """
    Symplectic Fast Fourier Transform (SFFT): Time-Frequency -> Delay-Doppler.
    Eq. (26): y_dd = FFT_doppler( IFFT_delay( y_tf ).T ).T
    """
    step1 = np.fft.ifft(y_tf, axis=0, norm='ortho')      # IFFT along Frequency/Delay
    return np.fft.fft(step1.T, axis=1, norm='ortho').T   # FFT along Time/Doppler


def otfs_modulate(x_dd: np.ndarray) -> np.ndarray:
    """Transforms Delay-Doppler grid x_dd to 1D time-domain signal."""
    x_tf = isfft(x_dd)
    return ofdm_modulate(x_tf)


def otfs_demodulate(x_time: np.ndarray) -> np.ndarray:
    """Transforms 1D time-domain signal to Delay-Doppler grid y_dd."""
    y_tf = ofdm_demodulate(x_time)
    return sfft(y_tf)