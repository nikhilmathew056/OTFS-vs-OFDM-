import numpy as np
from config import M, N
from ofdm import ofdm_modulate, ofdm_demodulate

def test_ofdm_loopback():
    rng = np.random.default_rng(123)
    tx_symbols = rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))
    x_time = ofdm_modulate(tx_symbols)
    rx_symbols = ofdm_demodulate(x_time)
    assert np.allclose(tx_symbols, rx_symbols, atol=1e-9), "OFDM loopback failed!"
    print("test_ofdm_loopback PASSED")

if __name__ == "__main__":
    test_ofdm_loopback()