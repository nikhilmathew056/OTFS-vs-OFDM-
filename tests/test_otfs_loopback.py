import numpy as np
from config import M, N
from otfs import otfs_modulate, otfs_demodulate, isfft, sfft

def test_otfs_loopback():
    rng = np.random.default_rng(456)
    x_dd = rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))

    # Test 2D Transform orthogonality
    x_tf = isfft(x_dd)
    rec_dd = sfft(x_tf)
    assert np.allclose(x_dd, rec_dd, atol=1e-9), "ISFFT/SFFT loopback failed!"

    # Test full OTFS modem loopback
    x_time = otfs_modulate(x_dd)
    rx_dd = otfs_demodulate(x_time)
    assert np.allclose(x_dd, rx_dd, atol=1e-9), "OTFS modem loopback failed!"
    print("test_otfs_loopback PASSED")

if __name__ == "__main__":
    test_otfs_loopback()