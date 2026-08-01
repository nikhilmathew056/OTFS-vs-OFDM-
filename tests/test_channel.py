import numpy as np
from config import SAMPLING_RATE
from channel import generate_channel, apply_channel

def test_channel_los():
    rng = np.random.default_rng(789)
    x = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    paths = generate_channel("los")
    
    # Pass snr_db=None for a true noiseless test
    y = apply_channel(x, paths, fs=SAMPLING_RATE, snr_db=None)
    assert np.allclose(x, y, atol=1e-9), "LOS channel application failed!"
    print("test_channel_los PASSED")

if __name__ == "__main__":
    test_channel_los()