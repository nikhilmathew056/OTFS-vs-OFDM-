import numpy as np
from modulation import random_bits, bits_to_symbols, symbols_to_bits

def test_modulation_loopback():
    for order, bits_per_sym in [("BPSK", 1), ("QPSK", 2), ("8PSK", 3)]:
        n_bits = 3000 * bits_per_sym
        bits = random_bits(n_bits, seed=42)
        symbols = bits_to_symbols(bits, order)
        rec_bits = symbols_to_bits(symbols, order)
        assert np.array_equal(bits, rec_bits), f"{order} modulation loopback failed!"
    print("test_modulation_loopback PASSED")

if __name__ == "__main__":
    test_modulation_loopback()