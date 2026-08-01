"""
Gray-coded symbol mapping and demapping for BPSK, QPSK, and 8-PSK.
All constellations are normalized to unit average symbol power.
"""
import numpy as np


def random_bits(n_bits: int, seed: int | None = None) -> np.ndarray:
    """Generates random 0/1 bits."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=np.int8)


# --- Constellation Tables ---
# BPSK
BPSK_MAP = {0: 1.0 + 0.0j, 1: -1.0 + 0.0j}

# QPSK (Gray-coded)
# 00 -> exp(j*pi/4), 01 -> exp(j*3pi/4), 11 -> exp(j*5pi/4), 10 -> exp(j*7pi/4)
QPSK_ANGLES = {
    (0, 0): np.pi / 4,
    (0, 1): 3 * np.pi / 4,
    (1, 1): 5 * np.pi / 4,
    (1, 0): 7 * np.pi / 4,
}
QPSK_MAP = {k: np.exp(1j * v) for k, v in QPSK_ANGLES.items()}

# 8PSK (Gray-coded)
# Gray order: 000, 001, 011, 010, 110, 111, 101, 100
_8PSK_GRAY_ORDER = [
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
    (1, 1, 0), (1, 1, 1), (1, 0, 1), (1, 0, 0)
]
_8PSK_MAP = {bit_tuple: np.exp(1j * 2 * np.pi * idx / 8) for idx, bit_tuple in enumerate(_8PSK_GRAY_ORDER)}


def bits_to_symbols(bits: np.ndarray, order: str) -> np.ndarray:
    """Maps bit array to complex symbols with unit average energy."""
    if order == "BPSK":
        return np.array([BPSK_MAP[b] for b in bits], dtype=np.complex128)
    elif order == "QPSK":
        bit_pairs = bits.reshape(-1, 2)
        return np.array([QPSK_MAP[tuple(p)] for p in bit_pairs], dtype=np.complex128)
    elif order == "8PSK":
        bit_triplets = bits.reshape(-1, 3)
        return np.array([_8PSK_MAP[tuple(t)] for t in bit_triplets], dtype=np.complex128)
    else:
        raise ValueError(f"Unsupported modulation order: {order}")


def symbols_to_bits(symbols: np.ndarray, order: str) -> np.ndarray:
    """Minimum Euclidean distance demapper."""
    symbols_flat = symbols.flatten()

    if order == "BPSK":
        const_pts = np.array([BPSK_MAP[0], BPSK_MAP[1]])
        bit_table = np.array([[0], [1]], dtype=np.int8)
    elif order == "QPSK":
        keys = list(QPSK_MAP.keys())
        const_pts = np.array([QPSK_MAP[k] for k in keys])
        bit_table = np.array(keys, dtype=np.int8)
    elif order == "8PSK":
        keys = list(_8PSK_MAP.keys())
        const_pts = np.array([_8PSK_MAP[k] for k in keys])
        bit_table = np.array(keys, dtype=np.int8)
    else:
        raise ValueError(f"Unsupported modulation order: {order}")

    # Compute Euclidean distance to each constellation point
    dists = np.abs(symbols_flat[:, np.newaxis] - const_pts[np.newaxis, :])
    min_indices = np.argmin(dists, axis=1)
    return bit_table[min_indices].flatten()