"""
Monte-Carlo Bit Error Rate (BER) simulation engine.
Executes end-to-end frame transmissions for OFDM and OTFS across mobility scenarios.
"""
import numpy as np
from config import M, N, MOD_ORDERS, SAMPLING_RATE
from modulation import random_bits, bits_to_symbols, symbols_to_bits
from channel import generate_channel, apply_channel
from ofdm import ofdm_modulate, ofdm_demodulate
from otfs import otfs_modulate, otfs_demodulate, sfft
from equalizer import compute_H_tf_from_paths, get_lmmse_channel_estimate, mmse_equalize_tf


def run_ber_point(scheme: str, mod_order: str, scenario_name: str, snr_db: float,
                  max_frames: int = 300, min_bit_errors: int = 150) -> float:
    """Runs Monte-Carlo BER evaluation for a given system configuration point."""
    bits_per_sym = MOD_ORDERS[mod_order]
    total_bits_per_frame = M * N * bits_per_sym

    total_bit_errors = 0
    total_bits = 0
    snr_linear = 10.0 ** (snr_db / 10.0)

    for frame_idx in range(max_frames):
        # 1. Data generation
        tx_bits = random_bits(total_bits_per_frame, seed=frame_idx + 1000)
        tx_symbols = bits_to_symbols(tx_bits, mod_order).reshape((M, N))

        # 2. Channel generation
        paths = generate_channel(scenario_name, seed=frame_idx + 5000)

        # 3. Modulation & Transmission
        if scheme == "OFDM":
            x_time = ofdm_modulate(tx_symbols)
        else:  # OTFS
            x_time = otfs_modulate(tx_symbols)

        # 4. Channel Application
        y_time = apply_channel(x_time, paths, fs=SAMPLING_RATE, snr_db=snr_db, seed=frame_idx + 9000)

        # 5. LMMSE Channel Estimation & Equalization
        H_tf_true = compute_H_tf_from_paths(paths)
        H_est = get_lmmse_channel_estimate(H_tf_true, snr_linear, scheme, seed=frame_idx + 12000)

        if scheme == "OFDM":
            Y_tf = ofdm_demodulate(y_time)
            X_hat_symbols = mmse_equalize_tf(Y_tf, H_est, snr_linear)
        else:  # OTFS
            Y_tf = ofdm_demodulate(y_time)
            X_hat_tf = mmse_equalize_tf(Y_tf, H_est, snr_linear)
            X_hat_symbols = sfft(X_hat_tf)

        # 6. Demapping & Bit Error Counting
        rx_bits = symbols_to_bits(X_hat_symbols, mod_order)
        bit_errors = np.sum(tx_bits != rx_bits)

        total_bit_errors += bit_errors
        total_bits += total_bits_per_frame

        # Early stopping criterion
        if total_bit_errors >= min_bit_errors and frame_idx >= 30:
            break

    return float(total_bit_errors) / float(total_bits)