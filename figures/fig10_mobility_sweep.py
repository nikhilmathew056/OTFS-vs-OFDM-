"""Figure 10: Comparative BER performance across speeds (20, 500, and 2000 km/h)."""
import os
import matplotlib.pyplot as plt
from config import SNR_DB_RANGE
from ber_sim import run_ber_point

def main():
    os.makedirs("figures/output", exist_ok=True)
    plt.figure(figsize=(8, 6))

    configs = [
        ("OFDM", "extreme_2000", "--*", "m", "OFDM 2000 km/h"),
        ("OTFS", "extreme_2000", "-*", "m", "OTFS 2000 km/h"),
        ("OFDM", "mid_500", "--o", "b", "OFDM 400 km/h"),
        ("OTFS", "mid_500", "-o", "b", "OTFS 400 km/h"),
        ("OFDM", "low_20", "--v", "k", "OFDM 20 km/h"),
        ("OTFS", "low_20", "-v", "k", "OTFS 20 km/h"),
    ]

    for scheme, scenario, fmt, color, label in configs:
        ber_vals = [run_ber_point(scheme, "QPSK", scenario, snr) for snr in SNR_DB_RANGE]
        plt.semilogy(SNR_DB_RANGE, ber_vals, fmt, color=color, label=label)

    plt.title("Figure 10. BER performance comparison under different mobility scenarios")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.ylim(1e-3, 1.0)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/output/fig10_mobility_sweep.png", dpi=300)
    plt.close()
    print("Generated figures/output/fig10_mobility_sweep.png")

if __name__ == "__main__":
    main()