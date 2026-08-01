"""Figure 5: BER performance comparison in an ideal Line-of-Sight (LoS) channel."""
import os
import matplotlib.pyplot as plt
from config import SNR_DB_RANGE
from ber_sim import run_ber_point

def main():
    os.makedirs("figures/output", exist_ok=True)
    plt.figure(figsize=(8, 6))

    styles = {
        ("OFDM", "BPSK"): ("--*", "m"),
        ("OTFS", "BPSK"): ("-*", "m"),
        ("OFDM", "QPSK"): ("--o", "b"),
        ("OTFS", "QPSK"): ("-o", "b"),
        ("OFDM", "8PSK"): ("--v", "k"),
        ("OTFS", "8PSK"): ("-v", "k"),
    }

    for (scheme, mod), (fmt, color) in styles.items():
        ber_vals = [run_ber_point(scheme, mod, "los", snr) for snr in SNR_DB_RANGE]
        plt.semilogy(SNR_DB_RANGE, ber_vals, fmt, color=color, label=f"{scheme} {mod}")

    plt.title("Figure 5. BER performance comparison of OTFS and OFDM in a LoS channel")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.ylim(1e-3, 1.0)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/output/fig5_los.png", dpi=300)
    plt.close()
    print("Generated figures/output/fig5_los.png")

if __name__ == "__main__":
    main()