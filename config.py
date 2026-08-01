"""
Global simulation parameters matching Table 1 of Okoyeigbo et al. (2025).
"""
from dataclasses import dataclass
import numpy as np

# System Grid Parameters
M: int = 128                  # Number of subcarriers (delay bins)
N: int = 64                   # Number of sub-symbols per frame (Doppler bins)
DELTA_F: float = 480e3        # Subcarrier spacing: 480 kHz
FC: float = 100e9             # Carrier frequency: 100 GHz (THz communications)
CP_LEN: int = 10              # Cyclic prefix / padding length (samples)
C_LIGHT: float = 3e8          # Speed of light (m/s)

# Timing Parameters
T_SUB: float = 1.0 / DELTA_F                       # Sub-symbol duration (no CP): ~2.0833 us
SAMPLING_RATE: float = M * DELTA_F                 # Sample rate fs = 61.44 MHz
TS: float = 1.0 / SAMPLING_RATE                    # Sample period: ~16.276 ns
T_CP: float = CP_LEN * TS                          # CP duration: ~162.76 ns
T_TOTAL: float = T_SUB + T_CP                      # Total symbol duration incl. CP: ~2.246 us
FRAME_T: float = N * T_SUB                         # Active frame duration for Doppler grid

CHAN_EST: str = "LMMSE"                            # Channel estimation / equalization method
MOD_ORDERS: dict[str, int] = {"BPSK": 1, "QPSK": 2, "8PSK": 3}
SNR_DB_RANGE: list[int] = list(range(-10, 21, 5))  # -10, -5, 0, 5, 10, 15, 20 dB


def f_D_max(v_kmh: float) -> float:
    """Computes maximum Doppler shift (Hz) for a given speed in km/h."""
    v_ms = (v_kmh * 1000.0) / 3600.0
    return (v_ms / C_LIGHT) * FC


MOBILITY_SCENARIOS: dict[str, dict] = {
    "los": {"v_kmh": 0.0, "f_d": 0.0, "desc": "Line-of-Sight Static"},
    "static_multipath": {"v_kmh": 0.0, "f_d": 0.0, "desc": "Static Multipath"},
    "low_20": {"v_kmh": 20.0, "f_d": f_D_max(20.0), "desc": "Low Mobility (20 km/h - Cars/V2X)"},
    "low_120": {"v_kmh": 120.0, "f_d": f_D_max(120.0), "desc": "Moderate Mobility (120 km/h - Trains)"},
    "mid_300": {"v_kmh": 300.0, "f_d": f_D_max(300.0), "desc": "High Mobility (300 km/h - HSR)"},
    "mid_500": {"v_kmh": 500.0, "f_d": f_D_max(500.0), "desc": "High Mobility (500 km/h - UAVs)"},
    "extreme_1000": {"v_kmh": 1000.0, "f_d": f_D_max(1000.0), "desc": "Extreme Mobility (1000 km/h - Jets)"},
    "extreme_2000": {"v_kmh": 2000.0, "f_d": f_D_max(2000.0), "desc": "Extreme Mobility (2000 km/h - Supersonic)"},
}

if __name__ == "__main__":
    print(f"{'Scenario':<15} | {'v (km/h)':<8} | {'f_D_max (kHz)':<12} | {'Max Doppler Bin (l_max)':<22}")
    print("-" * 65)
    for name, sc in MOBILITY_SCENARIOS.items():
        fd_khz = sc["f_d"] / 1e3
        l_max = round(sc["f_d"] * FRAME_T)
        print(f"{name:<15} | {sc['v_kmh']:<8.0f} | {fd_khz:<12.2f} | {l_max:<22}")