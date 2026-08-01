"""
Executes all unit tests and figure generation scripts.
"""
import time
import sys
import os

# Ensure root directory is on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests.test_modulation import test_modulation_loopback
from tests.test_ofdm_loopback import test_ofdm_loopback
from tests.test_otfs_loopback import test_otfs_loopback
from tests.test_channel import test_channel_los

# Package imports from figures directory
from figures import (
    fig5_los,
    fig6_static_multipath,
    fig7_low_mobility,
    fig8_mid_mobility,
    fig9_extreme_mobility,
    fig10_mobility_sweep,
)


def run_unit_tests():
    print("=== Running Unit Tests ===")
    test_modulation_loopback()
    test_ofdm_loopback()
    test_otfs_loopback()
    test_channel_los()
    print("All unit tests passed successfully!\n")


def main():
    start_time = time.time()
    run_unit_tests()

    fig_modules = [
        ("Fig 5 (LoS)", fig5_los),
        ("Fig 6 (Static Multipath)", fig6_static_multipath),
        ("Fig 7 (Low Mobility)", fig7_low_mobility),
        ("Fig 8 (Mid Mobility)", fig8_mid_mobility),
        ("Fig 9 (Extreme Mobility)", fig9_extreme_mobility),
        ("Fig 10 (Mobility Sweep)", fig10_mobility_sweep),
    ]

    print("=== Generating Paper Figures ===")
    for label, mod in fig_modules:
        print(f"Generating {label}...")
        try:
            mod.main()
        except Exception as e:
            print(f"Error generating {label}: {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted all tasks in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()