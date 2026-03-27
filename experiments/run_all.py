"""
LEM — Standalone Runner
========================
Runs the original lem_simulations.py experiments without
requiring Google Colab (patches IPython.display imports).

Usage:
    python run_all.py              # run all experiments
    python run_all.py --skip-pilot # skip the small pilot (Experiment 1)

Requirements:
    pip install -r requirements.txt
"""

import sys
import os
import types

# ─── Patch IPython.display so the original code runs outside Colab ───
import matplotlib
matplotlib.use('Agg')  # Must be set before any pyplot import

mock_display = types.ModuleType('IPython')
mock_display.display = types.ModuleType('IPython.display')
mock_display.display.Image = lambda *a, **k: None
mock_display.display.display = lambda *a, **k: None
mock_display.get_ipython = lambda: None
mock_display.version_info = (99, 0, 0)  # Fake high version to skip backend fix
sys.modules['IPython'] = mock_display
sys.modules['IPython.display'] = mock_display.display

os.makedirs("assets", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─── Import the original simulation functions ───
from lem_simulations import (
    run_experiment_1,
    run_experiment_1_scaled,
    run_experiment_1b,
    run_experiment_2,
)

if __name__ == "__main__":
    skip_pilot = "--skip-pilot" in sys.argv

    print("=" * 70)
    print("LEM SIMULATION SUITE")
    print("=" * 70)

    if not skip_pilot:
        print("\n--- EXPERIMENT 1: Geometric Identifiability (Pilot) ---")
        print("(Small scale: DIM=4, N=20 — not reported in paper)")
        run_experiment_1()
    else:
        print("\n--- Skipping Experiment 1 (Pilot) ---")

    print("\n--- EXPERIMENT 1 SCALED: N=500, DIM=512, 10 Seeds ---")
    print("(This produces Table 1 and Figure 1 of the paper)")
    run_experiment_1_scaled()

    print("\n--- EXPERIMENT 1b: Robustness (SIGMA sweep) ---")
    print("(This produces Table 2 and Figure 2 of the paper)")
    run_experiment_1b()

    print("\n--- EXPERIMENT 2: Topological Class Separation ---")
    print("(This produces Table 3 and Figure 3 of the paper)")
    run_experiment_2()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("Outputs saved to: assets/ and current directory")
    print("=" * 70)
