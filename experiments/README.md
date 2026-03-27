# LEM — Toy Experiments

Reproducible code for all numerical experiments reported in the LEM preprint.

## Structure

```
experiments/
├── lem_simulations.py    # All experiments in one file (original Colab code)
├── run_all.py            # Standalone runner (no Colab required)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Option A: Standalone (recommended)

```bash
python run_all.py               # run all 4 experiments
python run_all.py --skip-pilot  # skip the small pilot, run only paper experiments
```

### Option B: Google Colab

Upload `lem_simulations.py` to Colab and run each function in a separate cell:

```python
run_experiment_1()          # Pilot (DIM=4, N=20) — not in paper
run_experiment_1_scaled()   # Table 1, Figure 1 (DIM=512, N=500, 10 seeds)
run_experiment_1b()         # Table 2, Figure 2 (robustness / σ sweep)
run_experiment_2()          # Table 3, Figure 3 (topological class separation)
```

## Experiments

### Toy Experiment 1 (Pilot)
- **Not reported in the paper** — initial sanity check
- DIM=4, N=20, single seed
- Result: NN Accuracy ~0.97

### Toy Experiment 1 Scaled (Paper: Table 1, Figure 1)
- DIM=512, N=500, 10 independent seeds
- **Result: NN Accuracy 1.000 ± 0.000** (baseline: 0.002)
- Cosine (same user): 0.357, Cosine (diff user): 0.002

### Toy Experiment 1b — Robustness (Paper: Table 2, Figure 2)
- Varies σ ∈ {0.05, 0.15, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00}
- **Result: Phase transition at σ ≈ 0.25**
- Attractor distance grows linearly (slope ≈ 5.07)

### Toy Experiment 2 — Topological Separation (Paper: Table 3, Figure 3)
- Convergent (fixed-point sink) vs. Cyclic (Van der Pol oscillator)
- Embedded into R^16, persistent H1 homology via Vietoris-Rips
- **Result: SNR = 161.6** (max H1 cyclic: 5.265 vs. convergent: 0.033)

## Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| N | 500 | Number of simulated users |
| d | 512 | Embedding dimension |
| α | 0.4 | User influence weight |
| β | 0.3 | Domain/model influence weight |
| σ | 0.15 | Noise level (main experiment) |
| T | 200 | Trajectory length (steps) |
| K | 40 | Tail window for signature estimation |

## Outputs

Each experiment saves:
- Figures to `assets/` (PNG, matching paper figures exactly)
- Numerical results to the current directory (JSON / CSV)

All seeds are fixed for full reproducibility.
