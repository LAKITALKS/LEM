"""
================================================================================
LEM — Lazaros-Eudora Method
Simulation Code Repository
================================================================================

Author:  Lazaros Varvatis
Paper:   "The Lazaros-Eudora Method (LEM): A Topological Framework
          for User-LLM Interaction Dynamics"
Repo:    https://github.com/[your-username]/LEM

--------------------------------------------------------------------------------
CONTENTS
--------------------------------------------------------------------------------

  1. TOY EXPERIMENT 1        — Geometric Identifiability (small pilot)
  2. TOY EXPERIMENT 1 SCALED — Geometric Identifiability (N=500, DIM=512, 10 Seeds)
  3. TOY EXPERIMENT 1b       — Robustness Analysis (SIGMA sweep)
  4. TOY EXPERIMENT 2        — Topological Class Separation (TDA / Persistent Homology)

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

  pip install numpy matplotlib scikit-learn scipy ripser

--------------------------------------------------------------------------------
USAGE (Google Colab)
--------------------------------------------------------------------------------

  1. Upload this file to Colab or paste sections into cells
  2. Run each experiment in order (setup cell first)
  3. Outputs are saved to assets/ and exported as .png / .csv / .json

================================================================================
"""

import os
os.makedirs("assets", exist_ok=True)


# ==============================================================================
# EXPERIMENT 1 — GEOMETRIC IDENTIFIABILITY (Small Pilot, DIM=4, N=20)
# ==============================================================================
# Purpose:
#   Proof-of-concept. Tests whether user-specific attractor dynamics
#   generate re-identifiable trajectories in a minimal setting.
#   This is the initial sanity check before scaling.
#
# Key results (seed=42):
#   - NN Accuracy:          0.9700  (baseline: 0.0500)
#   - Attractor Distance:   0.2161
#   - Same-user cosine:     0.9782
#   - Diff-user cosine:     0.0307
# ==============================================================================

def run_experiment_1():
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    RNG_SEED = 42
    np.random.seed(RNG_SEED)

    DIM      = 4
    N_USERS  = 20
    N_MODELS = 2
    T_STEPS  = 60
    ALPHA    = 0.4
    BETA     = 0.3
    SIGMA    = 0.15
    LAST_K   = 20
    N_TESTS  = 200

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(n, 1e-8, None)

    def mean_last_k(traj, k=LAST_K):
        return traj[-k:].mean(axis=0)

    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def simulate_trajectory(user_sig, domain_attr):
        x = np.random.normal(size=DIM)
        traj = [x.copy()]
        for _ in range(T_STEPS - 1):
            noise = np.random.normal(scale=SIGMA, size=DIM)
            x = ((1 - ALPHA - BETA) * x
                 + ALPHA * user_sig
                 + BETA * domain_attr
                 + noise)
            traj.append(x.copy())
        return np.vstack(traj)

    def estimate_empirical_signature(traj, domain_attr):
        return mean_last_k(traj, LAST_K) - BETA * domain_attr

    def nearest_neighbor_accuracy(user_signatures, domain_attractors,
                                   empirical_signatures, model_idx):
        correct = 0
        n_users = user_signatures.shape[0]
        for _ in range(N_TESTS):
            i = np.random.randint(n_users)
            traj = simulate_trajectory(user_signatures[i], domain_attractors[model_idx])
            test_sig = estimate_empirical_signature(traj, domain_attractors[model_idx])
            dists = np.linalg.norm(empirical_signatures[model_idx] - test_sig, axis=1)
            if int(np.argmin(dists)) == i:
                correct += 1
        return correct / N_TESTS

    # Setup
    user_signatures   = normalize(np.random.normal(size=(N_USERS, DIM)))
    domain_attractors = normalize(np.random.normal(size=(N_MODELS, DIM)))
    domain_attractors[1] = domain_attractors[0] + 0.4 * np.random.normal(size=DIM)
    domain_attractors = normalize(domain_attractors)

    # Simulate
    trajectories = {}
    for i in range(N_USERS):
        for m in range(N_MODELS):
            trajectories[(i, m)] = simulate_trajectory(user_signatures[i],
                                                        domain_attractors[m])

    # Attractor distance
    distances = []
    for (i, m), traj in trajectories.items():
        mean_vec = mean_last_k(traj, LAST_K)
        target   = ALPHA * user_signatures[i] + BETA * domain_attractors[m]
        distances.append(np.linalg.norm(mean_vec - target))

    # Empirical signatures
    empirical_signatures = {m: [] for m in range(N_MODELS)}
    for i in range(N_USERS):
        for m in range(N_MODELS):
            sig = estimate_empirical_signature(trajectories[(i, m)], domain_attractors[m])
            empirical_signatures[m].append(sig)
    empirical_signatures = {m: np.vstack(v) for m, v in empirical_signatures.items()}

    # NN accuracy
    accs = [nearest_neighbor_accuracy(user_signatures, domain_attractors,
                                       empirical_signatures, m)
            for m in range(N_MODELS)]

    # Cosine similarities
    same_sims, diff_sims = [], []
    for i in range(N_USERS):
        sig0 = empirical_signatures[0][i]
        sig1 = empirical_signatures[1][i]
        same_sims.append(cosine_similarity(sig0, sig1))
        j = (i + np.random.randint(1, N_USERS)) % N_USERS
        diff_sims.append(cosine_similarity(sig0, empirical_signatures[1][j]))

    # PCA plot
    all_points = np.vstack([t for t in trajectories.values()])
    pca = PCA(n_components=2).fit(all_points)

    plt.figure(figsize=(6, 5))
    for u, c in zip([0, 1, 2], ["tab:blue", "tab:orange", "tab:green"]):
        proj = pca.transform(trajectories[(u, 0)])
        plt.plot(proj[:, 0], proj[:, 1], "-", alpha=0.7, color=c, label=f"user {u}")
        plt.scatter(proj[-1, 0], proj[-1, 1], color=c, s=30)
    plt.title("Toy Experiment 1: Geometric CVI (Model 0)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout()
    plt.savefig("assets/toy_v1_geometric_cvi.png", dpi=150)
    plt.close()

    metrics = {
        "random_baseline_accuracy":      round(1.0 / N_USERS, 4),
        "nn_reidentification_accuracy":  round(float(np.mean(accs)), 4),
        "mean_attractor_distance":       round(float(np.mean(distances)), 4),
        "same_user_cosine_similarity":   round(float(np.mean(same_sims)), 4),
        "different_user_cosine_similarity": round(float(np.mean(diff_sims)), 4),
    }
    with open("toy_v1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Toy Experiment 1 Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# ==============================================================================
# EXPERIMENT 1 SCALED — GEOMETRIC IDENTIFIABILITY (N=500, DIM=512, 10 Seeds)
# ==============================================================================
# Purpose:
#   Scaled version of Experiment 1. Tests statistical robustness
#   across 10 random seeds with N=500 users and DIM=512.
#   This is the version reported in the paper (Table 1).
#
# Key results (mean over 10 seeds):
#   - NN Accuracy:         1.0000 +/- 0.0000  (baseline: 0.0020)
#   - Attractor Distance:  0.7893 +/- 0.0009
#   - Same-user cosine:    0.3570 +/- 0.0020
#   - Diff-user cosine:    0.0022 +/- 0.0019
#
# Runtime: ~2 minutes on Colab CPU
# ==============================================================================

def run_experiment_1_scaled():
    import json
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import Image, display

    DIM      = 512
    N_USERS  = 500
    N_MODELS = 2
    T_STEPS  = 200
    ALPHA    = 0.4
    BETA     = 0.3
    SIGMA    = 0.15
    LAST_K   = 40
    N_TESTS  = 500
    SEEDS    = [42, 123, 999, 2024, 7, 314, 1337, 77, 256, 88]

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(n, 1e-8, None)

    def cosine_matrix(A, B):
        A_n = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)
        B_n = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-8, None)
        return np.sum(A_n * B_n, axis=1)

    def simulate_all_vectorized(user_sigs, domain_attr, last_k):
        N, D = user_sigs.shape
        x = np.random.normal(size=(N, D))
        buffer = []
        for t in range(T_STEPS):
            noise = np.random.normal(scale=SIGMA, size=(N, D))
            x = ((1 - ALPHA - BETA) * x
                 + ALPHA * user_sigs
                 + BETA * domain_attr[None, :]
                 + noise)
            if t >= T_STEPS - last_k:
                buffer.append(x.copy())
        return np.stack(buffer, axis=1)

    def simulate_single(user_sig, domain_attr):
        D = user_sig.shape[0]
        x = np.random.normal(size=D)
        traj = [x.copy()]
        for _ in range(T_STEPS - 1):
            noise = np.random.normal(scale=SIGMA, size=D)
            x = ((1 - ALPHA - BETA) * x
                 + ALPHA * user_sig
                 + BETA * domain_attr
                 + noise)
            traj.append(x.copy())
        return np.vstack(traj)

    def nn_accuracy(user_sigs, domain_attr, train_sigs):
        N = user_sigs.shape[0]
        correct = 0
        for _ in range(N_TESTS):
            i = np.random.randint(N)
            traj     = simulate_single(user_sigs[i], domain_attr)
            test_sig = traj[-LAST_K:].mean(axis=0) - BETA * domain_attr
            dists    = np.linalg.norm(train_sigs - test_sig, axis=1)
            if int(np.argmin(dists)) == i:
                correct += 1
        return correct / N_TESTS

    all_results = []
    print(f"LEM Toy Experiment 1 — Scaled")
    print(f"DIM={DIM} | N_USERS={N_USERS} | T_STEPS={T_STEPS} | Seeds={len(SEEDS)}\n")

    for seed_idx, seed in enumerate(SEEDS):
        t0 = time.time()
        np.random.seed(seed)
        print(f"Seed {seed} ({seed_idx+1}/{len(SEEDS)})")

        user_signatures  = normalize(np.random.normal(size=(N_USERS, DIM)))
        domain_attractors = normalize(np.random.normal(size=(N_MODELS, DIM)))
        domain_attractors[1] = normalize(
            (domain_attractors[0] + 0.4 * np.random.normal(size=DIM))[None, :])[0]

        emp_sigs    = {}
        attr_dists  = []
        for m in range(N_MODELS):
            buf       = simulate_all_vectorized(user_signatures, domain_attractors[m], LAST_K)
            mean_vecs = buf.mean(axis=1)
            emp_sigs[m] = mean_vecs - BETA * domain_attractors[m][None, :]
            targets   = ALPHA * user_signatures + BETA * domain_attractors[m][None, :]
            attr_dists.append(float(np.linalg.norm(mean_vecs - targets, axis=1).mean()))

        accs = [nn_accuracy(user_signatures, domain_attractors[m], emp_sigs[m])
                for m in range(N_MODELS)]

        same = float(cosine_matrix(emp_sigs[0], emp_sigs[1]).mean())
        diff = float(cosine_matrix(emp_sigs[0],
                                    emp_sigs[1][np.roll(np.arange(N_USERS), 1)]).mean())

        all_results.append({
            "seed":                            seed,
            "random_baseline_accuracy":        round(1.0 / N_USERS, 6),
            "nn_reidentification_accuracy":    round(float(np.mean(accs)), 6),
            "mean_attractor_distance":         round(float(np.mean(attr_dists)), 6),
            "same_user_cosine_similarity":     round(same, 6),
            "different_user_cosine_similarity": round(diff, 6),
        })
        print(f"  NN={all_results[-1]['nn_reidentification_accuracy']:.4f} "
              f"| AttrDist={all_results[-1]['mean_attractor_distance']:.4f} "
              f"| Same={same:.4f} | Diff={diff:.4f} "
              f"| {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_results)
    cols = ["nn_reidentification_accuracy", "mean_attractor_distance",
            "same_user_cosine_similarity",  "different_user_cosine_similarity"]
    summary = df[cols].agg(["mean", "std", "min", "max"]).T
    print("\n=== Summary ===")
    print(summary.to_string(float_format="{:.4f}".format))

    # Plot
    SEEDS_LIST = [r["seed"] for r in all_results]
    x = np.arange(len(SEEDS_LIST))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, df["nn_reidentification_accuracy"], color="steelblue", alpha=0.8)
    axes[0].axhline(1/N_USERS, color="red",    linestyle="--", label=f"Baseline ({1/N_USERS:.3f})")
    axes[0].axhline(df["nn_reidentification_accuracy"].mean(),
                    color="orange", linewidth=2, label="Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s) for s in SEEDS_LIST], rotation=45, fontsize=8)
    axes[0].set_title(f"NN Re-Identification Accuracy\n(DIM={DIM}, N={N_USERS})")
    axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.15); axes[0].grid(alpha=0.3)

    w = 0.35
    axes[1].bar(x - w/2, df["same_user_cosine_similarity"],
                width=w, color="tab:green", alpha=0.8, label="Same User")
    axes[1].bar(x + w/2, df["different_user_cosine_similarity"],
                width=w, color="tab:red",   alpha=0.8, label="Diff User")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in SEEDS_LIST], rotation=45, fontsize=8)
    axes[1].set_title(f"Signature Separability\n(DIM={DIM}, N={N_USERS})")
    axes[1].set_ylabel("Cosine Similarity"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle("LEM Toy V1 — Scaled Results (10 Seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig("assets/toy_v1_scaled_results.png", dpi=150)
    plt.close()

    df.to_csv("toy_v1_scaled_results.csv", index=False)
    summary.to_csv("toy_v1_scaled_summary.csv")
    with open("toy_v1_scaled_results.json", "w") as f:
        json.dump({"config": {"DIM": DIM, "N_USERS": N_USERS, "T_STEPS": T_STEPS,
                               "ALPHA": ALPHA, "BETA": BETA, "SIGMA": SIGMA,
                               "N_TESTS": N_TESTS, "SEEDS": SEEDS},
                   "results": all_results}, f, indent=2)

    display(Image("assets/toy_v1_scaled_results.png"))
    return df, summary


# ==============================================================================
# EXPERIMENT 1b — ROBUSTNESS ANALYSIS (SIGMA sweep, DIM=512, N=500)
# ==============================================================================
# Purpose:
#   Tests sensitivity of geometric identifiability to noise level sigma.
#   Reveals three regimes: stable, critical transition, noise-dominated.
#   This is the robustness analysis reported in the paper (Table 2).
#
# Key results (seed=42):
#   sigma=0.05: NN=1.000  |  sigma=0.30: NN=0.230  |  sigma=0.50: NN=0.020
#   Phase transition at sigma ~0.25
#   Attractor distance grows linearly: slope ~5.07
#
# Runtime: ~10-15 minutes on Colab CPU
# ==============================================================================

def run_experiment_1b():
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import Image, display

    DIM          = 512
    N_USERS      = 500
    N_MODELS     = 2
    T_STEPS      = 200
    ALPHA        = 0.4
    BETA         = 0.3
    LAST_K       = 40
    N_TESTS      = 300
    SEED         = 42
    SIGMA_VALUES = [0.05, 0.15, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(n, 1e-8, None)

    def simulate_all_vectorized(user_sigs, domain_attr, sigma):
        N, D = user_sigs.shape
        x = np.random.normal(size=(N, D))
        buffer = []
        for t in range(T_STEPS):
            noise = np.random.normal(scale=sigma, size=(N, D))
            x = ((1 - ALPHA - BETA) * x
                 + ALPHA * user_sigs
                 + BETA * domain_attr[None, :]
                 + noise)
            if t >= T_STEPS - LAST_K:
                buffer.append(x.copy())
        return np.stack(buffer, axis=1)

    def simulate_single(user_sig, domain_attr, sigma):
        D = user_sig.shape[0]
        x = np.random.normal(size=D)
        traj = [x.copy()]
        for _ in range(T_STEPS - 1):
            noise = np.random.normal(scale=sigma, size=D)
            x = ((1 - ALPHA - BETA) * x
                 + ALPHA * user_sig
                 + BETA * domain_attr
                 + noise)
            traj.append(x.copy())
        return np.vstack(traj)

    def nn_accuracy(user_sigs, domain_attr, train_sigs, sigma):
        N = user_sigs.shape[0]
        correct = 0
        for _ in range(N_TESTS):
            i = np.random.randint(N)
            traj     = simulate_single(user_sigs[i], domain_attr, sigma)
            test_sig = traj[-LAST_K:].mean(axis=0) - BETA * domain_attr
            dists    = np.linalg.norm(train_sigs - test_sig, axis=1)
            if int(np.argmin(dists)) == i:
                correct += 1
        return correct / N_TESTS

    def cosine_matrix(A, B):
        A_n = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)
        B_n = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-8, None)
        return np.sum(A_n * B_n, axis=1)

    # Fixed setup
    np.random.seed(SEED)
    user_signatures  = normalize(np.random.normal(size=(N_USERS, DIM)))
    domain_attractors = normalize(np.random.normal(size=(N_MODELS, DIM)))
    domain_attractors[1] = normalize(
        (domain_attractors[0] + 0.4 * np.random.normal(size=DIM))[None, :])[0]

    results = []
    print(f"LEM Toy Experiment 1b — Robustness")
    print(f"DIM={DIM} | N={N_USERS} | Seed={SEED}\n")
    print(f"{'SIGMA':<8} {'NN-Acc':<10} {'AttrDist':<12} {'CosSame':<10} {'CosDiff':<10}")
    print("-" * 55)

    for sigma in SIGMA_VALUES:
        np.random.seed(SEED)
        emp_sigs   = {}
        attr_dists = []
        for m in range(N_MODELS):
            buf       = simulate_all_vectorized(user_signatures, domain_attractors[m], sigma)
            mean_vecs = buf.mean(axis=1)
            emp_sigs[m] = mean_vecs - BETA * domain_attractors[m][None, :]
            targets   = ALPHA * user_signatures + BETA * domain_attractors[m][None, :]
            attr_dists.append(float(np.linalg.norm(mean_vecs - targets, axis=1).mean()))

        accs = [nn_accuracy(user_signatures, domain_attractors[m], emp_sigs[m], sigma)
                for m in range(N_MODELS)]

        same = float(cosine_matrix(emp_sigs[0], emp_sigs[1]).mean())
        diff = float(cosine_matrix(emp_sigs[0],
                                    emp_sigs[1][np.roll(np.arange(N_USERS), 1)]).mean())
        acc  = float(np.mean(accs))
        dist = float(np.mean(attr_dists))

        results.append({"sigma": sigma, "nn_accuracy": round(acc, 4),
                         "attractor_distance": round(dist, 4),
                         "cosine_same": round(same, 4), "cosine_diff": round(diff, 4)})
        print(f"{sigma:<8.2f} {acc:<10.4f} {dist:<12.4f} {same:<10.4f} {diff:<10.4f}")

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(df["sigma"], df["nn_accuracy"], "o-", color="steelblue", linewidth=2)
    axes[0].axhline(1/N_USERS, color="red", linestyle="--", label=f"Baseline ({1/N_USERS:.3f})")
    axes[0].axvline(0.15, color="gray", linestyle=":", alpha=0.7, label="sigma=0.15")
    axes[0].set_xlabel("Noise Level sigma"); axes[0].set_ylabel("NN Accuracy")
    axes[0].set_title("Robustness: NN-Accuracy vs Noise")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)

    axes[1].plot(df["sigma"], df["attractor_distance"], "o-", color="darkorange", linewidth=2)
    axes[1].axvline(0.15, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Noise Level sigma"); axes[1].set_ylabel("Attractor Distance")
    axes[1].set_title("Convergence vs Noise"); axes[1].grid(alpha=0.3)

    axes[2].plot(df["sigma"], df["cosine_same"], "o-", color="tab:green",
                 linewidth=2, label="Same User")
    axes[2].plot(df["sigma"], df["cosine_diff"], "s--", color="tab:red",
                 linewidth=2, label="Diff User")
    axes[2].axvline(0.15, color="gray", linestyle=":", alpha=0.7, label="sigma=0.15")
    axes[2].set_xlabel("Noise Level sigma"); axes[2].set_ylabel("Cosine Similarity")
    axes[2].set_title("Signature Separability vs Noise")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.suptitle(f"LEM Toy V1b — Robustness (DIM={DIM}, N={N_USERS})", fontsize=13)
    plt.tight_layout()
    plt.savefig("assets/toy_v1b_robustness.png", dpi=150)
    plt.close()

    df.to_csv("toy_v1b_robustness.csv", index=False)
    with open("toy_v1b_robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    display(Image("assets/toy_v1b_robustness.png"))
    return df


# ==============================================================================
# EXPERIMENT 2 — TOPOLOGICAL CLASS SEPARATION (TDA / Persistent Homology)
# ==============================================================================
# Purpose:
#   Core topological validation of LEM. Distinguishes convergent vs cyclic
#   interaction regimes using persistent H1 homology (Vietoris-Rips).
#   This is the central result of the paper (Table 3, Figure 2).
#
# Key results (seed=1337):
#   - Max H1 convergent:  0.0326  (noise-level, no persistent loop)
#   - Max H1 cyclic:      5.2649  (robust persistent loop = limit cycle)
#   - SNR:                161.6x
#
# Requires:  pip install ripser
# Runtime:   ~1-2 minutes on Colab CPU
# ==============================================================================

def run_experiment_2():
    import json
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    from ripser import ripser
    from sklearn.decomposition import PCA
    from IPython.display import Image, display

    warnings.filterwarnings("ignore")

    RNG_SEED    = 1337
    np.random.seed(RNG_SEED)

    T_MAX       = 40.0
    DT          = 0.1
    STEPS       = int(T_MAX / DT) + 1
    t           = np.linspace(0.0, T_MAX, STEPS)
    MU          = 1.5
    LAMBDA      = 0.5
    DIM_LATENT  = 16
    NOISE_LEVEL = 0.05

    # Convergent regime: stable fixed-point sink
    def system_fixed_point(state, t, lam):
        return [-lam * state[0], -lam * state[1]]

    # Cyclic regime: Van der Pol limit cycle
    def system_van_der_pol(state, t, mu):
        return [state[1], mu * (1 - state[0]**2) * state[1] - state[0]]

    def embed_high_dim(traj_2d):
        A = np.random.randn(2, DIM_LATENT)
        return (traj_2d @ A
                + np.random.normal(scale=NOISE_LEVEL,
                                   size=(traj_2d.shape[0], DIM_LATENT)))

    def compute_persistence(data_hd):
        return ripser(data_hd, maxdim=1)["dgms"]

    def persistence_stats(dgm):
        if dgm is None or len(dgm) == 0:
            return 0.0, 0.0
        lt = dgm[:, 1] - dgm[:, 0]
        lt = lt[np.isfinite(lt)]
        if len(lt) == 0:
            return 0.0, 0.0
        return float(np.max(lt)), float(np.mean(lt))

    def plot_pd(ax, dgm, color, title):
        if dgm is None or len(dgm) == 0:
            ax.text(0.5, 0.5, "no H1 features", ha="center", va="center")
            ax.set_axis_off()
            return
        lim = max(1.0, float(np.max(dgm)) + 0.1)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.scatter(dgm[:, 0], dgm[:, 1], c=color, s=30)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.set_title(title)

    print("1/4 Simulating trajectories...")
    traj_A_2d = odeint(system_fixed_point, [2.0, 1.5], t, args=(LAMBDA,))
    traj_B_2d = odeint(system_van_der_pol,  [0.1, 0.1], t, args=(MU,))

    print("2/4 Embedding into 16D latent space...")
    traj_A_hd = embed_high_dim(traj_A_2d)
    traj_B_hd = embed_high_dim(traj_B_2d)

    print("3/4 Computing Persistent Homology (Vietoris-Rips)...")
    dgms_A = compute_persistence(traj_A_hd)
    dgms_B = compute_persistence(traj_B_hd)

    maxA, meanA = persistence_stats(dgms_A[1])
    maxB, meanB = persistence_stats(dgms_B[1])
    snr = maxB / (maxA + 1e-9)

    print(f"  H1 Convergent: max={maxA:.4f}, mean={meanA:.4f}")
    print(f"  H1 Cyclic:     max={maxB:.4f}, mean={meanB:.4f}")
    print(f"  SNR:           {snr:.2f}x")

    print("4/4 Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    pA = PCA(n_components=2).fit_transform(traj_A_hd)
    pB = PCA(n_components=2).fit_transform(traj_B_hd)

    axes[0, 0].plot(pA[:, 0], pA[:, 1], color="tab:blue", alpha=0.6)
    axes[0, 0].scatter(pA[-1, 0], pA[-1, 1], color="tab:blue", marker="x", s=60)
    axes[0, 0].set_title("Convergent Regime (PCA)")
    axes[0, 0].set_xlabel("PC1"); axes[0, 0].set_ylabel("PC2")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(pB[:, 0], pB[:, 1], color="tab:orange", alpha=0.6)
    axes[0, 1].scatter(pB[-1, 0], pB[-1, 1], color="tab:orange", marker="x", s=60)
    axes[0, 1].set_title("Cyclic Regime (PCA)")
    axes[0, 1].set_xlabel("PC1"); axes[0, 1].set_ylabel("PC2")
    axes[0, 1].grid(alpha=0.3)

    plot_pd(axes[1, 0], dgms_A[1], "tab:blue",
            "PD H1 - Convergent  max=" + str(round(maxA, 3)))
    plot_pd(axes[1, 1], dgms_B[1], "tab:orange",
            "PD H1 - Cyclic  max=" + str(round(maxB, 3)))

    plt.suptitle("LEM Toy V2 - Topological Class Separation", fontsize=13)
    plt.tight_layout()
    plt.savefig("assets/toy_v2_moneyplot.png", dpi=150)
    plt.close()

    metrics = {
        "max_H1_convergent":  round(maxA, 4),
        "max_H1_cyclic":      round(maxB, 4),
        "mean_H1_convergent": round(meanA, 4),
        "mean_H1_cyclic":     round(meanB, 4),
        "snr":                round(snr, 4),
    }
    with open("toy_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Toy Experiment 2 Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    display(Image("assets/toy_v2_moneyplot.png"))
    return metrics


# ==============================================================================
# MAIN — Run all experiments sequentially
# ==============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LEM SIMULATION SUITE")
    print("=" * 70)

    print("\n--- EXPERIMENT 1: Geometric Identifiability (Pilot) ---")
    run_experiment_1()

    print("\n--- EXPERIMENT 1 SCALED: N=500, DIM=512, 10 Seeds ---")
    run_experiment_1_scaled()

    print("\n--- EXPERIMENT 1b: Robustness (SIGMA sweep) ---")
    run_experiment_1b()

    print("\n--- EXPERIMENT 2: Topological Class Separation ---")
    run_experiment_2()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("Outputs saved to: assets/")
    print("=" * 70)
