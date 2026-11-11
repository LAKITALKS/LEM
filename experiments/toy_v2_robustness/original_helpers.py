"""Verbatim numerical helpers (only dedented) from run_experiment_2.
Source commit e2c0487151ab1341ead26dba60ec44c72bc62af2,
experiments/lem_simulations.py lines 591–615. Globals retain original semantics.
"""
import numpy as np
from ripser import ripser

DIM_LATENT = 16
NOISE_LEVEL = 0.05

def system_fixed_point(state, t, lam):
    return [-lam * state[0], -lam * state[1]]

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
