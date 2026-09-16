"""Small deterministic software fixtures, never language-model study data."""
from __future__ import annotations

import numpy as np


def make_fixture_records(seed: int = 1706, profiles_per_regime: int = 6,
                         hidden_dimension: int = 16, turns: int = 24) -> list[dict]:
    """Labels are assigned independently of random state generation.

    Each known profile has two sessions per permitted phase. The toy wording
    intentionally lacks realistic regimes; no method is required to beat chance.
    No confirmation dialogs are generated.
    """
    rng = np.random.default_rng(seed)
    records = []
    for profile_index in range(3 * profiles_per_regime):
        for split in ("development", "validation"):
            for topic_index in range(2):
                context = "system: This is a synthetic software fixture.\n"
                contexts = []
                for turn in range(turns):
                    context += f"user: Compare option {rng.integers(0, 30)} with option {rng.integers(0, 30)}.\nassistant:"
                    contexts.append(context)
                    context += f" A neutral example has {rng.integers(0, 30)} units.\n"
                records.append({
                    "dialogue_id": f"fixture-{split}-{profile_index:02d}-{topic_index}",
                    "profile_id": f"fixture-profile-{profile_index:02d}",
                    "regime": f"fixture-regime-{profile_index // profiles_per_regime}",
                    "split": split, "topic_family": f"fixture-{split}-topic-{topic_index}",
                    "formulation_family": f"fixture-{split}-formulation-{topic_index}",
                    "contexts": contexts,
                    "states": np.cumsum(rng.normal(size=(turns, hidden_dimension)), axis=0),
                    "eligible_for_scientific_analysis": False, "data_kind": "synthetic_fixture",
                })
    return records
