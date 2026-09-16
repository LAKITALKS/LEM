"""Prespecified session assignments. Constructing a plan does not collect data."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONFIG = Path(__file__).with_name("config") / "study.json"
SPLITS = ("development", "validation", "confirmation")


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    profile_id: str
    regime: str
    split: str
    topic_family: str
    formulation_family: str
    seed: int
    turns: int


def stable_seed(*parts: object) -> int:
    """Stable across Python processes; never use randomized Python hash()."""
    return int.from_bytes(sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % 2**32


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject silent scientific changes or inconsistent executable specifications."""
    d, f, a, m = (config[name] for name in ("design", "features", "analysis", "model"))
    expected = {"profiles": 18, "sessions_per_profile": 6, "turns_per_session": 24,
                "planned_sessions": 108, "planned_model_responses": 2592, "variants_per_regime": 6}
    for key, value in expected.items():
        if d[key] != value:
            raise ValueError(f"Prespecified design requires {key}={value}")
    if d["regimes"] != ["retain_priority", "advance_when_addressed", "revise_from_answer"]:
        raise ValueError("Unknown or reordered regime specification")
    if set(d["splits"]) != set(SPLITS) or any(len(d["splits"][s]) != 2 for s in SPLITS):
        raise ValueError("Exactly two topics are required in each split")
    topics = config["simulator"]["topics"]
    if len(topics) != 6 or len({t["id"] for t in topics}) != 6:
        raise ValueError("Six unique topic families are required")
    families: dict[str, str] = {}
    for topic in topics:
        if topic["split"] not in SPLITS or topic["id"] not in d["splits"][topic["split"]]:
            raise ValueError("Topic split disagrees with design split")
        family = topic["formulation_family"]
        if family in families:
            raise ValueError("Formulation families must be unique to a topic/split")
        families[family] = topic["split"]
        templates = config["simulator"]["formulations"].get(family)
        if not templates or any(not all("{" + k + "}" in t for k in ("topic", "facet", "task", "request")) for t in templates):
            raise ValueError("Missing or malformed formulation family")
        if len(topic["facets"]) != 3 or any(len(set(f["keywords"])) < 2 for f in topic["facets"]):
            raise ValueError("Each topic requires three facets with at least two distinct keywords")
    if {t["id"] for t in topics} != {t for v in d["splits"].values() for t in v}:
        raise ValueError("Design and simulator topic sets disagree")
    if (f["window"], f["delay_lag"], f["delay_dimension"], f["delay_points"], f["pca_components"]) != (12, 1, 3, 10, 8):
        raise ValueError("Prespecified window/delay/PCA dimensions changed")
    if f["delay_points"] != f["window"] - f["delay_lag"] * (f["delay_dimension"] - 1):
        raise ValueError("Delay point count inconsistent with window")
    if f["endpoints"] != [12, 16, 20, 24] or a["recognition_grid"] != f["endpoints"] or d["primary_endpoint"] != 24:
        raise ValueError("Endpoint grids disagree with protocol")
    if config["extraction"]["layers"] != [0, 18, 36] or config["extraction"]["primary_layer"] != 36:
        raise ValueError("Unexpected extraction layer selection")
    if m["revision"] != m["tokenizer_revision"] or len(m["revision"]) != 40:
        raise ValueError("Model and tokenizer require the same pinned commit")
    if m["dtype"] != "bfloat16" or m["quantization"] is not None:
        raise ValueError("Silent precision or quantization changes are prohibited")
    if m["max_context_tokens"] != 8192 or m["overflow_policy"] != "fail_without_truncation":
        raise ValueError("Unexpected context policy")
    if a["C_grid"] != [0.1, 1.0, 10.0] or a["primary_comparisons"] != len(a["comparison_pairs"]):
        raise ValueError("Tuning or comparison family disagrees with protocol")
    if a["primary_interval_coverage"] != 1 - a["alpha"] / a["primary_comparisons"]:
        raise ValueError("Primary interval multiplicity correction is inconsistent")
    forbidden = d["regimes"] + ["profile_id", "regime", "generator metadata"]
    if any(token in config["system_prompt"].lower() for token in forbidden):
        raise ValueError("System prompt includes hidden study metadata")


def build_sessions(config: dict[str, Any]) -> list[SessionSpec]:
    validate_config(config)
    result = []
    for regime_index, regime in enumerate(config["design"]["regimes"]):
        for variant_index in range(config["design"]["variants_per_regime"]):
            profile_id = f"p{regime_index * 6 + variant_index + 1:02d}"
            for topic in config["simulator"]["topics"]:
                session_id = f"study-{profile_id}-{topic['id']}"
                result.append(SessionSpec(session_id, profile_id, regime, topic["split"], topic["id"],
                                          topic["formulation_family"], stable_seed(config["seeds"]["design"], session_id),
                                          config["design"]["turns_per_session"]))
    validate_splits(result)
    return result


def validate_splits(sessions: Iterable[SessionSpec]) -> None:
    """Validate the entire 108-session assignment, including crossed-factor balance.

    Call before selecting a collection subset. Prefixes/windows carry their parent
    session_id; this plan never assigns individual turns to different splits.
    """
    sessions = list(sessions)
    if len(sessions) != 108 or len({s.session_id for s in sessions}) != 108:
        raise ValueError("Plan must contain exactly 108 unique sessions")
    if any(s.turns != 24 or not s.session_id.startswith("study-") for s in sessions):
        raise ValueError("Invalid study session scope")
    topics, formulations, profile_regimes = {}, {}, {}
    by_split: dict[str, list[SessionSpec]] = defaultdict(list)
    for s in sessions:
        if s.split not in SPLITS:
            raise ValueError("Unknown split")
        for mapping, key in ((topics, s.topic_family), (formulations, s.formulation_family)):
            if key in mapping and mapping[key] != s.split:
                raise ValueError("Topic or formulation leakage across splits")
            mapping[key] = s.split
        if s.profile_id in profile_regimes and profile_regimes[s.profile_id] != s.regime:
            raise ValueError("Profile changes regime across sessions")
        profile_regimes[s.profile_id] = s.regime
        by_split[s.split].append(s)
    if len(topics) != 6 or len(formulations) != 6 or len(profile_regimes) != 18:
        raise ValueError("Incorrect topic, formulation, or profile count")
    if sorted(Counter(profile_regimes.values()).values()) != [6, 6, 6]:
        raise ValueError("Each of three regimes requires six profiles")
    if len({(s.profile_id, s.topic_family) for s in sessions}) != 108:
        raise ValueError("Duplicate profile/topic session")
    for split in SPLITS:
        subset = by_split[split]
        if len(subset) != 36 or set(s.profile_id for s in subset) != set(profile_regimes):
            raise ValueError("Each split must contain all 18 known profiles, twice")
        if len({s.topic_family for s in subset}) != 2 or len({s.formulation_family for s in subset}) != 2:
            raise ValueError("Each split requires two disjoint topic/formulation families")
        for topic in {s.topic_family for s in subset}:
            if Counter(s.regime for s in subset if s.topic_family == topic) != Counter({r: 6 for r in set(profile_regimes.values())}):
                raise ValueError("Regimes must be balanced within every topic")
