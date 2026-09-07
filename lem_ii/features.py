"""Train-only features and explicitly separate technical smoke measurements.

Metadata are inspected for guards, never vectorized. The only feature inputs are
the saved pre-answer context at the endpoint and preceding primary-layer states.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


METHODS = (
    "text",
    "text_static",
    "text_static_geometry_time",
    "text_static_geometry_time_topology",
)
RECORD_KEYS = frozenset({
    "dialogue_id", "profile_id", "regime", "split", "topic_family",
    "formulation_family", "contexts", "states",
    "eligible_for_scientific_analysis", "data_kind",
})


class NotEvaluable(ValueError):
    """Insufficient genuine observations, never replaced by zero features."""


class LeakageError(ValueError):
    """An input violates the declared split or feature boundary."""


def _feature_config(config: dict) -> dict:
    return config.get("features", config)


def validate_records(records: list[dict], *, config: dict | None = None,
                     allow_confirmation_metadata=False) -> None:
    """Validate entire loaded collection before any phase is selected.

    Profiles repeat intentionally, while dialogs, themes and formulation families
    do not cross phases. Metadata-only confirmation assignments may be audited.
    """
    gates = (config or {}).get("execution", {})
    kinds = {row.get("data_kind") for row in records}
    if len(kinds) > 1:
        raise LeakageError("Fixtures, technical smoke and scientific study records cannot be mixed")
    planned = None
    if kinds == {"real_model_study"} and gates.get("allow_scientific_analysis", False):
        from .design import build_sessions
        planned = {session.session_id: session for session in build_sessions(config)}
    ids: set[str] = set()
    owners: dict[str, dict[str, str]] = {"topic_family": {}, "formulation_family": {}}
    profiles: dict[str, str] = {}
    for record in records:
        unknown = set(record) - RECORD_KEYS
        missing = RECORD_KEYS - set(record)
        if unknown or missing:
            raise LeakageError(f"Record keys rejected: missing={sorted(missing)}, unknown={sorted(unknown)}")
        identifier = record["dialogue_id"]
        if identifier in ids:
            raise LeakageError(f"Duplicate dialogue {identifier}")
        ids.add(identifier)
        split = record["split"]
        if split not in {"development", "validation", "confirmation", "smoke"}:
            raise LeakageError(f"Unknown split {split}")
        if split == "confirmation" and not allow_confirmation_metadata and not gates.get("allow_confirmation_analysis", False):
            raise LeakageError("Confirmation loading, transformation and evaluation are disabled in this Auftrag")
        if record["data_kind"] not in {"synthetic_fixture", "real_model_smoke", "real_model_study"}:
            raise LeakageError("Unknown data kind")
        if record["data_kind"] == "real_model_study":
            if not gates.get("allow_scientific_analysis", False):
                raise LeakageError("Scientific analysis needs separate future execution authorization")
            if not record["eligible_for_scientific_analysis"]:
                raise LeakageError("Real study record is not eligible for scientific analysis")
        elif record["eligible_for_scientific_analysis"]:
            raise LeakageError("Fixtures and smoke cannot be eligible scientific data")
        if planned is not None:
            specification = planned.get(identifier)
            if specification is None or any(record[field] != getattr(specification, field) for field in
                                             ("profile_id", "regime", "split", "topic_family", "formulation_family")):
                raise LeakageError("Study dialog metadata do not match the prespecified session assignment")
        profile = record["profile_id"]
        if profile in profiles and profiles[profile] != record["regime"]:
            raise LeakageError(f"Regime changes within profile bundle {profile}")
        profiles[profile] = record["regime"]
        for field, field_owners in owners.items():
            value = record[field]
            if value in field_owners and field_owners[value] != split:
                raise LeakageError(f"{field} crosses phases: {value}")
            field_owners[value] = split
        contexts = record["contexts"]
        states = np.asarray(record["states"])
        if states.ndim != 2 or states.shape[0] != len(contexts) or states.shape[1] == 0:
            raise LeakageError("Contexts and saved primary-layer states must align one-to-one")
        if not np.isfinite(states).all():
            raise NotEvaluable("Nonfinite state values; extraction failures cannot become zeros")
        if not all(isinstance(context, str) and context for context in contexts):
            raise LeakageError("Every state requires its exact nonempty visible pre-answer context")
        # Detect identifiers/metadata fields; ordinary words that happen to be regime
        # names are not prohibited because the visible behavior itself is the target.
        for context in contexts:
            if re.search(r"\b(?:profile_id|regime_label|generator_metadata|formulation_family)\b", context):
                raise LeakageError("Generator metadata appeared in visible text")
            if len(str(profile)) >= 3 and re.search(r"(?<!\w)" + re.escape(str(profile)) + r"(?!\w)", context):
                raise LeakageError("Profile identifier appeared in visible text")


def delay_embedding(sequence: np.ndarray, lag: int = 1, dimension: int = 3) -> np.ndarray:
    """Rows [x_t, x_(t+lag), ..., x_(t+(dimension-1)*lag)]."""
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2 or lag < 1 or dimension < 1:
        raise ValueError("Delay embedding requires a 2D sequence, positive lag/dimension")
    count = len(sequence) - (dimension - 1) * lag
    if count < 2:
        raise NotEvaluable(f"Delay embedding has {count} points, requires at least two")
    return np.concatenate([sequence[j * lag:j * lag + count] for j in range(dimension)], axis=1)


def persistence_features(points: np.ndarray) -> np.ndarray:
    """Euclidean Vietoris-Rips H0/H1: finite count, sum, max, entropy.

    The single essential H0 class and any other infinite bars are excluded.
    A genuinely empty finite diagram has count/sum/max/entropy zero, which is
    distinct from a sequence that failed extraction or was too short.
    """
    from ripser import ripser

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise NotEvaluable("Persistence requires at least two finite genuine points")
    summaries: list[float] = []
    distances = squareform(pdist(points, metric="euclidean"))
    for diagram in ripser(distances, maxdim=1, distance_matrix=True, coeff=2)["dgms"]:
        finite = diagram[np.isfinite(diagram).all(axis=1)]
        lifetimes = finite[:, 1] - finite[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        total = float(lifetimes.sum())
        entropy = float(-np.sum((lifetimes / total) * np.log(lifetimes / total))) if total else 0.0
        summaries.extend([float(len(lifetimes)), total, float(lifetimes.max()) if len(lifetimes) else 0.0, entropy])
    return np.asarray(summaries, dtype=np.float64)


def _distance_summary(points: np.ndarray) -> list[float]:
    distances = pdist(points, metric="euclidean")
    centered = points - points.mean(axis=0)
    return [float(distances.mean()), float(distances.std()), float(distances.max()),
            float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))]


def activation_blocks(sequence: np.ndarray, config: dict, *, endpoint: int,
                      shuffle_seed: int | None = None) -> dict[str, np.ndarray]:
    """All paths share the exact same window and delay-embedded points."""
    settings = _feature_config(config)
    window = int(settings.get("window", 12))
    lag = int(settings.get("delay_lag", 1))
    dimension = int(settings.get("delay_dimension", 3))
    sequence = np.asarray(sequence, dtype=np.float64)
    if endpoint < window or len(sequence) < endpoint:
        raise NotEvaluable(f"Endpoint {endpoint} requires {window} observed window states; received {len(sequence)}")
    if sequence.ndim != 2 or not np.isfinite(sequence[:endpoint]).all():
        raise NotEvaluable("Invalid activation sequence")
    observed = sequence[endpoint - window:endpoint].copy()
    if shuffle_seed is not None:
        # Keep the measured current state fixed, so the entire static baseline
        # remains identical while preceding temporal ordering changes.
        order = np.append(np.random.default_rng(shuffle_seed).permutation(window - 1), window - 1)
        observed = observed[order]
    static = np.concatenate([observed[-1], observed.mean(axis=0), observed.std(axis=0, ddof=0),
                             _distance_summary(observed)])
    steps = np.diff(observed, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    denominators = norms[:-1] * norms[1:]
    cosines = np.divide(np.sum(steps[:-1] * steps[1:], axis=1), denominators,
                        out=np.zeros_like(denominators), where=denominators > 0)
    geometry_time = [float(norms.mean()), float(norms.std()), float(norms.max()),
                      float(cosines.mean()), float(cosines.std())]
    for displacement_lag in (1, 2, 3):
        lengths = np.linalg.norm(observed[displacement_lag:] - observed[:-displacement_lag], axis=1)
        geometry_time += [float(lengths.mean()), float(lengths.std())]
    # Recomputed AFTER shuffling predecessors while keeping the endpoint fixed.
    # Permuting already embedded points would be an
    # invariant point-cloud control and would not test temporal dependence.
    embedded = delay_embedding(observed, lag, dimension)
    geometry_time += _distance_summary(embedded)
    deviations = embedded.std(axis=0, ddof=0)
    geometry_time += [float(deviations.mean()), float(deviations.max())]
    return {"static": static, "geometry_time": np.asarray(geometry_time),
            "topology": persistence_features(embedded)}


def style_features(text: str) -> np.ndarray:
    words = re.findall(r"\b\w+\b", text)
    nchar = max(len(text), 1)
    return np.asarray([len(text), len(words), len(text.splitlines()),
                       sum(map(len, words)) / max(len(words), 1),
                       len(set(word.lower() for word in words)) / max(len(words), 1),
                       text.count("?") / nchar, text.count("!") / nchar,
                       sum(char.isdigit() for char in text) / nchar], dtype=np.float64)


class FeaturePipeline:
    """Development-fitted pipeline for a single predeclared endpoint."""

    def __init__(self, config: dict):
        self.config = config

    def fit(self, records: list[dict], endpoint: int = 24) -> "FeaturePipeline":
        validate_records(records, config=self.config)
        if not records or any(r["split"] != "development" or r["data_kind"] == "real_model_smoke" for r in records):
            raise LeakageError("Fit only development fixture records or separately authorized development study records")
        self.endpoint = endpoint
        self.fit_data_kind = records[0]["data_kind"]
        self.fit_dialogue_ids = tuple(r["dialogue_id"] for r in records)
        if any(len(r["states"]) < endpoint for r in records):
            raise NotEvaluable("A development dialog does not reach this endpoint")
        settings = _feature_config(self.config)
        components = int(settings.get("pca_components", 8))
        states = np.concatenate([np.asarray(r["states"][:endpoint], dtype=np.float64) for r in records])
        if min(states.shape) < components:
            raise NotEvaluable("Too few genuine state samples or dimensions for the fixed PCA")
        self.pca = PCA(n_components=components, svd_solver="full", whiten=False)
        projected = self.pca.fit_transform(states)
        self.state_scaler = StandardScaler().fit(projected)
        texts = [r["contexts"][endpoint - 1] for r in records]
        self.word = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=12000, min_df=1, sublinear_tf=True)
        self.word.fit(texts)
        self.char.fit(texts)
        self.style_scaler = StandardScaler().fit(np.vstack([style_features(text) for text in texts]))
        raw = self._raw_blocks(records)
        self.block_scalers = {name: StandardScaler().fit(values) for name, values in raw.items()}
        return self

    def _raw_blocks(self, records: list[dict], *, time_shuffle=False) -> dict[str, np.ndarray]:
        rows: dict[str, list] = {"static": [], "geometry_time": [], "topology": []}
        for record in records:
            states = np.asarray(record["states"][:self.endpoint], dtype=np.float64)
            projected = self.state_scaler.transform(self.pca.transform(states))
            seed = None
            if time_shuffle:
                master_seed = self.config.get("seeds", {}).get("time_shuffle", 1705)
                identity = f"{master_seed}:{record['dialogue_id']}:{self.endpoint}"
                seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            blocks = activation_blocks(projected, self.config, endpoint=self.endpoint, shuffle_seed=seed)
            for name in rows:
                rows[name].append(blocks[name])
        return {name: np.vstack(values) for name, values in rows.items()}

    def transform(self, records: list[dict], endpoint: int | None = None, *, time_shuffle=False) -> dict[str, sparse.csr_matrix]:
        validate_records(records, config=self.config)
        if not hasattr(self, "pca"):
            raise RuntimeError("Fit development transformations first")
        if endpoint is not None and endpoint != self.endpoint:
            raise ValueError("An endpoint needs its own development-fitted feature pipeline")
        if any(r["data_kind"] == "real_model_smoke" for r in records):
            raise LeakageError("Real smoke uses compute_technical_features; scientific transformations are not fit on smoke")
        if any(r["data_kind"] != self.fit_data_kind for r in records):
            raise LeakageError("A fitted artifact cannot mix fixture and genuine study records")
        if any(len(r["states"]) < self.endpoint for r in records):
            raise NotEvaluable("A dialog does not reach this endpoint")
        texts = [r["contexts"][self.endpoint - 1] for r in records]
        style = self.style_scaler.transform(np.vstack([style_features(text) for text in texts]))
        text = sparse.hstack([self.word.transform(texts), self.char.transform(texts), sparse.csr_matrix(style)], format="csr")
        raw = self._raw_blocks(records, time_shuffle=time_shuffle)
        blocks = {name: sparse.csr_matrix(self.block_scalers[name].transform(value)) for name, value in raw.items()}
        static = sparse.hstack([text, blocks["static"]], format="csr")
        geometry = sparse.hstack([static, blocks["geometry_time"]], format="csr")
        topology = sparse.hstack([geometry, blocks["topology"]], format="csr")
        return dict(zip(METHODS, [text, static, geometry, topology]))


def compute_technical_features(states: np.ndarray, config: dict, endpoint: int = 24) -> dict[str, Any]:
    """Exercise every activation feature on saved smoke states, without learning.

    Full hidden dimensions are unit-normalized per vector; this is a technical
    path, not the scientific PCA representation or a classification result.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise NotEvaluable("Invalid saved smoke states")
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise NotEvaluable("Zero activation vector cannot be unit normalized")
    normalized = states / norms
    blocks = activation_blocks(normalized, config, endpoint=endpoint)
    shuffled = activation_blocks(normalized, config, endpoint=endpoint,
                                 shuffle_seed=config.get("seeds", {}).get("time_shuffle", 1705))
    settings = _feature_config(config)
    window = int(settings.get("window", 12))
    observed = normalized[endpoint-window:endpoint]
    plain = persistence_features(observed)
    permuted = persistence_features(observed[::-1])
    return {"data_kind": "real_model_smoke", "eligible_for_scientific_analysis": False,
            "representation": "technical_only_full_hidden_per_vector_L2_no_fitted_transform",
            "endpoint": endpoint, "hidden_dimension": states.shape[1],
            "delay_point_count": window - (int(settings.get("delay_dimension", 3)) - 1) * int(settings.get("delay_lag", 1)),
            "features": {name: values.tolist() for name, values in blocks.items()},
            "time_shuffled_features": {name: values.tolist() for name, values in shuffled.items()},
            "point_cloud_permutation_invariant": bool(np.allclose(plain, permuted, rtol=1e-6, atol=1e-7)),
            "finite_features": bool(all(np.isfinite(values).all() for values in blocks.values()))}
