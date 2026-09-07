"""Atomic turn checkpoints with content hashes and strict compatibility checks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


class SessionStore:
    def __init__(self, root, spec, binding, *, resume=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", spec.session_id):
            raise ValueError("Unsafe session id")
        self.path = Path(root) / spec.session_id
        self.spec, self.binding = spec, binding
        if self.path.exists():
            if not resume:
                raise FileExistsError("Session already exists; explicit validated resume required")
            self.manifest = json.loads((self.path / "manifest.json").read_text())
            if self.manifest["binding"] != binding or self.manifest["spec"] != asdict(spec):
                raise ValueError("Resume model/code/config/session binding mismatch")
            self.validate()
        else:
            self.path.mkdir(parents=True)
            self.manifest = {"schema": "lem-ii.session.v1", "spec": asdict(spec), "binding": binding,
                             "status": "running", "completed_turns": 0, "turn_hashes": [], "errors": []}
            atomic_json(self.path / "labels.json", asdict(spec))
            self._save()

    def _save(self):
        atomic_json(self.path / "manifest.json", self.manifest)

    def append(self, measurement, user_message, audit):
        turn = self.manifest["completed_turns"] + 1
        if self.manifest["status"] == "complete" or turn > self.spec.turns:
            raise ValueError("Duplicate or excess session turn")
        states = np.asarray(measurement.states)
        if states.dtype != np.float32 or states.ndim != 2 or not np.isfinite(states).all():
            raise ValueError("Invalid states; never replace extraction failure with zeros")
        if list(states.shape) != self.binding["state_shape"]:
            raise ValueError("State layer/hidden dimension mismatch")
        destination = self.path / f"turn-{turn:02d}"
        if destination.exists():
            raise ValueError("Uncommitted checkpoint exists; refuse silent overwrite")
        temporary = Path(tempfile.mkdtemp(prefix="pending-", dir=self.path))
        np.save(temporary / "states.npy", states, allow_pickle=False)
        atomic_json(temporary / "visible.json", {"turn": turn, "user": user_message, **measurement.record})
        atomic_json(temporary / "audit.json", audit)
        hashes = {name: file_hash(temporary / name) for name in ("states.npy", "visible.json", "audit.json")}
        os.rename(temporary, destination)
        self.manifest["turn_hashes"].append(hashes)
        self.manifest["completed_turns"] = turn
        self.manifest["status"] = "complete" if turn == self.spec.turns else "running"
        self._save()

    def fail(self, error):
        self.manifest["status"] = "failed"
        self.manifest["errors"].append({"type": type(error).__name__, "message": str(error)[:500]})
        self._save()

    def validate(self):
        count = self.manifest["completed_turns"]
        if not 0 <= count <= self.spec.turns or len(self.manifest["turn_hashes"]) != count:
            raise ValueError("Corrupt turn count")
        if (self.manifest["status"] == "complete") != (count == self.spec.turns):
            raise ValueError("Session completion status and count disagree")
        expected = {f"turn-{turn:02d}" for turn in range(1, count + 1)}
        if {p.name for p in self.path.glob("turn-*")} != expected or list(self.path.glob("pending-*")):
            raise ValueError("Missing, extra, or partially committed checkpoint")
        if json.loads((self.path / "labels.json").read_text()) != asdict(self.spec):
            raise ValueError("Label metadata mismatch")
        for index, hashes in enumerate(self.manifest["turn_hashes"], 1):
            path = self.path / f"turn-{index:02d}"
            if set(hashes) != {"states.npy", "visible.json", "audit.json"} or any(file_hash(path / name) != h for name, h in hashes.items()):
                raise ValueError("Checkpoint checksum mismatch")
            states = np.load(path / "states.npy", allow_pickle=False)
            if list(states.shape) != self.binding["state_shape"] or states.dtype != np.float32 or not np.isfinite(states).all():
                raise ValueError("Invalid saved states")
            record = json.loads((path / "visible.json").read_text())
            if record["turn"] != index or record["input_tokens"] != len(record["input_token_ids"]):
                raise ValueError("Invalid saved turn metadata")
            inputs, outputs = record["input_token_ids"], record["output_token_ids"]
            if (not inputs or not outputs or any(type(t) is not int or t < 0 for t in inputs + outputs)
                    or record["output_tokens"] != len(outputs)
                    or len(outputs) > self.binding["max_new_tokens"]):
                raise ValueError("Saved input/output token counts or limits disagree")
            if (record["generation_prefix_position"] != len(inputs) - 1
                    or record["generation_prefix_token_id"] != inputs[-1]
                    or inputs[-3:] != [151644, 77091, 198]
                    or record["user_message_end_position"] != len(inputs) - 4
                    or record["user_message_end_token_id"] != inputs[-4]):
                raise ValueError("Saved measurement position differs from actual token prefix")
            expected_stop = "eos" if outputs[-1] in self.binding["eos_token_ids"] else "max_new_tokens"
            if record["stop_reason"] != expected_stop or (expected_stop == "max_new_tokens" and len(outputs) != self.binding["max_new_tokens"]):
                raise ValueError("Saved stop reason is inconsistent with output tokens")
            if hashlib.sha256(np.asarray(record["input_token_ids"], dtype="<i8").tobytes()).hexdigest() != record["input_token_sha256"]:
                raise ValueError("Saved input token checksum mismatch")
        return self.manifest

    def records(self):
        self.validate()
        return [json.loads((self.path / f"turn-{t:02d}" / "visible.json").read_text())
                for t in range(1, self.manifest["completed_turns"] + 1)]

    def analysis_record(self, primary_layer_index):
        rows = self.records()
        if self.manifest["status"] != "complete":
            raise ValueError("Incomplete session is not an analysis case")
        states = np.stack([np.load(self.path / f"turn-{t:02d}" / "states.npy", allow_pickle=False)[primary_layer_index]
                           for t in range(1, len(rows) + 1)])
        return {"dialogue_id": self.spec.session_id, "profile_id": self.spec.profile_id, "regime": self.spec.regime,
                "split": self.spec.split, "topic_family": self.spec.topic_family,
                "formulation_family": self.spec.formulation_family, "contexts": [r["context"] for r in rows],
                "states": states, "data_kind": self.binding["data_kind"],
                "eligible_for_scientific_analysis": self.binding["eligible_for_scientific_analysis"]}
