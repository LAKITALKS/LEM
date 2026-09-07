"""Bounded answer-responsive synthetic users; hidden audit metadata stays separate."""
from __future__ import annotations

import random
import re
from typing import Any

from .design import SessionSpec, stable_seed

WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
DIGITS = re.compile(r"\d+(?:[.,]\d+)?")


def response_features(answer: str, facets: list[dict[str, Any]], hedge_terms: list[str]) -> dict[str, Any]:
    """Exact case-folded word matches: no semantic scorer or second model."""
    words = WORD.findall(answer.casefold())
    word_set = set(words)
    return {
        "word_count": len(words), "character_count": len(answer),
        "digit_groups": len(DIGITS.findall(answer)),
        "hedge_count": sum(words.count(term.casefold()) for term in hedge_terms),
        "coverage": [len(word_set & {w.casefold() for w in facet["keywords"]}) for facet in facets],
    }


class UserSimulator:
    """One object per session. next_message returns visible text only.

    audit_log contains the hidden policy execution and lengths for separate storage.
    Previous responses affect focus selection and/or the request mode. This is a
    lexical state machine, not a model of human users or semantic understanding.
    """

    def __init__(self, config: dict[str, Any], spec: SessionSpec):
        self.config, self.spec = config, spec
        if spec.split == "confirmation" and not config["execution"]["confirmation_generation_allowed"]:
            raise PermissionError("Confirmation generation is not authorized in this technical task")
        if spec.regime not in config["design"]["regimes"]:
            raise ValueError("Unknown regime")
        if not re.fullmatch(r"p(?:0[1-9]|1[0-8])", spec.profile_id):
            raise ValueError("Profile id must identify one of the 18 planned variants")
        profile_number = int(spec.profile_id[1:]) - 1
        if config["design"]["regimes"][profile_number // 6] != spec.regime:
            raise ValueError("Profile id and regime disagree")
        self.variant = profile_number % 6
        self.priority = self.variant % 3
        self.order = [(self.priority + i) % 3 for i in range(3)]
        self.thresholds = config["simulator"]["sensitivity"]["strict" if self.variant >= 3 else "lenient"]
        matches = [t for t in config["simulator"]["topics"] if t["id"] == spec.topic_family]
        if len(matches) != 1:
            raise ValueError("Unknown topic family")
        self.topic = matches[0]
        if self.topic["formulation_family"] != spec.formulation_family or (spec.split != "smoke" and self.topic["split"] != spec.split):
            raise ValueError("Topic/formulation/split assignment is inconsistent")
        if not 1 <= spec.turns <= config["design"]["turns_per_session"]:
            raise ValueError("Session must be bounded to at most 24 turns")
        self.wording_seed = stable_seed(config["seeds"]["design"], "wording", spec.topic_family, self.variant)
        self.rng = random.Random(self.wording_seed)
        self.focus = self.priority
        self.audit_log: list[dict[str, Any]] = []

    @property
    def last_audit(self) -> dict[str, Any] | None:
        return self.audit_log[-1] if self.audit_log else None

    def next_message(self, previous_assistant: str | None = None) -> str:
        turn = len(self.audit_log) + 1
        if turn > self.spec.turns:
            raise StopIteration("Prespecified session turn limit reached")
        if turn == 1 and previous_assistant is not None:
            raise ValueError("A new session must start without a previous assistant response")
        if turn > 1 and not isinstance(previous_assistant, str):
            raise ValueError("A previous assistant response is required after turn 1")
        previous_focus = self.focus
        features = None
        adequate = False
        concrete = False
        action = "initial_priority"
        if previous_assistant is not None:
            features = response_features(previous_assistant, self.topic["facets"], self.config["simulator"]["hedge_terms"])
            t = self.thresholds
            adequate = (features["coverage"][self.focus] >= t["minimum_distinct_keywords"]
                        and features["word_count"] >= t["minimum_answer_words"]
                        and features["hedge_count"] <= t["maximum_hedges"])
            concrete = (features["word_count"] >= t["minimum_answer_words"]
                        and features["digit_groups"] >= t["minimum_digit_groups"]
                        and features["hedge_count"] <= t["maximum_hedges"])
            if self.spec.regime == "retain_priority":
                self.focus = self.priority
                action = "retain_priority"
            elif self.spec.regime == "advance_when_addressed":
                self.focus = (self.focus + 1) % 3 if adequate else self.focus
                action = "advance" if adequate else "retain_unaddressed"
            else:
                coverage = features["coverage"]
                target = min(coverage) if concrete else max(coverage)
                self.focus = next(i for i in self.order if coverage[i] == target)
                action = "least_covered" if concrete else "most_covered"
        mode = "apply" if adequate else "clarify"
        facet = self.topic["facets"][self.focus]
        template_index = self.rng.randrange(len(self.config["simulator"]["formulations"][self.spec.formulation_family]))
        template = self.config["simulator"]["formulations"][self.spec.formulation_family][template_index]
        text = template.format(topic=self.topic["title"], facet=facet["name"], task=facet["task"],
                               request=self.config["simulator"]["followup_modes"][mode])
        user_words = len(WORD.findall(text))
        if user_words > self.config["simulator"]["max_user_words"]:
            raise ValueError("User message exceeds the prespecified word limit; no truncation applied")
        self.audit_log.append({
            "turn": turn, "session_id": self.spec.session_id, "profile_id": self.spec.profile_id,
            "regime": self.spec.regime, "variant_index": self.variant, "seed": self.spec.seed,
            "wording_seed": self.wording_seed, "previous_focus": previous_focus, "selected_focus": self.focus,
            "response_features": features, "adequate": adequate, "concrete": concrete,
            "action": action, "request_mode": mode, "template_index": template_index,
            "user_word_count": user_words, "user_character_count": len(text),
            "previous_assistant_empty": previous_assistant == "" if previous_assistant is not None else None,
        })
        return text
