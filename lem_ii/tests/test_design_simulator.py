"""Production-path checks using invented assistant text, never LLM study data."""
import copy
from dataclasses import replace
import unittest

from lem_ii.design import build_sessions, load_config, validate_config, validate_splits
from lem_ii.simulator import UserSimulator, response_features


CONCRETE = ("Use soil and seeds for the first planting area. Allocate 2 beds and water them for 3 minutes. "
            "Add compost before planting, then observe the area carefully and write down what happens every morning.")
UNCERTAIN = "Maybe. Perhaps this is unclear."


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.sessions = build_sessions(self.config)

    def test_complete_crossed_split_is_reproducible(self):
        self.assertEqual(self.sessions, build_sessions(load_config()))
        self.assertEqual(len(self.sessions), 108)
        self.assertEqual(len({s.seed for s in self.sessions}), 108)
        self.assertEqual(sum(s.turns for s in self.sessions), 2592)
        for field in ("topic_family", "formulation_family"):
            split_sets = [{getattr(s, field) for s in self.sessions if s.split == split}
                          for split in ("development", "validation", "confirmation")]
            self.assertTrue(all(not a & b for i, a in enumerate(split_sets) for b in split_sets[i + 1:]))

    def test_actual_plan_rejects_duplicate_and_cross_split_topic_or_formulation(self):
        duplicated = self.sessions[:-1] + [self.sessions[0]]
        with self.assertRaises(ValueError):
            validate_splits(duplicated)
        index = next(i for i, s in enumerate(self.sessions) if s.split == "validation")
        for field in ("topic_family", "formulation_family"):
            changed = list(self.sessions)
            changed[index] = replace(changed[index], **{field: getattr(self.sessions[0], field)})
            with self.assertRaises(ValueError):
                validate_splits(changed)

    def test_config_rejects_wrong_delay_count_and_prompt_metadata(self):
        config = copy.deepcopy(self.config)
        config["features"]["delay_points"] = 12
        with self.assertRaises(ValueError):
            validate_config(config)
        config = copy.deepcopy(self.config)
        config["system_prompt"] += " The regime is retain_priority."
        with self.assertRaises(ValueError):
            validate_config(config)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.sessions = build_sessions(self.config)

    def spec(self, profile):
        return next(s for s in self.sessions if s.profile_id == profile and s.topic_family == "community_garden")

    def test_same_visible_initial_prompt_for_matched_variants_across_regimes(self):
        for variant in range(6):
            sims = [UserSimulator(self.config, self.spec(f"p{regime * 6 + variant + 1:02d}")) for regime in range(3)]
            messages = [sim.next_message() for sim in sims]
            self.assertEqual(len(set(messages)), 1)
            for message, sim in zip(messages, sims):
                self.assertIsInstance(message, str)
                for forbidden in self.config["design"]["regimes"] + [sim.spec.profile_id, "profile_id", "wording_seed"]:
                    self.assertNotIn(forbidden, message)
                    self.assertNotIn(forbidden, self.config["system_prompt"])
                self.assertIn("regime", sim.last_audit)

    def test_previous_model_answer_changes_next_action_for_all_three_regimes(self):
        for profile in ("p01", "p07", "p13"):
            with self.subTest(profile=profile):
                clear = UserSimulator(self.config, self.spec(profile))
                vague = UserSimulator(self.config, self.spec(profile))
                self.assertEqual(clear.next_message(), vague.next_message())
                self.assertNotEqual(clear.next_message(CONCRETE), vague.next_message(UNCERTAIN))
                self.assertNotEqual(clear.last_audit["request_mode"], vague.last_audit["request_mode"])
                if profile != "p01":
                    self.assertNotEqual(clear.last_audit["selected_focus"], vague.last_audit["selected_focus"])

    def test_policy_actions_match_observable_rules_and_exact_word_matching(self):
        expected = {"p01": (0, "retain_priority"), "p07": (1, "advance"), "p13": (1, "least_covered")}
        for profile, (focus, action) in expected.items():
            sim = UserSimulator(self.config, self.spec(profile))
            sim.next_message()
            sim.next_message(CONCRETE)
            self.assertEqual(sim.last_audit["selected_focus"], focus)
            self.assertEqual(sim.last_audit["action"], action)
        facets = self.config["simulator"]["topics"][0]["facets"]
        features = response_features("SOIL soil soiling perhaps 12.5 3", facets, ["perhaps"])
        self.assertEqual(features["coverage"][0], 1)
        self.assertEqual(features["digit_groups"], 2)
        self.assertEqual(features["hedge_count"], 1)

    def test_reproducible_bounded_session_and_state_reset(self):
        first = UserSimulator(self.config, self.spec("p07"))
        second = UserSimulator(self.config, self.spec("p07"))
        for turn in range(24):
            response = None if turn == 0 else CONCRETE if turn % 2 else UNCERTAIN
            self.assertEqual(first.next_message(response), second.next_message(response))
            self.assertEqual(first.last_audit, second.last_audit)
            self.assertLessEqual(first.last_audit["user_word_count"], 100)
        with self.assertRaises(StopIteration):
            first.next_message(CONCRETE)
        fresh = UserSimulator(self.config, self.spec("p07"))
        self.assertEqual(fresh.audit_log, [])
        self.assertEqual(fresh.next_message(), UserSimulator(self.config, self.spec("p07")).next_message())

    def test_previous_response_contract_empty_answer_and_confirmation_guard(self):
        sim = UserSimulator(self.config, self.spec("p01"))
        with self.assertRaises(ValueError):
            sim.next_message("accidental previous session")
        sim.next_message()
        with self.assertRaises(ValueError):
            sim.next_message()
        sim.next_message("")
        self.assertTrue(sim.last_audit["previous_assistant_empty"])
        self.assertEqual(sim.last_audit["response_features"]["word_count"], 0)
        confirmation = next(s for s in self.sessions if s.split == "confirmation")
        with self.assertRaises(PermissionError):
            UserSimulator(self.config, confirmation)
        smoke = replace(self.spec("p01"), session_id="smoke-rule-test-001", split="smoke", turns=2)
        self.assertNotIn(smoke.session_id, {s.session_id for s in self.sessions})
        self.assertIsInstance(UserSimulator(self.config, smoke).next_message(), str)


if __name__ == "__main__":
    unittest.main()
