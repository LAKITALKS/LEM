"""Real billing-schema regression and independently known cost arithmetic."""
import json
import subprocess
import unittest
from unittest.mock import patch
from lem_ii import launch_smoke
from lem_ii.summarize_smoke import billing_costs


class AccountingTests(unittest.TestCase):
    def test_provider_rows_exclude_other_apps_and_include_outside_worker_gpu_time(self):
        rows = [{"object_id": "owned", "resource": "L4", "cost": "0.0666"},
                {"object_id": "owned", "resource": "CPU", "cost": "0.01"},
                {"object_id": "unrelated", "resource": "L4", "cost": "900"}]
        result = billing_costs(rows, "owned", 280)
        self.assertAlmostEqual(result["reported_gross_usage_usd"], .0766)
        self.assertAlmostEqual(result["gpu_billable_seconds_inferred_from_rate"], 300)
        self.assertAlmostEqual(result["outside_worker_gpu_seconds"], 20)
        self.assertTrue(result["metering_can_lag"])
        for cost in ("NaN", "-1", "inf"):
            rows[0]["cost"] = cost
            with self.assertRaises(ValueError):
                billing_costs(rows, "owned", 280)

    def test_actual_modal_snake_case_stopped_app_schema_is_verified(self):
        rows = [{"app_id": "ap-owned", "state": "stopped", "tasks": "0"}]
        ack = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(launch_smoke.subprocess, "run", return_value=ack), \
                patch.object(launch_smoke, "command", return_value=json.dumps(rows)):
            self.assertTrue(launch_smoke.stop_owned_app("ap-owned")["stop_requested_or_already_stopped"])


if __name__ == "__main__":
    unittest.main()
