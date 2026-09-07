"""Run the same regressions against corrected code and the old mathematical target.

The red mutant retains the helper interface and all production call sites: only
the missing denominator is restored. A second red run tests the actual archived
production bodies, without introducing a helper dependency into those bodies.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/lem_simulations.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/fixed_point_correction/tests")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "experiments/tests", "-v"]
    source = SOURCE.read_text()
    correct = "target = (alpha * user_signatures + beta * domain_vector) / (alpha + beta)"
    wrong = "target = alpha * user_signatures + beta * domain_vector"
    assert source.count(correct) == 1
    results = {}
    with tempfile.TemporaryDirectory(prefix="lem-mathematical-mutant-") as folder:
        mutant = Path(folder) / "lem_simulations.py"
        mutant.write_text(source.replace(correct, wrong))
        archived = Path(folder) / "archived_simulations.py"
        archived.write_bytes(subprocess.check_output([
            "git", "show", "f15073f2cab3fb68857ffb16035d6675957c2e06:experiments/lem_simulations.py"], cwd=ROOT))
        for label, selected, expected_failures, pattern in [
            ("red_old_target", mutant, 5, []),
            ("red_archived_production", archived, 3, ["-k", "production_path"]),
            ("green_corrected", SOURCE, 0, []),
        ]:
            env = dict(os.environ, LEM_SIMULATIONS_SOURCE=str(selected))
            actual_command = command + pattern
            result = subprocess.run(actual_command, cwd=ROOT, env=env, text=True, capture_output=True)
            log = result.stdout + result.stderr
            (args.output_dir / f"{label}.log").write_text(log)
            if expected_failures:
                assert result.returncode == 1, log
                assert f"FAILED (failures={expected_failures})" in log and "ERROR:" not in log, log
            else:
                assert result.returncode == 0 and "Ran 5 tests" in log, log
            results[label] = {"exit_code": result.returncode, "expected_assertion_failures": expected_failures,
                              "command": actual_command, "source_selection": label}
            print(f"{label}: expected outcome confirmed ({expected_failures} mathematical assertion failures)")
    (args.output_dir / "verification.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
