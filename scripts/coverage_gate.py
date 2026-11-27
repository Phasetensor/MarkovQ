"""Independently enforce the two coverage thresholds."""

import json
from pathlib import Path

totals = json.loads(Path("coverage.json").read_text())["totals"]
for kind, passed, total, target in [
    ("statement", "covered_lines", "num_statements", 95),
    ("branch", "covered_branches", "num_branches", 90),
]:
    value = 100 * totals[passed] / totals[total]
    print(f"{kind} coverage: {value:.2f}%")
    if value < target:
        raise SystemExit(f"{kind} coverage below {target}%")
