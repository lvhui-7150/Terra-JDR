"""Common command dispatcher for verification, figures, and experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "verify": ROOT / "code" / "core" / "replay.py",
    "figures": ROOT / "code" / "visualization" / "make_figures.py",
    "experiments": ROOT / "code" / "core" / "supplement_experiments.py",
    "q1": ROOT / "code" / "core" / "single_site_model.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    args = parser.parse_args()
    command = [sys.executable, str(TASKS[args.task])]
    if args.task == "q1":
        command += [
            "--inputs",
            str(ROOT / "data" / "processed" / "transport.json"),
            "--dem",
            str(ROOT / "data" / "processed" / "dem.tif"),
            "--step-m",
            "2",
            "--out",
            str(ROOT / "results" / "optimization" / "q1_recomputed"),
        ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
