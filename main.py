"""Canonical project entry point.

Usage examples:
    python3 main.py baseline
    python3 main.py new_idea_1
    python3 main.py new_idea_2
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_python_script(script_path: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script_path), *args], check=True, cwd=PROJECT_ROOT)


def run_baseline() -> None:
    run_python_script(PROJECT_ROOT / "baseline" / "reproduction" / "run_baseline.py")


def run_new_idea_1() -> None:
    run_python_script(PROJECT_ROOT / "New_idea" / "1" / "run_new_idea.py")


def run_new_idea_2() -> None:
    run_python_script(PROJECT_ROOT / "New_idea" / "2" / "run_hybrid_benchmark.py")


def run_paper_tables() -> None:
    run_python_script(PROJECT_ROOT / "baseline" / "reproduction" / "generate_paper_tables.py")
    run_python_script(PROJECT_ROOT / "New_idea" / "2" / "generate_paper_tables.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the thesis benchmark pipeline.")
    parser.add_argument(
        "target",
        choices=["baseline", "new_idea_1", "new_idea_2", "paper_tables", "all"],
        help="Which benchmark pipeline to run.",
    )
    args = parser.parse_args()

    if args.target in {"baseline", "all"}:
        run_baseline()
    if args.target in {"new_idea_1", "all"}:
        run_new_idea_1()
    if args.target in {"new_idea_2", "all"}:
        run_new_idea_2()
    if args.target in {"paper_tables", "all"}:
        run_paper_tables()


if __name__ == "__main__":
    main()
