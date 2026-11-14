#!/usr/bin/env python3
r"""
Consolidated Gen 1 runner for all 30 benchmark subsets.

This is the single, manual-run script left in this folder after cleaning up the
older exploratory copies. It uses the verified subset registry in the project and
reuses the stable WNCROD implementation from the baseline runner.

Usage:
    python implementations/4_Gen\ 1/run_gen1.py
    python "implementations/4_Gen 1/run_gen1.py" --family MA
    python "implementations/4_Gen 1/run_gen1.py" --subset MA1
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset, SUBSET_REGISTRY
from evaluators import evaluate_predictions

RESULT_DIR = PROJECT_ROOT / "result" / "4_Gen 1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAMBDAS = [round(x, 1) for x in np.arange(0.1, 2.05, 0.1)]

FAMILY_GROUPS = {
    "A": ["A1", "A2", "A3"],
    "C": ["C1", "C2", "C3"],
    "G": ["G1", "G2", "G3"],
    "H": ["H1", "H2", "H3"],
    "L": ["L1", "L2", "L3"],
    "MA": ["MA1", "MA2", "MA3"],
    "M": ["M1", "M2", "M3"],
    "MU": ["MU1", "MU2", "MU3"],
    "TH": ["TH1", "TH2", "TH3"],
    "W": ["W1", "W2", "W3"],
}


def load_baseline_runner():
    """Load the existing, verified WNCROD implementation from the baseline runner."""
    module_path = PROJECT_ROOT / "implementations" / "1_reproduction" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load baseline runner from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = load_baseline_runner()
WNCROD_optimized = BASELINE.WNCROD_optimized


def sweep_subset(sub_name: str, lambdas: list[float]):
    _, y, X_tem = load_subset(sub_name, validate=False)
    data_path = PROJECT_ROOT / "baseline" / "datasets" / SUBSET_REGISTRY[sub_name]["mat_file"]
    data_raw = __import__("scipy.io").io.loadmat(str(data_path))["trandata"][:, :-1].astype(float)

    sweep_aucs = []
    sweep_aps = []
    for lam in lambdas:
        scores = WNCROD_optimized(data_raw, X_tem, lammda=lam)
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem, elapsed_time=0.0)
        sweep_aucs.append(metrics["auroc"])
        sweep_aps.append(metrics["ap"])

    best_idx_ap = int(np.argmax(sweep_aps))
    best_idx_roc = int(np.argmax(sweep_aucs))
    return {
        "subset": sub_name,
        "lambda_values": lambdas,
        "roc_auc": sweep_aucs,
        "ap": sweep_aps,
        "best_lambda_ap": lambdas[best_idx_ap],
        "best_ap": sweep_aps[best_idx_ap],
        "best_lambda_roc": lambdas[best_idx_roc],
        "best_roc": sweep_aucs[best_idx_roc],
    }


def print_family_summary(family_name: str, family_subsets: list[str], results: dict):
    print(f"\n=== {family_name} ===")
    for sub_name in family_subsets:
        item = results[sub_name]
        print(
            f"[{sub_name}] best λ(AP)={item['best_lambda_ap']:.1f}, best AP={item['best_ap']:.4f}; "
            f"best λ(ROC)={item['best_lambda_roc']:.1f}, best ROC-AUC={item['best_roc']:.4f}"
        )


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub_name in sorted(results):
        item = results[sub_name]
        rows.append({
            "subset": sub_name,
            "best_lambda_ap": round(float(item["best_lambda_ap"]), 1),
            "best_ap": round(float(item["best_ap"]), 6),
            "best_lambda_roc": round(float(item["best_lambda_roc"]), 1),
            "best_roc_auc": round(float(item["best_roc"]), 6),
        })

    csv_path = output_dir / "gen1_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["subset", "best_lambda_ap", "best_ap", "best_lambda_roc", "best_roc_auc"])
        writer.writeheader()
        writer.writerows(rows)

    txt_path = output_dir / "gen1_summary.txt"
    lines = ["Gen 1 Summary", "================", ""]
    for sub_name in sorted(results):
        item = results[sub_name]
        lines.append(
            f"[{sub_name}] best λ(AP)={item['best_lambda_ap']:.1f}, best AP={item['best_ap']:.4f}; "
            f"best λ(ROC)={item['best_lambda_roc']:.1f}, best ROC-AUC={item['best_roc']:.4f}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[SAVE] CSV: {csv_path}")
    print(f"[SAVE] TXT: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the consolidated Gen 1 idea across all 30 subsets.")
    parser.add_argument("--family", type=str, choices=sorted(FAMILY_GROUPS.keys()), help="Run only one family group.")
    parser.add_argument("--subset", type=str, choices=list(SUBSET_REGISTRY.keys()), help="Run only one subset.")
    parser.add_argument("--lambdas", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0")
    args = parser.parse_args()

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    if not lambdas:
        lambdas = DEFAULT_LAMBDAS

    if args.subset:
        target_subsets = [args.subset]
    elif args.family:
        target_subsets = FAMILY_GROUPS[args.family]
    else:
        target_subsets = sorted(SUBSET_REGISTRY.keys())

    results = {}
    for sub_name in target_subsets:
        print(f"[RUN] {sub_name} | lambda sweep = {len(lambdas)} points")
        results[sub_name] = sweep_subset(sub_name, lambdas)

    if args.subset:
        item = results[args.subset]
        print(f"\nSubset summary for {args.subset}:")
        print(f"Best AP: {item['best_ap']:.4f} at λ={item['best_lambda_ap']:.1f}")
        print(f"Best ROC-AUC: {item['best_roc']:.4f} at λ={item['best_lambda_roc']:.1f}")
        save_results(results, RESULT_DIR)
        return

    for family_name, family_subsets in sorted(FAMILY_GROUPS.items()):
        family_hits = [s for s in family_subsets if s in results]
        if family_hits:
            print_family_summary(family_name, family_hits, results)

    save_results(results, RESULT_DIR)
    print("\nDone. Results available under: " + str(RESULT_DIR))


if __name__ == "__main__":
    main()
