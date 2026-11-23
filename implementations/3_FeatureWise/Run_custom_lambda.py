"""Run 3WNCROD with a user-supplied feature-wise lambda vector.

Example:
    python implementations/3_FeatureWise/Run_custom_lambda.py A1 --lambdas 1,0.8,1.2,1,1,1

The feature-wise lambda vector must contain one positive value per condition
attribute in the selected dataset. Scores and a metric summary are written to
``implementations/3_FeatureWise/custom_results``.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions
from wncrod_featurewise import WNCROD_featurewise


def parse_lambda_vector(value: str) -> np.ndarray:
    try:
        values = np.asarray([float(part.strip()) for part in value.split(",")])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "lambdas must be comma-separated numbers, for example 1,0.8,1.2"
        ) from exc
    if values.size == 0 or not np.isfinite(values).all() or not (values > 0).all():
        raise argparse.ArgumentTypeError("every lambda must be finite and greater than zero")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 3WNCROD once with a user-provided feature-wise lambda vector."
    )
    parser.add_argument(
        "dataset",
        type=str.upper,
        choices=sorted(SUBSET_REGISTRY),
        metavar="DATASET",
        help="Benchmark subset name, for example A1 or TH2.",
    )
    parser.add_argument(
        "--lambdas",
        required=True,
        type=parse_lambda_vector,
        metavar="L1,L2,...",
        help="Comma-separated lambda value for every condition attribute, in column order.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _, y, X_tem = load_subset(args.dataset, validate=False)
    mat_file = SUBSET_REGISTRY[args.dataset]["mat_file"]
    mat_path = PROJECT_ROOT / "baseline" / "datasets" / mat_file
    data = scipy.io.loadmat(str(mat_path))["trandata"][:, :-1].astype(float)

    if args.lambdas.size != data.shape[1]:
        parser.error(
            f"{args.dataset} has {data.shape[1]} condition attributes; "
            f"received {args.lambdas.size} lambda value(s)."
        )

    start = time.perf_counter()
    scores = WNCROD_featurewise(data, X_tem, args.lambdas)
    elapsed = time.perf_counter() - start
    metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)

    output_dir = Path(__file__).resolve().parent / "custom_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_stem = f"{args.dataset}_{timestamp}"
    scores_path = output_dir / f"{output_stem}_scores.csv"
    summary_path = output_dir / f"{output_stem}_summary.json"

    pd.DataFrame(
        {
            "dataset_row_index": X_tem,
            "true_label": y[X_tem],
            "anomaly_score": scores,
        }
    ).to_csv(scores_path, index=False)

    summary = {
        "dataset": args.dataset,
        "lambda_vector": args.lambdas.tolist(),
        "condition_attribute_count": int(data.shape[1]),
        "selected_object_count": int(X_tem.size),
        "average_precision": metrics["ap"],
        "roc_auc": metrics["auroc"],
        "runtime_seconds": elapsed,
        "scores_file": str(scores_path),
    }
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=False)

    print(f"Dataset: {args.dataset}")
    print(f"Lambda vector: {args.lambdas.tolist()}")
    print(f"AP: {metrics['ap']:.4f} | ROC-AUC: {metrics['auroc']:.4f}")
    print(f"Runtime: {elapsed:.4f} seconds")
    print(f"Scores: {scores_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
