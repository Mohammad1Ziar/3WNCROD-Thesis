from pathlib import Path

from evaluators.results_store import RESULTS_ROOT, resolve_result_path


def test_results_root_is_under_evaluators():
    root = Path(RESULTS_ROOT).resolve()
    assert root.name == "results"
    assert "evaluators" in str(root)


def test_resolve_result_path_is_canonical():
    path = resolve_result_path(
        experiment_name="baseline",
        family_name="Mammography",
        subset_name="MA1",
        filename="MA1_metrics.csv",
    )
    assert path.parent.name == "Mammography"
    assert path.name == "MA1_metrics.csv"
    assert str(path).startswith(str(Path(RESULTS_ROOT).resolve()))
