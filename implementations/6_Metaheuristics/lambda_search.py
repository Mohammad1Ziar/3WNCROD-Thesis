"""Core of the metaheuristic lambda-search benchmark for 3WNCROD+.

This module turns the feature-wise radius search of 3WNCROD into a *single,
fairly comparable* optimisation problem and provides one adapter per
metaheuristic.

Problem
-------
For every numerical, non-constant condition attribute ``c_j`` the radius of
Eq. (8) is ``delta_j = std(c_j) / lambda_j``.  The paper uses one global
``lambda``; the feature-wise idea uses one ``lambda_j`` per attribute.  Only
attributes with ``std > 0`` have a lambda-dependent radius, so the search space
is exactly ``[0.1, 3.0]^m_active``.

Objective
---------
``maximise`` the selected metric (AP or ROC-AUC) of the resulting ranking.
The objective is deterministic and noise-free, and one evaluation costs one
full 3WNCROD execution.

Fairness
--------
Every method shares one *evaluation budget*: the number of 3WNCROD executions.
No method is allowed to exceed it.  The budget is enforced inside the
objective, so methods that overshoot during a generation are stopped
immediately and the exact number of evaluations is reported per run.

Two decoding modes
------------------
``continuous``  the decision vector holds the lambda values directly.
``key``         the decision vector holds *indices* into the 0.1-step lambda
                grid, which is what a discrete method (ACO, BRKGA) works on;
                the objective rounds them to the nearest grid point.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluators import evaluate_predictions

# The optimised 3WNCROD kernel with a feature-wise radius (implementation 3).
_CORE_PATH = (
    PROJECT_ROOT / "implementations" / "3_FeatureWise" / "wncrod_featurewise.py"
)
_CORE_SPEC = importlib.util.spec_from_file_location("wncrod_featurewise", _CORE_PATH)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load 3WNCROD implementation from {_CORE_PATH}")
_CORE_MODULE = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules.setdefault("wncrod_featurewise", _CORE_MODULE)
_CORE_SPEC.loader.exec_module(_CORE_MODULE)
WNCROD_featurewise = _CORE_MODULE.WNCROD_featurewise

# The lambda grid of the paper's search: 0.1 ... 3.0 in steps of 0.1.
LAMBDA_GRID = np.round(np.arange(0.1, 3.05, 0.1), 2)
LAMBDA_LOW, LAMBDA_HIGH = float(LAMBDA_GRID[0]), float(LAMBDA_GRID[-1])

# Number of grid points, used by the discrete (key) decoders.
N_GRID = len(LAMBDA_GRID)


# ---------------------------------------------------------------------------
#  the objective
# ---------------------------------------------------------------------------
class LambdaObjective:
    """Counts algorithm executions and tracks the best lambda vector found.

    Parameters
    ----------
    data, X_tem, y : as returned by ``data_loaders.load_subset``
    metric_key : ``"ap"`` or ``"auroc"``
    active : indices of the attributes whose radius depends on lambda
    mode : ``"continuous"`` or ``"key"``
    """

    def __init__(self, data, X_tem, y, metric_key: str, active: np.ndarray,
                 mode: str = "continuous", subset: str = "", project: bool = True):
        self.data = np.ascontiguousarray(data, dtype=float)
        self.X_tem = X_tem
        self.y = y
        self.metric_key = metric_key
        self.subset = subset
        # ``project=True`` evaluates every candidate on the 0.1-step grid, so
        # that all methods search exactly the same (discrete) landscape the
        # thesis uses.  ``project=False`` evaluates the raw continuous lambda,
        # which measures the value of the continuous relaxation instead.
        self.project = project
        self.active = np.asarray(active, dtype=int)
        self.n_active = int(self.active.size)
        self.mode = mode
        self.n_evals = 0
        self.seconds = 0.0
        self.best_value = -np.inf
        self.best_x: Optional[np.ndarray] = None
        self.best_raw: Optional[np.ndarray] = None

    # -- decoding ----------------------------------------------------------
    def decode(self, x) -> np.ndarray:
        """Map a raw decision vector to the lambda values that are evaluated."""
        x = np.asarray(x, dtype=float).ravel()
        if self.mode == "key":
            idx = np.clip(np.rint(x), 0, N_GRID - 1).astype(int)
            return LAMBDA_GRID[idx]
        x = np.clip(x, LAMBDA_LOW, LAMBDA_HIGH)
        if self.project:
            idx = np.clip(np.rint((x - LAMBDA_LOW) / 0.1), 0, N_GRID - 1).astype(int)
            return LAMBDA_GRID[idx]
        return x

    def full_vector(self, lam_active: np.ndarray) -> np.ndarray:
        """Expand the active lambdas to the full attribute vector (inactive = 1)."""
        full = np.ones(self.data.shape[1], dtype=float)
        full[self.active] = lam_active
        return full

    # -- evaluation --------------------------------------------------------
    def __call__(self, x) -> float:
        """Evaluate one candidate; returns the metric (higher is better)."""
        lam_active = self.decode(x)
        t0 = time.perf_counter()
        scores = WNCROD_featurewise(self.data, self.X_tem, self.full_vector(lam_active))
        metrics = evaluate_predictions(y_true=self.y, scores=scores, X_tem=self.X_tem)
        self.seconds += time.perf_counter() - t0
        self.n_evals += 1
        value = float(metrics[self.metric_key])
        if value > self.best_value:
            self.best_value = value
            self.best_x = lam_active.copy()
            self.best_raw = np.asarray(x, dtype=float).ravel().copy()
        return value


# ---------------------------------------------------------------------------
#  result record
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    method: str
    subset: str
    metric: str
    m_active: int
    budget: int
    evaluations: int
    seconds: float
    best_value: float           # metric at the vector the method returned
    snapped_value: float        # metric after snapping lambda to the 0.1 grid
    best_lambda: str            # textual lambda vector (grid values, full m)
    seed: int
    status: str = "ok"
    extra: Dict[str, float] = field(default_factory=dict)


def format_lambda(vec) -> str:
    values = [f"{v:g}" for v in np.asarray(vec, dtype=float)]
    return "[" + ",".join(values) + "]"


# ---------------------------------------------------------------------------
#  helpers shared by the adapters
# ---------------------------------------------------------------------------
def active_attributes(data: np.ndarray) -> np.ndarray:
    """Indices of the numerical, non-constant attributes (lambda-dependent)."""
    m = data.shape[1]
    numerical = np.all(data <= 1.0, axis=0)
    active = np.zeros(m, dtype=bool)
    if numerical.any():
        active[numerical] = np.std(data[:, numerical], axis=0) > 0.0
    return np.flatnonzero(active)


def snap_to_grid(vec) -> np.ndarray:
    """Snap lambda values to the nearest point of the 0.1-step grid."""
    x = np.clip(np.asarray(vec, dtype=float), LAMBDA_LOW, LAMBDA_HIGH)
    idx = np.clip(np.rint((x - LAMBDA_LOW) / 0.1), 0, N_GRID - 1).astype(int)
    return LAMBDA_GRID[idx]


def _finish(obj: LambdaObjective, method: str, subset: str, metric: str,
            budget: int, seed: int, x_returned=None,
            status: str = "ok", extra: Optional[Dict] = None) -> SearchResult:
    """Build the result record, including the grid-snapped re-evaluation.

    The re-evaluation of the snapped vector is a *reporting* step and is not
    charged to the search budget, so the reported evaluation count stays
    within the budget that the method was given.
    """
    n_search = obj.n_evals
    sec_search = obj.seconds
    best_value = obj.best_value
    best_x = obj.best_x

    lam_active = obj.decode(best_x if best_x is not None
                             else (x_returned if x_returned is not None
                                   else np.ones(obj.n_active)))
    snapped = snap_to_grid(lam_active)
    if np.array_equal(snapped, lam_active):
        snapped_value = best_value
    else:
        snapped_value = obj(snapped)
    obj.n_evals = n_search
    obj.seconds = sec_search

    full = obj.full_vector(snapped)
    record = SearchResult(
        method=method,
        subset=subset,
        metric=metric,
        m_active=obj.n_active,
        budget=budget,
        evaluations=n_search,
        seconds=sec_search,
        best_value=best_value,
        snapped_value=snapped_value,
        best_lambda=format_lambda(full),
        seed=seed,
        status=status,
        extra=extra or {},
    )
    if obj.best_raw is not None:
        record.extra["raw_lambda"] = format_lambda(obj.full_vector(obj.best_raw))
    return record


# ---------------------------------------------------------------------------
#  method adapters
# ---------------------------------------------------------------------------
def _pymoo_problem(obj: LambdaObjective):
    from pymoo.core.problem import ElementwiseProblem

    xl, xu = (0.0, N_GRID - 1.0) if obj.mode == "key" else (LAMBDA_LOW, LAMBDA_HIGH)

    class _Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=obj.n_active, n_obj=1, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.array([-obj(x)])

    return _Problem()


def _run_pymoo(obj: LambdaObjective, algorithm, budget: int, seed: int,
               pop_size: int) -> tuple:
    """Run a pymoo algorithm under a hard evaluation budget."""
    from pymoo.optimize import minimize
    from pymoo.termination.max_eval import MaximumFunctionCallTermination

    problem = _pymoo_problem(obj)
    # pymoo checks the termination between generations, so the budget is
    # lowered by one population to guarantee that the cap is never exceeded.
    term = MaximumFunctionCallTermination(n_max_evals=max(1, budget - pop_size + 1))
    res = minimize(problem, algorithm, term, seed=seed, verbose=False)
    return res, problem


def run_greedy_cd(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Greedy coordinate ascent -- the method already used by the thesis.

    One pass over the active attributes; for every attribute all grid values
    are tried while the other lambdas stay fixed.  Stops at the budget.
    """
    rng = np.random.default_rng(seed)
    lam = np.ones(obj.n_active)
    best = obj(lam)

    order = rng.permutation(obj.n_active)
    for j in order:
        if obj.n_evals >= budget:
            break
        current = lam[j]
        best_c, best_v = current, best
        for c in LAMBDA_GRID:
            if obj.n_evals >= budget:
                break
            if c == current:
                continue
            trial = lam.copy()
            trial[j] = c
            value = obj(trial)
            if value > best_v:
                best_v, best_c = value, c
            if value >= 1.0:
                break
        lam[j] = best_c
        best = best_v
    return _finish(obj, "GreedyCD", obj.subset, obj.metric_key, budget, seed)


def run_random_search(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Uniform random sampling of the continuous lambda box."""
    rng = np.random.default_rng(seed)
    while obj.n_evals < budget:
        obj(rng.uniform(LAMBDA_LOW, LAMBDA_HIGH, obj.n_active))
    return _finish(obj, "RandomSearch", obj.subset, obj.metric_key, budget, seed)


def run_nelder_mead(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """SciPy Nelder-Mead -- a purely local simplex method (reference)."""
    from scipy.optimize import minimize as scipy_minimize

    x0 = np.ones(obj.n_active)

    def fun(x):
        if obj.n_evals >= budget:
            return obj.best_value - obj(x) + 1e3
        return -obj(x)

    res = scipy_minimize(fun, x0, method="Nelder-Mead",
                         options={"maxfev": budget, "fatol": 1e-10, "xatol": 1e-8})
    return _finish(obj, "NelderMead", obj.subset, obj.metric_key, budget, seed,
                   x_returned=res.x)


def run_sa(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Simulated annealing (continuous neighbourhood, geometric cooling)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(LAMBDA_LOW, LAMBDA_HIGH, obj.n_active)
    fx = obj(x)
    best_x, best_f = x.copy(), fx
    t_start, t_end = 1.0, 0.01
    for k in range(max(1, budget - 1)):
        temp = t_start * (t_end / t_start) ** (k / max(1, budget - 1))
        cand = np.clip(x + rng.normal(0.0, max(0.05, temp), obj.n_active),
                       LAMBDA_LOW, LAMBDA_HIGH)
        fc = obj(cand)
        if fc > fx or rng.random() < np.exp((fc - fx) / max(temp, 1e-12)):
            x, fx = cand, fc
            if fx > best_f:
                best_x, best_f = x.copy(), fx
    return _finish(obj, "SimulatedAnnealing", obj.subset, obj.metric_key, budget, seed,
                   x_returned=best_x)


def run_aco(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Ant colony optimisation over the *discrete* 0.1-step lambda grid.

    The decision variables are indices into ``LAMBDA_GRID``; the pheromone
    table has one row per active attribute and one column per grid value.
    """
    rng = np.random.default_rng(seed)
    key_obj = LambdaObjective(obj.data, obj.X_tem, obj.y, obj.metric_key,
                              obj.active, mode="key", subset=obj.subset)
    m, k = key_obj.n_active, N_GRID
    tau = np.ones((m, k))
    alpha, beta, rho = 1.0, 1.0, 0.3
    decay = max(0.05, 1.0 - rho / max(budget, 1))
    rows = np.arange(m)

    while key_obj.n_evals < budget:
        # one ant = one solution = one evaluation
        probs = tau ** alpha
        picks = np.empty(m, dtype=int)
        for j in range(m):
            p = probs[j] / probs[j].sum()
            picks[j] = rng.choice(k, p=p)
        idx = np.clip(picks, 0, k - 1)

        previous_best = key_obj.best_value
        if not np.isfinite(previous_best):
            previous_best = -1.0            # AP and ROC-AUC live in [0, 1]
        value = key_obj(idx.astype(float))

        # reinforce the pheromone of this ant according to its quality
        gain = max(0.0, value - previous_best)
        tau[rows, idx] += alpha * gain / max(1.0, m)
        tau = np.maximum(tau * decay, 1e-6)

    # transfer the state of the key objective back to the continuous one
    obj.n_evals = key_obj.n_evals
    obj.seconds = key_obj.seconds
    obj.best_value = key_obj.best_value
    obj.best_x = key_obj.best_x
    return _finish(obj, "AntColony", obj.subset, obj.metric_key, budget, seed)


def run_cmaes(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """CMA-ES (pymoo) -- covariance-matrix adapting evolution strategy."""
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

    pop = max(8, 4 + int(3 * np.log(max(obj.n_active, 2))))
    algorithm = CMAES(pop_size=pop, sigma=0.5, restarts=0)
    _run_pymoo(obj, algorithm, budget, seed, pop)
    return _finish(obj, "CMA-ES", obj.subset, obj.metric_key, budget, seed)


def run_cmaes_notol(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """CMA-ES without its own convergence test.

    On a *projected* (piecewise constant) objective the default convergence
    test of CMA-ES (``tolfun``) fires after a few flat generations and the run
    stops long before the budget is used.  Disabling it lets CMA-ES use the
    whole budget, which is the fair comparison against the other methods.
    """
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

    pop = max(8, 4 + int(3 * np.log(max(obj.n_active, 2))))
    algorithm = CMAES(pop_size=pop, sigma=0.5, restarts=0, tolfun=None, tolx=None)
    _run_pymoo(obj, algorithm, budget, seed, pop)
    return _finish(obj, "CMA-ES (no tol)", obj.subset, obj.metric_key, budget, seed)


def run_de(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Differential evolution (pymoo, DE/rand/1/bin)."""
    from pymoo.algorithms.soo.nonconvex.de import DE

    pop = max(10, 5 * obj.n_active)
    algorithm = DE(pop_size=pop, variant="DE/rand/1/bin", F=0.7, CR=0.9)
    _run_pymoo(obj, algorithm, budget, seed, pop)
    return _finish(obj, "DE", obj.subset, obj.metric_key, budget, seed)


def run_pso(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Particle swarm optimisation (pymoo)."""
    from pymoo.algorithms.soo.nonconvex.pso import PSO

    pop = max(10, 5 * obj.n_active)
    algorithm = PSO(pop_size=pop, w=0.9, c1=2.0, c2=2.0, adaptive=True)
    _run_pymoo(obj, algorithm, budget, seed, pop)
    return _finish(obj, "PSO", obj.subset, obj.metric_key, budget, seed)


def run_ga(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Genetic algorithm with SBX crossover and polynomial mutation (pymoo)."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    pop = max(20, 4 * obj.n_active)
    algorithm = GA(pop_size=pop, crossover=SBX(prob=0.9, eta=15),
                   mutation=PM(prob=0.8, eta=20), eliminate_duplicates=True)
    _run_pymoo(obj, algorithm, budget, seed, pop)
    return _finish(obj, "GA", obj.subset, obj.metric_key, budget, seed)


def run_brkga(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """BRKGA -- biased random-key genetic algorithm on the discrete grid.

    The decision vector holds random keys in ``[0, N_GRID - 1]``; the
    objective rounds them to grid indices.
    """
    from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

    key_obj = LambdaObjective(obj.data, obj.X_tem, obj.y, obj.metric_key,
                              obj.active, mode="key")
    n_elites = max(6, min(20, budget // 10))
    n_offsprings = max(10, min(40, budget // 5))
    n_mutants = max(4, min(10, budget // 20))
    algorithm = BRKGA(n_elites=n_elites, n_offsprings=n_offsprings,
                      n_mutants=n_mutants, bias=0.7)
    _run_pymoo(key_obj, algorithm, budget, seed, n_elites + n_offsprings + n_mutants)
    # transfer the state of the key objective back to the continuous one
    obj.n_evals = key_obj.n_evals
    obj.seconds = key_obj.seconds
    obj.best_value = key_obj.best_value
    obj.best_x = key_obj.best_x
    return _finish(obj, "BRKGA", obj.subset, obj.metric_key, budget, seed)


def run_optuna_tpe(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Bayesian optimisation with Optuna's TPE sampler (the strategy the
    idea document ranks first)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def trial_fn(trial):
        x = np.array([trial.suggest_float(f"l{j}", LAMBDA_LOW, LAMBDA_HIGH)
                      for j in range(obj.n_active)])
        return obj(x)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_fn, n_trials=budget)
    return _finish(obj, "OptunaTPE", obj.subset, obj.metric_key, budget, seed)


def _coordinate_polish(obj: LambdaObjective, lam: np.ndarray, best: float,
                       budget: int, seed: int) -> tuple:
    """Exact grid coordinate ascent -- the polish stage of the hybrid.

    Returns the polished lambda vector, its value and the number of
    evaluations it used.
    """
    lam = np.asarray(lam, dtype=float).copy()
    order = np.random.default_rng(seed).permutation(obj.n_active)
    for j in order:
        if obj.n_evals >= budget:
            break
        current = lam[j]
        best_c, best_v = current, best
        for c in LAMBDA_GRID:
            if obj.n_evals >= budget:
                break
            if c == current:
                continue
            trial = lam.copy()
            trial[j] = c
            value = obj(trial)
            if value > best_v:
                best_v, best_c = value, c
            if value >= 1.0:
                break
        lam[j] = best_c
        best = best_v
    return lam, best


def _run_hybrid(obj: LambdaObjective, budget: int, seed: int,
                global_method: str) -> SearchResult:
    """Global metaheuristic search followed by exact grid coordinate polish.

    Phase 1 explores the box with a population method (DE or CMA-ES) and
    phase 2 walks the 0.1-step grid coordinate by coordinate, starting from
    the best vector of phase 1.  This is the strategy the idea document
    ranks first ("hybrid"), with the local polish stage that the thesis
    already implements.
    """
    pop = max(10, 5 * obj.n_active) if global_method == "DE" else max(8, 4 + int(3 * np.log(max(obj.n_active, 2))))
    half = max(pop + 1, budget // 2)
    global_budget = half - pop + 1

    if global_method == "DE":
        from pymoo.algorithms.soo.nonconvex.de import DE
        algorithm = DE(pop_size=pop, variant="DE/rand/1/bin", F=0.7, CR=0.9)
    else:
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
        algorithm = CMAES(pop_size=pop, sigma=0.5, restarts=0)

    _run_pymoo(obj, algorithm, global_budget, seed, pop)

    start = obj.best_x.copy() if obj.best_x is not None else np.ones(obj.n_active)
    lam, best = _coordinate_polish(obj, start, obj.best_value, budget, seed)
    obj.best_x = lam
    obj.best_value = max(obj.best_value, best)
    label = "HybridDE_CD" if global_method == "DE" else "HybridCMAES_CD"
    return _finish(obj, label, obj.subset, obj.metric_key, budget, seed)


def run_hybrid_de(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """Differential evolution followed by exact grid coordinate ascent."""
    return _run_hybrid(obj, budget, seed, "DE")


def run_hybrid_cmaes(obj: LambdaObjective, budget: int, seed: int) -> SearchResult:
    """CMA-ES followed by exact grid coordinate ascent."""
    return _run_hybrid(obj, budget, seed, "CMA-ES")


#: All adapters, keyed by their short name.
METHODS: Dict[str, Callable[[LambdaObjective, int, int], SearchResult]] = {
    "GreedyCD": run_greedy_cd,
    "RandomSearch": run_random_search,
    "NelderMead": run_nelder_mead,
    "CMA-ES": run_cmaes,
    "DE": run_de,
    "PSO": run_pso,
    "GA": run_ga,
    "BRKGA": run_brkga,
    "AntColony": run_aco,
    "SimulatedAnnealing": run_sa,
    "OptunaTPE": run_optuna_tpe,
    "CMA-ES (no tol)": run_cmaes_notol,
    "HybridDE_CD": run_hybrid_de,
    "HybridCMAES_CD": run_hybrid_cmaes,
}

#: Methods that work on grid *indices* instead of continuous lambda values.
KEY_METHODS = {"BRKGA"}
