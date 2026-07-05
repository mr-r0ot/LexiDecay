"""
optimizer.py — Hyperparameter Optimization for LexiDecay v2.

Implements random search (with optional exhaustive grid search) over the
full hyperparameter space. Each trial is an independent fit+evaluate cycle,
making the search embarrassingly parallel.

Supported:
  random search (n_iter random samples from param_grid)
  grid search (exhaustive, use with small grids)
  stratified k-fold cross-validation per trial
  multiprocessing via ProcessPoolExecutor
  rich progress display (if rich is installed)
  hyperparameter comparison table
"""

from __future__ import annotations

import itertools
import math
import os
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evaluation import Evaluator, _stratified_kfold, _stratified_split

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    params:  Dict[str, Any]
    score:   float
    scores_per_fold: List[float] = field(default_factory=list)
    std:     float = 0.0

    def __lt__(self, other: "TrialResult") -> bool:
        return self.score < other.score


@dataclass
class OptimizationResult:
    best_params:  Dict[str, Any]
    best_score:   float
    all_trials:   List[TrialResult]
    metric:       str
    n_trials:     int
    n_folds:      int
    search_mode:  str   # "random" or "grid"


# ---------------------------------------------------------------------------
# Default hyperparameter space
# ---------------------------------------------------------------------------

DEFAULT_PARAM_SPACE: Dict[str, List[Any]] = {
    "window_size":           [3, 5, 7],
    "phrase_g2_threshold":   [6.63, 10.83, 15.0],
    "phrase_npmi_threshold": [0.2, 0.3, 0.4],
    "phrase_min_support":    [3, 5, 10],
    "max_phrase_length":     [2, 3],
    "propagation_depth":     [1, 2, 3],
    "propagation_decay":     [0.6, 0.75, 0.85, 0.95],
    "propagation_top_k":     [5, 10, 20],
    "propagation_restart":   [0.1, 0.15, 0.2],
    "min_threshold":         [0.001, 0.005, 0.01],
    "phrase_boost":          [1.0, 1.5, 2.0],
    "pruning_min_weight":    [0.005, 0.01, 0.02],
    "evidence_weights": [
        {"direct": 1.0, "phrase": 1.2, "context": 0.8, "propagation": 0.6, "interaction": 0.4},
        {"direct": 1.5, "phrase": 1.0, "context": 1.0, "propagation": 0.5, "interaction": 0.3},
        {"direct": 1.0, "phrase": 1.5, "context": 0.5, "propagation": 0.8, "interaction": 0.2},
        {"direct": 0.8, "phrase": 1.0, "context": 1.2, "propagation": 1.0, "interaction": 0.6},
    ],
}

# Smaller space for quick experiments
FAST_PARAM_SPACE: Dict[str, List[Any]] = {
    "window_size":        [3, 5],
    "propagation_depth":  [1, 2],
    "propagation_decay":  [0.75, 0.85],
    "propagation_top_k":  [5, 10],
    "phrase_min_support": [3, 5],
    "phrase_boost":       [1.0, 1.5],
}


# ---------------------------------------------------------------------------
# Trial worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _trial_worker(payload: Dict) -> Tuple[Dict[str, Any], float, List[float]]:
    """
    Execute one hyperparameter trial.
    Imported at top-level to be pickle-safe for ProcessPoolExecutor.
    """
    from .classifier import LexiDecayV2
    ev = Evaluator()

    params   = payload["params"]
    metric   = payload["metric"]
    n_folds  = payload["n_folds"]
    seed     = payload["seed"]
    X        = payload["X"]
    y        = payload["y"]

    splits   = _stratified_kfold(X, y, n_folds=n_folds, seed=seed)
    fold_scores: List[float] = []

    for train_idx, val_idx in splits:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val   = [X[i] for i in val_idx]
        y_val   = [y[i] for i in val_idx]

        if not X_train or not X_val:
            continue
        if len(set(y_train)) < 2:
            continue

        model = LexiDecayV2(**params)
        model.fit(X_train, y_train)
        ev_result = ev.evaluate(model, X_val, y_val)
        score = ev_result.get(metric, 0.0)
        if isinstance(score, (int, float)):
            fold_scores.append(float(score))

    mean_score = sum(fold_scores) / max(len(fold_scores), 1)
    return params, mean_score, fold_scores


# ---------------------------------------------------------------------------
# HyperparameterOptimizer
# ---------------------------------------------------------------------------

class HyperparameterOptimizer:
    """
    Hyperparameter optimization for LexiDecayV2.

    Supports:
      - random search (n_iter random samples from param_grid)
      - grid search (exhaustive — pass search_mode="grid")
      - stratified k-fold cross-validation per trial
      - multiprocessing (n_jobs > 1)
      - optional rich progress bars

    Parameters
    ----------
    param_space : dict mapping hyperparameter names to lists of candidate values.
                  If None, uses DEFAULT_PARAM_SPACE.
    n_iter      : number of random samples (ignored for grid search)
    n_folds     : CV folds per trial
    metric      : evaluation metric to optimize (must be a key from Evaluator.evaluate)
    search_mode : "random" or "grid"
    n_jobs      : number of parallel workers (-1 = all CPU cores)
    seed        : random seed for reproducibility
    verbose     : print progress
    """

    def __init__(
        self,
        param_space:  Optional[Dict[str, List[Any]]] = None,
        n_iter:       int  = 40,
        n_folds:      int  = 3,
        metric:       str  = "f1_macro",
        search_mode:  str  = "random",
        n_jobs:       int  = 1,
        seed:         int  = 42,
        verbose:      bool = True,
    ):
        self.param_space = param_space or dict(DEFAULT_PARAM_SPACE)
        self.n_iter      = n_iter
        self.n_folds     = n_folds
        self.metric      = metric
        self.search_mode = search_mode
        self.n_jobs      = n_jobs
        self.seed        = seed
        self.verbose     = verbose

    def optimize(
        self,
        X: List[str],
        y: List[str],
    ) -> OptimizationResult:
        """
        Run hyperparameter search.

        Parameters
        ----------
        X, y : full dataset (will be split internally via k-fold CV)

        Returns
        -------
        OptimizationResult with best_params and full trial history.
        """
        combos = self._sample_combinations()
        if self.verbose:
            print(f"[LexiDecayV2 Optimizer] Mode={self.search_mode}, "
                  f"Trials={len(combos)}, Folds={self.n_folds}, "
                  f"Metric={self.metric}, Workers={self._n_workers()}")

        payloads = [
            {
                "params": combo,
                "metric": self.metric,
                "n_folds": self.n_folds,
                "seed": self.seed,
                "X": X,
                "y": y,
            }
            for combo in combos
        ]

        results: List[TrialResult] = []

        if self._n_workers() > 1 and len(combos) > 1:
            results = self._run_parallel(payloads)
        else:
            results = self._run_sequential(payloads)

        results.sort(key=lambda r: -r.score)
        best = results[0] if results else TrialResult(params={}, score=0.0)

        if self.verbose:
            print(f"[LexiDecayV2 Optimizer] Best {self.metric}: {best.score:.4f}")
            print(f"  Best params: {best.params}")

        return OptimizationResult(
            best_params=best.params,
            best_score=best.score,
            all_trials=results,
            metric=self.metric,
            n_trials=len(results),
            n_folds=self.n_folds,
            search_mode=self.search_mode,
        )

    def comparison_table(self, result: OptimizationResult, top_n: int = 10) -> str:
        """Format top-N trials as a readable table."""
        top = result.all_trials[:top_n]
        if not top:
            return "No trials completed."

        param_names = list(top[0].params.keys())
        col_w = 12
        header = f"{'Rank':>4}  {result.metric:>10}  {'Std':>8}  " + \
                 "  ".join(f"{p[:col_w]:<{col_w}}" for p in param_names)
        lines = [header, "-" * len(header)]

        for rank, trial in enumerate(top, 1):
            row = f"{rank:>4}  {trial.score:>10.4f}  {trial.std:>8.4f}  " + \
                  "  ".join(
                      f"{str(trial.params.get(p,''))[:col_w]:<{col_w}}"
                      for p in param_names
                  )
            lines.append(row)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_combinations(self) -> List[Dict[str, Any]]:
        """Sample parameter combinations based on search_mode."""
        names  = list(self.param_space.keys())
        values = [self.param_space[n] for n in names]
        all_combos = [
            dict(zip(names, combo))
            for combo in itertools.product(*values)
        ]

        if self.search_mode == "grid":
            return all_combos

        # Random search
        rng = random.Random(self.seed)
        n   = min(self.n_iter, len(all_combos))
        return rng.sample(all_combos, k=n)

    def _n_workers(self) -> int:
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return max(self.n_jobs, 1)

    def _run_sequential(self, payloads: List[Dict]) -> List[TrialResult]:
        results = []
        for i, payload in enumerate(payloads):
            if self.verbose and (i % 5 == 0 or i == len(payloads) - 1):
                print(f"  Trial {i+1}/{len(payloads)}", end="\r", flush=True)
            params, score, fold_scores = _trial_worker(payload)
            std = _std(fold_scores)
            results.append(TrialResult(params=params, score=score,
                                       scores_per_fold=fold_scores, std=std))
        if self.verbose:
            print()
        return results

    def _run_parallel(self, payloads: List[Dict]) -> List[TrialResult]:
        results: List[TrialResult] = []
        completed = 0
        with ProcessPoolExecutor(max_workers=self._n_workers()) as executor:
            future_map = {executor.submit(_trial_worker, p): p for p in payloads}
            for future in as_completed(future_map):
                completed += 1
                if self.verbose:
                    print(f"  Completed {completed}/{len(payloads)}", end="\r", flush=True)
                try:
                    params, score, fold_scores = future.result()
                    std = _std(fold_scores)
                    results.append(TrialResult(params=params, score=score,
                                               scores_per_fold=fold_scores, std=std))
                except Exception as exc:
                    if self.verbose:
                        print(f"\n  [Warning] Trial failed: {exc}")
                    payload = future_map[future]
                    results.append(TrialResult(params=payload["params"], score=-1.0))
        if self.verbose:
            print()
        return results


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
