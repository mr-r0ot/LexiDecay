"""
evaluation.py — Complete Evaluation Suite for LexiDecay v2.

No sklearn dependency. All metrics implemented from first principles.

Supported:
  accuracy, precision, recall, F1 (macro / micro / weighted)
  confusion_matrix
  ROC curve + AUC (one-vs-rest, trapezoidal integration)
  Precision-Recall curve + AP (one-vs-rest)
  classification_report (textual)
  stratified k-fold cross-validation
  learning curve (performance vs training size)
  hyperparameter comparison table
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b > _EPS else 0.0


def _trapezoidal_auc(xs: List[float], ys: List[float]) -> float:
    """Area under curve via trapezoidal rule. xs must be monotone."""
    area = 0.0
    for i in range(len(xs) - 1):
        area += abs(xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2.0
    return area


def _stratified_split(
    X: List[str],
    y: List[str],
    test_size: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        by_label[label].append(i)

    train_idx: List[int] = []
    test_idx:  List[int] = []

    for label, indices in by_label.items():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * test_size))) if n > 1 else 0
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def _stratified_kfold(
    X: List[str],
    y: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Stratified k-fold split. Returns list of (train_indices, val_indices) tuples.
    """
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        by_label[label].append(i)

    # Assign fold IDs per stratum
    fold_assignments: List[int] = [0] * len(y)
    for label, indices in by_label.items():
        idx = list(indices)
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            fold_assignments[i] = k % n_folds

    splits = []
    for fold in range(n_folds):
        val_idx   = [i for i in range(len(y)) if fold_assignments[i] == fold]
        train_idx = [i for i in range(len(y)) if fold_assignments[i] != fold]
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Confusion matrix + per-class counts
# ---------------------------------------------------------------------------

def _build_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> List[List[int]]:
    """Returns CM[i][j] = count where true=labels[i], pred=labels[j]."""
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def _per_class_counts(
    cm: List[List[int]],
    labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """Extract TP, FP, FN, TN per class from confusion matrix."""
    n = len(labels)
    counts: Dict[str, Dict[str, int]] = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        tn = sum(cm[j][k] for j in range(n) for k in range(n)) - tp - fp - fn
        counts[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return counts


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    labels:    List[str]
    precision: Dict[str, float]
    recall:    Dict[str, float]
    f1:        Dict[str, float]
    support:   Dict[str, int]
    accuracy:  float
    macro_avg: Dict[str, float]
    micro_avg: Dict[str, float]
    weighted_avg: Dict[str, float]

    def to_string(self) -> str:
        lines = ["Classification Report", "=" * 70]
        header = f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
        lines.append(header)
        lines.append("-" * 70)
        for l in self.labels:
            lines.append(
                f"{l:<20} {self.precision.get(l,0):>10.4f} "
                f"{self.recall.get(l,0):>10.4f} {self.f1.get(l,0):>10.4f} "
                f"{self.support.get(l,0):>10}"
            )
        lines.append("-" * 70)
        for avg_name, d in [
            ("macro avg",    self.macro_avg),
            ("micro avg",    self.micro_avg),
            ("weighted avg", self.weighted_avg),
        ]:
            lines.append(
                f"{avg_name:<20} {d.get('precision',0):>10.4f} "
                f"{d.get('recall',0):>10.4f} {d.get('f1',0):>10.4f} "
                f"{sum(self.support.values()):>10}"
            )
        lines.append(f"\nAccuracy: {self.accuracy:.4f}")
        return "\n".join(lines)


@dataclass
class ROCResult:
    """ROC curves and AUC per label (one-vs-rest)."""
    fpr:      Dict[str, List[float]]   # {label: [fpr values]}
    tpr:      Dict[str, List[float]]   # {label: [tpr values]}
    auc:      Dict[str, float]
    macro_auc: float


@dataclass
class PRResult:
    """Precision-Recall curves and Average Precision per label."""
    precision: Dict[str, List[float]]
    recall:    Dict[str, List[float]]
    ap:        Dict[str, float]        # average precision per label
    macro_ap:  float


@dataclass
class CVResult:
    """Cross-validation results across k folds."""
    fold_scores:  List[Dict[str, float]]
    mean_scores:  Dict[str, float]
    std_scores:   Dict[str, float]
    n_folds:      int


@dataclass
class LearningCurveResult:
    """Model performance vs training size."""
    train_sizes:  List[int]
    train_scores: List[float]   # metric on training set
    val_scores:   List[float]   # metric on validation set


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Complete model evaluation suite.

    All methods accept a fitted LexiDecayV2 model (or any object with
    .predict() and .predict_proba() methods).
    """

    def accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        if not y_true:
            return 0.0
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    def precision_recall_f1(
        self,
        y_true: List[str],
        y_pred: List[str],
        average: str = "macro",
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 with the given averaging strategy.
        average ∈ {"macro", "micro", "weighted"}
        """
        labels  = sorted(set(y_true) | set(y_pred))
        cm      = _build_confusion_matrix(y_true, y_pred, labels)
        counts  = _per_class_counts(cm, labels)
        support = {l: sum(cm[i]) for i, l in enumerate(labels)}

        if average == "micro":
            tp_sum = sum(c["tp"] for c in counts.values())
            fp_sum = sum(c["fp"] for c in counts.values())
            fn_sum = sum(c["fn"] for c in counts.values())
            p = _safe_div(tp_sum, tp_sum + fp_sum)
            r = _safe_div(tp_sum, tp_sum + fn_sum)
            f = _safe_div(2 * p * r, p + r)
            return {"precision": p, "recall": r, "f1": f}

        per_p, per_r, per_f, weights = [], [], [], []
        for l in labels:
            c  = counts[l]
            p  = _safe_div(c["tp"], c["tp"] + c["fp"])
            r  = _safe_div(c["tp"], c["tp"] + c["fn"])
            f  = _safe_div(2 * p * r, p + r)
            per_p.append(p); per_r.append(r); per_f.append(f)
            weights.append(support[l])

        total_w = max(sum(weights), 1)
        if average == "macro":
            n = max(len(labels), 1)
            return {
                "precision": sum(per_p) / n,
                "recall":    sum(per_r) / n,
                "f1":        sum(per_f) / n,
            }
        else:  # weighted
            return {
                "precision": sum(p * w for p, w in zip(per_p, weights)) / total_w,
                "recall":    sum(r * w for r, w in zip(per_r, weights)) / total_w,
                "f1":        sum(f * w for f, w in zip(per_f, weights)) / total_w,
            }

    def confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], List[str]]:
        """Returns (confusion_matrix_2d, ordered_labels)."""
        labs = labels or sorted(set(y_true) | set(y_pred))
        return _build_confusion_matrix(y_true, y_pred, labs), labs

    def classification_report(
        self,
        y_true: List[str],
        y_pred: List[str],
    ) -> ClassificationReport:
        """Full per-class classification report with averaging rows."""
        labels  = sorted(set(y_true) | set(y_pred))
        cm      = _build_confusion_matrix(y_true, y_pred, labels)
        counts  = _per_class_counts(cm, labels)
        support = {l: sum(cm[i]) for i, l in enumerate(labels)}
        acc     = self.accuracy(y_true, y_pred)

        prec, rec, f1_d = {}, {}, {}
        for l in labels:
            c = counts[l]
            p = _safe_div(c["tp"], c["tp"] + c["fp"])
            r = _safe_div(c["tp"], c["tp"] + c["fn"])
            f = _safe_div(2 * p * r, p + r)
            prec[l] = p; rec[l] = r; f1_d[l] = f

        macro = self.precision_recall_f1(y_true, y_pred, "macro")
        micro = self.precision_recall_f1(y_true, y_pred, "micro")
        wgtd  = self.precision_recall_f1(y_true, y_pred, "weighted")

        return ClassificationReport(
            labels=labels, precision=prec, recall=rec, f1=f1_d,
            support=support, accuracy=acc,
            macro_avg=macro, micro_avg=micro, weighted_avg=wgtd,
        )

    def evaluate(self, model, X: List[str], y: List[str]) -> Dict[str, Any]:
        """
        Full evaluation suite for one test set.
        Returns dict with all scalar metrics, confusion matrix, and report.
        """
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)
        report  = self.classification_report(y, y_pred)
        cm, labs = self.confusion_matrix(y, y_pred)

        roc_res = self.roc_auc(y, y_proba, sorted(set(y)))
        pr_res  = self.pr_auc(y, y_proba, sorted(set(y)))

        return {
            "accuracy":          report.accuracy,
            "precision_macro":   report.macro_avg["precision"],
            "precision_micro":   report.micro_avg["precision"],
            "precision_weighted":report.weighted_avg["precision"],
            "recall_macro":      report.macro_avg["recall"],
            "recall_micro":      report.micro_avg["recall"],
            "recall_weighted":   report.weighted_avg["recall"],
            "f1_macro":          report.macro_avg["f1"],
            "f1_micro":          report.micro_avg["f1"],
            "f1_weighted":       report.weighted_avg["f1"],
            "roc_auc_macro":     roc_res.macro_auc,
            "roc_auc_per_label": roc_res.auc,
            "pr_ap_macro":       pr_res.macro_ap,
            "pr_ap_per_label":   pr_res.ap,
            "confusion_matrix":  cm,
            "labels":            labs,
            "classification_report": report,
            "roc":               roc_res,
            "pr":                pr_res,
        }

    # ------------------------------------------------------------------
    # ROC AUC (one-vs-rest, pure Python + trapezoidal integration)
    # ------------------------------------------------------------------

    def roc_auc(
        self,
        y_true:  List[str],
        y_proba: List[Dict[str, float]],
        labels:  Optional[List[str]] = None,
    ) -> ROCResult:
        """
        One-vs-rest ROC AUC for all labels.
        y_proba: list of {label: probability} dicts from predict_proba.
        """
        labs = labels or sorted(set(y_true))
        fpr_dict: Dict[str, List[float]] = {}
        tpr_dict: Dict[str, List[float]] = {}
        auc_dict: Dict[str, float]       = {}

        for label in labs:
            binary = [1 if yi == label else 0 for yi in y_true]
            scores = [p.get(label, 0.0) for p in y_proba]

            # Sort by score descending
            paired = sorted(zip(scores, binary), key=lambda x: -x[0])

            pos = sum(binary)
            neg = len(binary) - pos
            if pos == 0 or neg == 0:
                fpr_dict[label] = [0.0, 1.0]
                tpr_dict[label] = [0.0, 1.0]
                auc_dict[label] = 0.5
                continue

            tp = fp = 0
            fprs = [0.0]
            tprs = [0.0]
            prev_score = None

            for score, b in paired:
                # Aggregate ties before recording a point
                if score != prev_score and prev_score is not None:
                    fprs.append(fp / neg)
                    tprs.append(tp / pos)
                if b:
                    tp += 1
                else:
                    fp += 1
                prev_score = score

            fprs.append(fp / neg)
            tprs.append(tp / pos)
            fprs.append(1.0)
            tprs.append(1.0)

            fpr_dict[label] = fprs
            tpr_dict[label] = tprs
            auc_dict[label] = _trapezoidal_auc(fprs, tprs)

        macro_auc = sum(auc_dict.values()) / max(len(auc_dict), 1)
        return ROCResult(fpr=fpr_dict, tpr=tpr_dict, auc=auc_dict, macro_auc=macro_auc)

    # ------------------------------------------------------------------
    # Precision-Recall AUC (one-vs-rest)
    # ------------------------------------------------------------------

    def pr_auc(
        self,
        y_true:  List[str],
        y_proba: List[Dict[str, float]],
        labels:  Optional[List[str]] = None,
    ) -> PRResult:
        """One-vs-rest Precision-Recall curves and Average Precision."""
        labs = labels or sorted(set(y_true))
        prec_dict: Dict[str, List[float]] = {}
        rec_dict:  Dict[str, List[float]] = {}
        ap_dict:   Dict[str, float]       = {}

        for label in labs:
            binary = [1 if yi == label else 0 for yi in y_true]
            scores = [p.get(label, 0.0) for p in y_proba]

            paired = sorted(zip(scores, binary), key=lambda x: -x[0])
            pos = sum(binary)
            if pos == 0:
                prec_dict[label] = [1.0, 0.0]
                rec_dict[label]  = [0.0, 1.0]
                ap_dict[label]   = 0.0
                continue

            tp = fp = 0
            precs = []
            recs  = []
            ap    = 0.0
            prev_recall = 0.0

            for score, b in paired:
                if b:
                    tp += 1
                else:
                    fp += 1
                p = _safe_div(tp, tp + fp)
                r = _safe_div(tp, pos)
                precs.append(p)
                recs.append(r)
                # Interpolated AP: area under step function
                ap += p * (r - prev_recall)
                prev_recall = r

            prec_dict[label] = precs
            rec_dict[label]  = recs
            ap_dict[label]   = ap

        macro_ap = sum(ap_dict.values()) / max(len(ap_dict), 1)
        return PRResult(precision=prec_dict, recall=rec_dict, ap=ap_dict, macro_ap=macro_ap)

    # ------------------------------------------------------------------
    # Cross-Validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model_factory: Callable[[], Any],
        X: List[str],
        y: List[str],
        n_folds:  int = 5,
        metric:   str = "f1_macro",
        seed:     int = 42,
    ) -> CVResult:
        """
        Stratified k-fold cross-validation.

        Parameters
        ----------
        model_factory : callable returning a fresh (unfitted) LexiDecayV2
        X, y          : full dataset
        n_folds       : number of folds
        metric        : metric key from evaluate() result dict
        """
        splits = _stratified_kfold(X, y, n_folds=n_folds, seed=seed)
        fold_scores: List[Dict[str, float]] = []

        for train_idx, val_idx in splits:
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val   = [X[i] for i in val_idx]
            y_val   = [y[i] for i in val_idx]

            if not X_train or not X_val:
                continue

            m = model_factory()
            m.fit(X_train, y_train)
            ev = self.evaluate(m, X_val, y_val)

            scalar_ev = {k: v for k, v in ev.items() if isinstance(v, float)}
            fold_scores.append(scalar_ev)

        if not fold_scores:
            return CVResult(fold_scores=[], mean_scores={}, std_scores={}, n_folds=n_folds)

        all_keys = fold_scores[0].keys()
        mean_scores = {
            k: sum(s[k] for s in fold_scores) / len(fold_scores)
            for k in all_keys
        }
        std_scores = {
            k: math.sqrt(
                sum((s[k] - mean_scores[k]) ** 2 for s in fold_scores) / len(fold_scores)
            )
            for k in all_keys
        }

        return CVResult(
            fold_scores=fold_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            n_folds=n_folds,
        )

    # ------------------------------------------------------------------
    # Learning Curve
    # ------------------------------------------------------------------

    def learning_curve(
        self,
        model_factory: Callable[[], Any],
        X: List[str],
        y: List[str],
        train_sizes: Optional[List[float]] = None,
        test_size:   float = 0.2,
        metric:      str   = "f1_macro",
        seed:        int   = 42,
    ) -> LearningCurveResult:
        """
        Compute model performance as a function of training set size.
        train_sizes: list of fractions of training data to use (default [.1,.3,.5,.7,.9])
        """
        sizes = train_sizes or [0.1, 0.3, 0.5, 0.7, 0.9]
        X_train_full, X_test, y_train_full, y_test = _stratified_split(
            X, y, test_size=test_size, seed=seed
        )

        rng = random.Random(seed)
        n_full = len(X_train_full)

        train_scores_out: List[float] = []
        val_scores_out:   List[float] = []
        sizes_out:        List[int]   = []

        for frac in sizes:
            n_use = max(int(round(n_full * frac)), 2)
            idx   = list(range(n_full))
            rng.shuffle(idx)
            idx   = idx[:n_use]
            Xs    = [X_train_full[i] for i in idx]
            ys    = [y_train_full[i] for i in idx]

            # Need at least one example per class for stratification
            present = set(ys)
            if len(present) < 2:
                continue

            m = model_factory()
            m.fit(Xs, ys)

            ev_train = self.evaluate(m, Xs, ys)
            ev_val   = self.evaluate(m, X_test, y_test)

            train_scores_out.append(ev_train.get(metric, 0.0))
            val_scores_out.append(ev_val.get(metric, 0.0))
            sizes_out.append(n_use)

        return LearningCurveResult(
            train_sizes=sizes_out,
            train_scores=train_scores_out,
            val_scores=val_scores_out,
        )
