"""
classifier.py — LexiDecayV2: Main Orchestrator.

Integrates all five pipeline stages:
  Stage 1: Token Engine      → TokenizedDocument
  Stage 2: RelationGraph     → single knowledge store (built/updated here)
  Stage 3: PhraseDiscovery   → phrase nodes in graph
  Stage 4: PropagationEngine → bounded PPR per category
  Stage 5: EvidenceAggregator→ 5-source evidence → final scores

Public API:
  fit(X, y)                    full training (resets graph)
  partial_fit(X, y)            incremental online learning (no reset)
  classify(text)               → PredictionResult (full explainability)
  predict(X)                   → List[str] (labels)
  predict_proba(X)             → List[Dict[str, float]]
  batch_predict(texts, n_jobs) → List[PredictionResult]
  fine_tune(text, true_label)  → None (delegates to GraphFineTuner)
  save(path) / load(path)      persistence
  get_graph_stats()            diagnostic information
"""

from __future__ import annotations

import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .relation_graph     import RelationGraph
from .token_engine       import Token, TokenEngine, TokenizedDocument
from .phrase_discovery   import PhraseDiscovery
from .propagation        import PropagationEngine, PropagationPath, PropagationResult
from .evidence           import (
    CategoryEvidence, EvidenceAggregator, EvidenceBundle, EvidenceItem,
    _DEFAULT_WEIGHTS,
)
from .feature_extractor  import StructuralFeatureExtractor

_EPS = 1e-12
_VERSION = "2.0.0"

# Evidence weights tuned for robustness: context/propagation are reduced to prevent
# phantom connections (words near spam words in the graph bleeding evidence into
# innocent ham texts). Direct + phrase evidence is kept high.
_CLASSIFIER_DEFAULT_WEIGHTS: Dict[str, float] = {
    "direct":      1.0,
    "phrase":      1.5,
    "context":     0.25,
    "propagation": 0.10,
    "interaction": 0.10,
}


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """Single feature contribution, suitable for top-N positive/negative lists."""
    token_or_edge: str
    evidence_type: str
    label:         str
    contribution:  float
    explanation:   str


@dataclass
class PredictionResult:
    """
    Complete prediction output with full explainability.
    Every field is derived exclusively from the RelationGraph.
    """
    text:             str
    predicted_label:  str
    probability:      float
    scores:           Dict[str, float]
    probabilities:    Dict[str, float]
    tokens:           List[Token]
    phrases_found:    List[str]                        # phrase node IDs detected
    evidence:         Dict[str, CategoryEvidence]      # per-category breakdowns
    activated_paths:  List[PropagationPath]            # full propagation trace
    top_positive_features: List[FeatureContribution]   # top 10 for predicted label
    top_negative_features: List[FeatureContribution]   # most negative for predicted
    explanation:      str                              # human-readable summary
    inference_time_ms: float = 0.0

    def get_score(self, label: str) -> float:
        return self.scores.get(label, 0.0)

    def get_probability(self, label: str) -> float:
        return self.probabilities.get(label, 0.0)


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_s = max(scores.values())
    exps  = {k: math.exp(v - max_s) for k, v in scores.items()}
    total = max(sum(exps.values()), _EPS)
    return {k: v / total for k, v in exps.items()}


def _build_explanation(
    predicted: str,
    probabilities: Dict[str, float],
    bundle: EvidenceBundle,
    top_pos: List[FeatureContribution],
) -> str:
    lines = [
        f"Prediction: '{predicted}'  (confidence: {probabilities.get(predicted, 0.0):.1%})",
        "",
        "Category scores:",
    ]
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        ce = bundle.per_category.get(label)
        if ce:
            lines.append(
                f"  {label:<20} {prob:.4f}  "
                f"[direct={ce.direct_score:.3f}, "
                f"phrase={ce.phrase_score:.3f}, "
                f"context={ce.context_score:.3f}, "
                f"prop={ce.propagation_score:.3f}, "
                f"interact={ce.interaction_score:.3f}]"
            )
    if top_pos:
        lines.append("")
        lines.append("Top positive features for predicted label:")
        for feat in top_pos[:5]:
            lines.append(f"  [{feat.evidence_type}] {feat.token_or_edge}  +{feat.contribution:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch-inference worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _batch_worker(payload: Tuple[bytes, List[str]]) -> List[Dict]:
    """Worker for ProcessPoolExecutor: deserialize graph, classify chunk."""
    graph_bytes, texts = payload
    model: LexiDecayV2 = pickle.loads(graph_bytes)
    return [_result_to_dict(model.classify(t)) for t in texts]


def _result_to_dict(r: PredictionResult) -> Dict:
    return {
        "predicted_label": r.predicted_label,
        "probability":     r.probability,
        "scores":          r.scores,
        "probabilities":   r.probabilities,
        "phrases_found":   r.phrases_found,
        "explanation":     r.explanation,
        "inference_time_ms": r.inference_time_ms,
    }


# ---------------------------------------------------------------------------
# LexiDecayV2
# ---------------------------------------------------------------------------

class LexiDecayV2:
    """
    LexiDecay v2 — Explainable Statistical Graph-Based Text Classifier.

    The RelationGraph is the sole knowledge store. All training, inference,
    fine-tuning, and persistence operate exclusively on that graph.

    Parameters
    ----------
    window_size          : co-occurrence window radius for graph construction
    phrase_g2_threshold  : G² cutoff for phrase acceptance (10.83 → p<0.001)
    phrase_npmi_threshold: additional NPMI filter for phrase acceptance
    phrase_min_support   : minimum co-occurrence count for phrase discovery
    max_phrase_length    : maximum n-gram length (2=bigrams, 3=trigrams)
    propagation_depth    : max PPR hops from seed nodes (1–3)
    propagation_decay    : activation decay per hop
    propagation_top_k    : max neighbors to explore per node per hop
    propagation_restart  : PPR restart probability (α)
    min_threshold        : prune propagation activations below this strength
    evidence_weights     : dict of weights for 5 evidence sources
    phrase_boost         : multiplier on phrase evidence (default 1.5)
    pruning_min_weight   : minimum edge weight retained after pruning
    pruning_min_cooc     : minimum co-occurrence count retained after pruning
    pruning_min_doc_freq : minimum document frequency retained after pruning
    pmi_smoothing        : additive smoothing for PMI/NPMI computation
    uncertainty_lambda   : smoothing for node uncertainty (default 5.0)
    use_structural_features : add regex-based pseudo-tokens (__LONG_NUM__, __CAPS_WORD__,
                              __SHORT_CODE__, __MONEY__, __URL__) before tokenization.
                              Dramatically improves precision on real-world noisy text
                              (e.g. SMS spam: __LONG_NUM__ is 703x more frequent in spam).
    add_class_prior      : add log P(c) to raw scores before softmax, correcting for
                           class imbalance. Useful when training data is skewed (e.g.
                           85% ham / 15% spam). Default False (caller can use
                           calibrate_threshold() instead for binary tasks).
    """

    def __init__(
        self,
        window_size:              int   = 5,
        phrase_g2_threshold:      float = 10.83,
        phrase_npmi_threshold:    float = 0.3,
        phrase_min_support:       int   = 3,
        max_phrase_length:        int   = 2,
        propagation_depth:        int   = 1,
        propagation_decay:        float = 0.85,
        propagation_top_k:        int   = 10,
        propagation_restart:      float = 0.15,
        min_threshold:            float = 0.005,
        evidence_weights:         Optional[Dict[str, float]] = None,
        phrase_boost:             float = 1.5,
        pruning_min_weight:       float = 0.01,
        pruning_min_cooc:         float = 2.0,
        pruning_min_doc_freq:     int   = 2,
        pmi_smoothing:            float = _EPS,
        uncertainty_lambda:       float = 5.0,
        use_structural_features:  bool  = True,
        structural_presence_repeat: int = 1,
        structural_absence_repeat:  int = 1,
        use_bigrams:              bool  = True,
        add_class_prior:          bool  = False,
        propagation_min_seed_disc: float = 0.0,
    ):
        self.graph = RelationGraph(
            window_size=window_size,
            pmi_smoothing=pmi_smoothing,
            uncertainty_lambda=uncertainty_lambda,
        )
        feat_ext = (
            StructuralFeatureExtractor(
                presence_repeat=structural_presence_repeat,
                absence_repeat=structural_absence_repeat,
                use_bigrams=use_bigrams,
            )
            if use_structural_features else None
        )
        self.token_engine = TokenEngine(window_size=window_size, feature_extractor=feat_ext)
        self.phrase_discovery = PhraseDiscovery(
            g2_threshold=phrase_g2_threshold,
            npmi_threshold=phrase_npmi_threshold,
            min_support=phrase_min_support,
            max_phrase_length=max_phrase_length,
        )
        self.propagation_engine = PropagationEngine(
            max_depth=propagation_depth,
            decay=propagation_decay,
            top_k=propagation_top_k,
            restart_alpha=propagation_restart,
            min_threshold=min_threshold,
            min_seed_disc=propagation_min_seed_disc,
        )
        # Use classifier-specific default weights (reduced context/propagation to
        # prevent phantom connections bleeding into innocent ham texts)
        _eff_weights = evidence_weights if evidence_weights is not None else dict(_CLASSIFIER_DEFAULT_WEIGHTS)
        self.aggregator = EvidenceAggregator(
            evidence_weights=_eff_weights,
            phrase_boost=phrase_boost,
        )

        # Store config for save/load and hyperparameter access
        self._config: Dict[str, Any] = {
            "window_size":              window_size,
            "phrase_g2_threshold":      phrase_g2_threshold,
            "phrase_npmi_threshold":    phrase_npmi_threshold,
            "phrase_min_support":       phrase_min_support,
            "max_phrase_length":        max_phrase_length,
            "propagation_depth":        propagation_depth,
            "propagation_decay":        propagation_decay,
            "propagation_top_k":        propagation_top_k,
            "propagation_restart":      propagation_restart,
            "min_threshold":            min_threshold,
            "evidence_weights":         _eff_weights,
            "phrase_boost":             phrase_boost,
            "pruning_min_weight":       pruning_min_weight,
            "pruning_min_cooc":         pruning_min_cooc,
            "pruning_min_doc_freq":     pruning_min_doc_freq,
            "pmi_smoothing":            pmi_smoothing,
            "uncertainty_lambda":       uncertainty_lambda,
            "use_structural_features":       use_structural_features,
            "structural_presence_repeat":    structural_presence_repeat,
            "structural_absence_repeat":     structural_absence_repeat,
            "use_bigrams":                   use_bigrams,
            "add_class_prior":               add_class_prior,
            "propagation_min_seed_disc":     propagation_min_seed_disc,
        }
        self._pruning_min_weight   = pruning_min_weight
        self._pruning_min_cooc     = pruning_min_cooc
        self._pruning_min_doc_freq = pruning_min_doc_freq
        self._thresholds: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: List[str], y: List[str]) -> "LexiDecayV2":
        """
        Full training. Resets the graph, processes all documents,
        discovers phrases, recomputes weights, and prunes.

        Parameters
        ----------
        X : list of raw text documents
        y : list of category labels (same length as X)
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have equal length ({len(X)} vs {len(y)})")
        if not X:
            raise ValueError("Training set is empty.")

        self.graph.reset()

        # Pass 1: tokenize + update graph (raw counts only, no weight computation)
        for text, label in zip(X, y):
            doc = self.token_engine.tokenize(text)
            self.graph.update(doc.token_texts, label)

        # Pass 2: phrase discovery (reads raw co-occurrence from graph edges)
        self.phrase_discovery.discover(self.graph)

        # Pass 3: compute all derived statistics once
        self.graph.recompute_weights()

        # Pass 4: prune weak edges
        self.graph.prune(
            min_weight=self._pruning_min_weight,
            min_cooc=self._pruning_min_cooc,
            min_doc_freq=self._pruning_min_doc_freq,
        )

        return self

    def partial_fit(self, X: List[str], y: List[str]) -> "LexiDecayV2":
        """
        Incremental online learning. Adds new documents WITHOUT resetting
        the graph. Only newly affected edges/nodes are recomputed.

        Safe to call multiple times in sequence.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have equal length ({len(X)} vs {len(y)})")

        dirty_edges: set = set()
        dirty_nodes: set = set()

        for text, label in zip(X, y):
            doc = self.token_engine.tokenize(text)
            new_e, new_n = self.graph.update(
                doc.token_texts, label, return_dirty=True
            )
            dirty_edges.update(new_e)
            dirty_nodes.update(new_n)

        # Incremental phrase discovery
        self.phrase_discovery.discover(self.graph, incremental=True)

        # Recompute only dirty edges and nodes
        self.graph.recompute_weights(
            only_edges=dirty_edges,
            only_nodes=dirty_nodes,
        )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, text: str) -> PredictionResult:
        """
        Full prediction with complete explainability.

        Returns PredictionResult containing:
          - predicted_label, probability, scores, probabilities
          - all 5 evidence types per category
          - propagation paths
          - top positive and negative features
          - human-readable explanation
        """
        self._ensure_ready()
        t0 = time.perf_counter()

        # Stage 1: Tokenize
        doc = self.token_engine.tokenize(text)

        if not doc.tokens:
            return self._empty_result(text, t0)

        # Stage 3: Detect phrases in input
        phrases = self.phrase_discovery.find_phrases(doc.token_texts, self.graph)
        all_input_tokens = doc.token_texts + [p[0] for p in phrases]

        # Stage 4: Propagation for each label
        prop_results: Dict[str, PropagationResult] = {}
        for label in self.graph.labels:
            prop_results[label] = self.propagation_engine.propagate(
                all_input_tokens, label, self.graph
            )

        # Stage 5: Evidence aggregation
        bundle = self.aggregator.aggregate(
            tokens=doc.tokens,
            graph=self.graph,
            prop_results=prop_results,
            phrases=phrases,
            window_size=self.graph.window_size,
        )

        # Scores and probabilities
        raw_scores = {label: ce.total_score for label, ce in bundle.per_category.items()}

        # Optional class-prior correction: add log P(c) to each category score.
        # Corrects for label imbalance (e.g. 85% ham / 15% spam) so softmax
        # properly weights the base rates.
        if self._config.get("add_class_prior", False):
            N = max(self.graph.total_docs, 1)
            for lbl in raw_scores:
                n_c = max(self.graph.total_docs_per_label.get(lbl, 1), 1)
                raw_scores[lbl] += math.log(n_c / N)

        probabilities = _softmax(raw_scores)
        predicted     = max(probabilities, key=probabilities.get)

        # Build feature lists from all evidence items
        all_items = [
            FeatureContribution(
                token_or_edge=item.source,
                evidence_type=item.evidence_type,
                label=item.label,
                contribution=item.contribution,
                explanation=item.explanation,
            )
            for ce in bundle.per_category.values()
            for item in ce.items
        ]
        pred_items = [f for f in all_items if f.label == predicted]
        top_pos = sorted([f for f in pred_items if f.contribution > 0],
                         key=lambda x: -x.contribution)[:10]
        top_neg = sorted([f for f in pred_items if f.contribution < 0],
                         key=lambda x: x.contribution)[:10]

        all_paths = [
            path
            for pr in prop_results.values()
            for path in pr.paths
        ]

        explanation = _build_explanation(predicted, probabilities, bundle, top_pos)

        return PredictionResult(
            text=text,
            predicted_label=predicted,
            probability=probabilities.get(predicted, 0.0),
            scores=raw_scores,
            probabilities=probabilities,
            tokens=doc.tokens,
            phrases_found=[p[0] for p in phrases],
            evidence=bundle.per_category,
            activated_paths=all_paths,
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            explanation=explanation,
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def predict(self, X: List[str]) -> List[str]:
        """
        Batch label prediction.

        If calibrate_threshold() has been called for a label, uses the stored
        threshold to decide the boundary rather than argmax(softmax).
        This is critical for imbalanced binary tasks (e.g. 85% ham / 15% spam).
        """
        results = [self.classify(text) for text in X]
        if not self._thresholds:
            return [r.predicted_label for r in results]

        preds: List[str] = []
        for r in results:
            label = r.predicted_label
            for pos_label, threshold in self._thresholds.items():
                prob = r.probabilities.get(pos_label, 0.0)
                neg_labels = {k: v for k, v in r.probabilities.items() if k != pos_label}
                if not neg_labels:
                    break
                best_neg = max(neg_labels, key=neg_labels.get)
                label = pos_label if prob > threshold else best_neg
                break  # only one threshold per predict call
            preds.append(label)
        return preds

    def calibrate_threshold(
        self,
        X: List[str],
        y: List[str],
        positive_label: str = "spam",
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find the optimal classification threshold for a binary task.

        Scans 200 threshold candidates in (0.005, 0.995) and picks the one
        that maximises F1 of positive_label on the provided (X, y) split.
        Stores the result in self._thresholds[positive_label] so that
        subsequent calls to predict() and batch_predict() use it automatically.

        Parameters
        ----------
        X              : texts to score (typically training set)
        y              : true labels
        positive_label : the label to optimise (default 'spam')
        metric         : currently only 'f1' is supported

        Returns
        -------
        (best_threshold, best_f1_score)
        """
        other_labels = [l for l in self.graph.labels if l != positive_label]
        if len(other_labels) != 1:
            raise ValueError(
                "calibrate_threshold() requires exactly 2 labels "
                f"(got {self.graph.labels})"
            )
        neg_label = other_labels[0]

        probas = self.predict_proba(X)
        scores = [p.get(positive_label, 0.0) for p in probas]

        best_t, best_f1 = 0.5, 0.0
        for t_i in range(1, 200):
            t = t_i / 200.0
            preds = [positive_label if s > t else neg_label for s in scores]
            tp = sum(1 for yt, yp in zip(y, preds) if yt == yp == positive_label)
            fp = sum(1 for yt, yp in zip(y, preds) if yt != positive_label and yp == positive_label)
            fn = sum(1 for yt, yp in zip(y, preds) if yt == positive_label and yp != positive_label)
            p_ = tp / (tp + fp + _EPS)
            r_ = tp / (tp + fn + _EPS)
            f1 = 2.0 * p_ * r_ / (p_ + r_ + _EPS)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        self._thresholds[positive_label] = best_t
        return best_t, best_f1

    def predict_proba(self, X: List[str]) -> List[Dict[str, float]]:
        """Batch probability prediction. Returns list of {label: probability} dicts."""
        return [self.classify(text).probabilities for text in X]

    def batch_predict(
        self,
        texts: List[str],
        n_jobs: int = 1,
        chunk_size: int = 50,
    ) -> List[PredictionResult]:
        """
        Parallel batch prediction using ProcessPoolExecutor.
        n_jobs=1 uses the main process (no serialization overhead).
        n_jobs>1 distributes work across processes.
        """
        if n_jobs == 1 or len(texts) <= chunk_size:
            return [self.classify(t) for t in texts]

        # Serialize this model for worker processes
        model_bytes = pickle.dumps(self)
        chunks = [
            (model_bytes, texts[i: i + chunk_size])
            for i in range(0, len(texts), chunk_size)
        ]

        results_flat: List[PredictionResult] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_batch_worker, c): idx for idx, c in enumerate(chunks)}
            ordered: Dict[int, List] = {}
            for future in as_completed(futures):
                idx = futures[future]
                ordered[idx] = future.result()

        for idx in sorted(ordered.keys()):
            for d in ordered[idx]:
                # Reconstruct lightweight PredictionResult from dict
                pr = PredictionResult(
                    text=texts[0],  # placeholder; workers return dicts
                    predicted_label=d["predicted_label"],
                    probability=d["probability"],
                    scores=d["scores"],
                    probabilities=d["probabilities"],
                    tokens=[],
                    phrases_found=d["phrases_found"],
                    evidence={},
                    activated_paths=[],
                    top_positive_features=[],
                    top_negative_features=[],
                    explanation=d["explanation"],
                    inference_time_ms=d["inference_time_ms"],
                )
                results_flat.append(pr)
        return results_flat

    # ------------------------------------------------------------------
    # Fine-tuning (delegates to GraphFineTuner)
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        text: str,
        true_label: str,
        lr: float = 0.1,
    ) -> Optional["PredictionResult"]:
        """
        Structured perceptron update on the graph.
        Modifies edge weights and node discriminativeness in-place.
        Returns the result before fine-tuning (for inspection).
        No secondary model, no rule engine — only graph modification.
        """
        # Import here to avoid circular import
        from .fine_tuner import GraphFineTuner
        result = self.classify(text)
        if result.predicted_label != true_label:
            GraphFineTuner(self.graph, lr=lr).update(result, true_label)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the complete model (graph + config + thresholds) to pickle.
        Only self.graph, self._config, and self._thresholds are stored.
        """
        payload = {
            "version":    _VERSION,
            "graph":      self.graph,
            "config":     self._config,
            "thresholds": self._thresholds,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "LexiDecayV2":
        """Load a model previously saved with save(). Returns LexiDecayV2 instance."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        config = payload.get("config", {})
        model  = cls(**config)
        model.graph = payload["graph"]
        model._thresholds = payload.get("thresholds", {})
        return model

    # ------------------------------------------------------------------
    # Graph access and diagnostics
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> Dict:
        """Return graph diagnostics for logging and visualization."""
        return self.graph.get_stats()

    @property
    def labels(self) -> List[str]:
        return list(self.graph.labels)

    @property
    def n_nodes(self) -> int:
        return len(self.graph.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.graph.edges)

    def prune_graph(self) -> int:
        """Manually trigger graph pruning. Returns number of edges removed."""
        n = self.graph.prune(
            min_weight=self._pruning_min_weight,
            min_cooc=self._pruning_min_cooc,
            min_doc_freq=self._pruning_min_doc_freq,
        )
        return n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if not self.graph.labels:
            raise ValueError(
                "Model has no categories. Call fit() or partial_fit() before classify()."
            )
        if self.graph._weights_dirty:
            self.graph.recompute_weights()

    def _empty_result(self, text: str, t0: float) -> PredictionResult:
        """Return a uniform-probability result for empty input."""
        labels = self.graph.labels
        n = max(len(labels), 1)
        uniform = {l: 1.0 / n for l in labels}
        predicted = labels[0] if labels else ""
        return PredictionResult(
            text=text,
            predicted_label=predicted,
            probability=1.0 / n,
            scores={l: 0.0 for l in labels},
            probabilities=uniform,
            tokens=[],
            phrases_found=[],
            evidence={},
            activated_paths=[],
            top_positive_features=[],
            top_negative_features=[],
            explanation="Empty input — uniform probabilities returned.",
            inference_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def __repr__(self) -> str:
        return (
            f"LexiDecayV2(labels={self.labels}, "
            f"nodes={self.n_nodes}, edges={self.n_edges})"
        )
