"""
fine_tuner.py — Graph-Level Structured Perceptron Fine-Tuning.

Loss function: multiclass hinge (structured perceptron variant)
  L(D, y*) = max(0, score(D, ŷ) − score(D, y*) + margin)

Gradient w.r.t. edge weight w(u,v,c):
  ∂L/∂w(u,v,ŷ) = +contrib_edge(u,v,ŷ,D)    [penalize: reduced wrong label]
  ∂L/∂w(u,v,y*)= −contrib_edge(u,v,y*,D)    [reward: increase true label]

Update rules (only on wrong predictions):
  w(u,v,ŷ) ← w(u,v,ŷ) − lr · |contrib| · error_magnitude
  w(u,v,y*)← w(u,v,y*) + lr · |contrib| · error_magnitude
  disc(v,ŷ) ← disc(v,ŷ) − lr · |contrib| · error_magnitude   [direct/phrase/prop]
  disc(v,y*)← disc(v,y*) + lr · |contrib| · error_magnitude

error_magnitude = prob(ŷ|D) − prob(y*|D)   ∈ (0,1)

INVARIANT: Only edges/nodes present in the inference evidence trace
are ever modified. No new data structures are created.
The graph remains the sole knowledge store after fine-tuning.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .relation_graph import RelationGraph
from .classifier import PredictionResult

_EPS = 1e-12


class GraphFineTuner:
    """
    Structured perceptron update on RelationGraph edge weights
    and node discriminativeness scores.

    Parameters
    ----------
    graph : the RelationGraph to update in-place
    lr    : learning rate (step size for weight updates)
    margin: hinge margin (default 1.0 for structured perceptron)
    """

    def __init__(
        self,
        graph: RelationGraph,
        lr:     float = 0.1,
        margin: float = 1.0,
    ):
        self.graph  = graph
        self.lr     = lr
        self.margin = margin

    def update(
        self,
        result:     PredictionResult,
        true_label: str,
    ) -> None:
        """
        Apply one structured perceptron step given a wrong prediction.

        Parameters
        ----------
        result     : PredictionResult from LexiDecayV2.classify()
        true_label : the correct label for this document
        """
        predicted = result.predicted_label
        if predicted == true_label:
            return  # correct prediction — no update needed

        if true_label not in self.graph.labels:
            return  # unknown label — cannot update

        # Error magnitude: probability gap between wrong and correct label
        error_magnitude = (
            result.probability
            - result.probabilities.get(true_label, 0.0)
        )
        if error_magnitude <= 0:
            return

        # --- Penalize evidence that supported wrong prediction ---
        predicted_evidence = result.evidence.get(predicted)
        if predicted_evidence:
            self._adjust_evidence(
                evidence_items=predicted_evidence.items,
                label=predicted,
                direction=-1,   # reduce wrong-label support
                error_magnitude=error_magnitude,
            )

        # --- Boost evidence that should have supported true label ---
        true_evidence = result.evidence.get(true_label)
        if true_evidence:
            self._adjust_evidence(
                evidence_items=true_evidence.items,
                label=true_label,
                direction=+1,   # increase true-label support
                error_magnitude=error_magnitude,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjust_evidence(
        self,
        evidence_items,
        label:           str,
        direction:       int,   # +1 or -1
        error_magnitude: float,
    ) -> None:
        """Apply weight delta to all evidence items for `label`."""
        for item in evidence_items:
            delta = direction * self.lr * abs(item.contribution) * error_magnitude

            if item.evidence_type == "context":
                edge_key = self._parse_context_edge(item.source)
                if edge_key:
                    self.graph.update_edge_weight(edge_key, label, delta)

            elif item.evidence_type == "interaction":
                # Interaction source format: "(u)→[m]→(v)"
                # Penalize/reward both sub-edges: (u,m) and (v,m)
                sub_keys = self._parse_interaction_edges(item.source)
                for ek in sub_keys:
                    self.graph.update_edge_weight(ek, label, delta * 0.5)

            elif item.evidence_type in ("direct", "phrase", "propagation"):
                # Node-level discriminativeness update
                node_id = item.source
                self.graph.update_node_discriminativeness(node_id, label, delta)

    @staticmethod
    def _parse_context_edge(source: str) -> Optional[Tuple[str, str]]:
        """
        Parse context evidence source "(u,v)" → canonical edge key (min,max).
        Returns None if parsing fails.
        """
        m = re.match(r"^\((.+?),(.+?)\)$", source.strip())
        if not m:
            return None
        a, b = m.group(1).strip(), m.group(2).strip()
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _parse_interaction_edges(source: str) -> list:
        """
        Parse interaction source "(u)→[m]→(v)" → [(u,m), (v,m)] edge keys.
        Returns empty list if parsing fails.
        """
        m = re.match(r"^\((.+?)\)→\[(.+?)\]→\((.+?)\)$", source.strip())
        if not m:
            return []
        u, med, v = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        ek1 = (u, med) if u < med else (med, u)
        ek2 = (v, med) if v < med else (med, v)
        return [ek1, ek2]
