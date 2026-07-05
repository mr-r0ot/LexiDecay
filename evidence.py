"""
evidence.py — Stage 5: Multi-Source Evidence Aggregation.

Five orthogonal evidence sources combine to produce a final category score.
All evidence is derived exclusively from the RelationGraph.

Score formula:
  score(D, c) = w1·E_direct + w2·E_phrase + w3·E_context
              + w4·E_propagation + w5·E_interaction

E_direct(D,c)   = Σ_v freq(v,D)·disc(v,c)·IDF(v)·uncertainty(v,c)
E_phrase(D,c)   = Σ_p disc(p,c)·IDF(p)·uncertainty(p,c)·phrase_boost
E_context(D,c)  = Σ_{(u,v)∈pairs, d≤W} NPMI(u,v,c)·conf(u,v,c)·(1/d)
E_prop(D,c)     = Σ_{v∈propagated\input} activation(v)·disc(v,c)·IDF(v)
E_interact(D,c) = Σ_{(u,v)∈pairs} Σ_{m∈N(u)∩N(v)\input}
                      w(u,m,c)·w(v,m,c)·disc(m,c)

All individual contributions are stored in EvidenceItem for complete
explainability — the user can inspect every single term.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .relation_graph import RelationGraph
from .propagation import PropagationPath, PropagationResult
from .token_engine import Token

_EPS = 1e-12

# Default evidence weights (each independently tunable as a hyperparameter)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "direct":      1.0,
    "phrase":      1.2,
    "context":     0.8,
    "propagation": 0.6,
    "interaction": 0.4,
}


# ---------------------------------------------------------------------------
# Evidence item — atomic explainability unit
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """
    One atomic contribution to a category's score.
    Each item is independently explainable and auditable.
    """
    evidence_type: str    # "direct" | "phrase" | "context" | "propagation" | "interaction"
    source:        str    # token id, "(u,v)", or "(u)→[m]→(v)"
    label:         str    # the category this contribution belongs to
    contribution:  float  # signed contribution value (positive → supports label)
    explanation:   str    # human-readable explanation string


# ---------------------------------------------------------------------------
# Per-category evidence breakdown
# ---------------------------------------------------------------------------

@dataclass
class CategoryEvidence:
    """
    Complete evidence breakdown for one (document, category) pair.
    total_score is the weighted sum of the five component scores.
    items contains every individual EvidenceItem for full auditability.
    """
    label:             str
    direct_score:      float
    phrase_score:      float
    context_score:     float
    propagation_score: float
    interaction_score: float
    total_score:       float
    items:             List[EvidenceItem] = field(default_factory=list)

    def get_top_items(self, n: int = 10, positive_only: bool = False) -> List[EvidenceItem]:
        items = [i for i in self.items if (not positive_only or i.contribution > 0)]
        return sorted(items, key=lambda x: -x.contribution)[:n]

    def get_bottom_items(self, n: int = 10) -> List[EvidenceItem]:
        return sorted(self.items, key=lambda x: x.contribution)[:n]


# ---------------------------------------------------------------------------
# Bundle returned from aggregate()
# ---------------------------------------------------------------------------

@dataclass
class EvidenceBundle:
    """
    Complete evidence for all categories for one document.
    """
    per_category:       Dict[str, CategoryEvidence]
    input_tokens:       List[Token]
    phrases_found:      List[Tuple[str, int, int]]   # (phrase_id, start, end)
    propagation_results: Dict[str, PropagationResult]


# ---------------------------------------------------------------------------
# EvidenceAggregator
# ---------------------------------------------------------------------------

class EvidenceAggregator:
    """
    Computes the five evidence types for all categories simultaneously.

    Parameters
    ----------
    evidence_weights : weights for each of the 5 evidence sources
    phrase_boost     : multiplier applied to phrase evidence (> 1 rewards
                       high-precision multi-word expressions)
    """

    def __init__(
        self,
        evidence_weights: Optional[Dict[str, float]] = None,
        phrase_boost: float = 1.5,
    ):
        self.evidence_weights = evidence_weights or dict(_DEFAULT_WEIGHTS)
        self.phrase_boost     = phrase_boost

    def aggregate(
        self,
        tokens:      List[Token],
        graph:       RelationGraph,
        prop_results: Dict[str, PropagationResult],
        phrases:     List[Tuple[str, int, int]],
        window_size: int = 5,
    ) -> EvidenceBundle:
        """
        Compute all five evidence types for every label.

        Parameters
        ----------
        tokens       : flat token list from TokenizedDocument
        graph        : the sole knowledge store
        prop_results : {label: PropagationResult} from PropagationEngine
        phrases      : list of (phrase_id, start, end) from PhraseDiscovery
        window_size  : co-occurrence window for context evidence

        Returns
        -------
        EvidenceBundle with full per-category breakdowns.
        """
        w           = self.evidence_weights
        phrase_ids  = {p[0] for p in phrases}
        token_texts = [t.text for t in tokens]
        token_set   = set(token_texts)
        token_freq  = Counter(token_texts)
        per_category: Dict[str, CategoryEvidence] = {}

        for label in graph.labels:
            items:      List[EvidenceItem] = []
            activations = prop_results.get(label, PropagationResult({}, [], label)).activations

            # ---- E1: Direct evidence ----
            direct_score = 0.0
            for v, freq_v in token_freq.items():
                if v not in graph.nodes:
                    continue
                node  = graph.nodes[v]
                disc  = node.discriminativeness.get(label, 0.0)
                idf   = node.idf
                unc   = node.uncertainty.get(label, 1.0)
                # Designed structural pseudo-tokens (starting with __ but NOT word bigrams)
                # are DESIGNED features with known discriminativeness.  When they never
                # appear in a class during training (df_c=0 → uncertainty=0), the normal
                # formula silences their contribution entirely.  Apply a minimum certainty
                # of 0.10 so their discriminativeness still acts.
                # Example: __satisfaction_survey__ (0 spam docs, disc_spam=-26) should
                # push the spam score strongly negative when found in a test document.
                #
                # Word bigrams (__bg_*__) need gentler treatment:
                # - At unc=0.10 (same as structural): HAM-ONLY bigrams like __bg_if_one__
                #   each contribute -22, which crushes spam recall for texts using normal
                #   English constructions (e.g. "if one vid is not enough for today...").
                # - At unc=0.00 (no floor): ham texts lose all bigram-based protection and
                #   many borderline hams tip into spam.
                # unc=0.02 is the sweet spot: contributes ~-4.5 per HAM-ONLY bigram
                # (enough to provide moderate ham protection) without dominating
                # spam-discriminative signals in spam texts with ≤5 such bigrams.
                if unc == 0.0 and v.startswith("__"):
                    unc = 0.02 if v.startswith("__bg_") else 0.10
                contrib = freq_v * disc * idf * unc
                direct_score += contrib
                if abs(contrib) > _EPS:
                    items.append(EvidenceItem(
                        evidence_type="direct",
                        source=v,
                        label=label,
                        contribution=contrib,
                        explanation=(
                            f"'{v}' freq={freq_v}, "
                            f"disc={disc:.4f}, idf={idf:.4f}, unc={unc:.4f}"
                        ),
                    ))

            # ---- E2: Phrase evidence ----
            phrase_score = 0.0
            for pid in phrase_ids:
                if pid not in graph.nodes:
                    continue
                node  = graph.nodes[pid]
                disc  = node.discriminativeness.get(label, 0.0)
                idf   = node.idf
                unc   = node.uncertainty.get(label, 1.0)
                if unc == 0.0 and pid.startswith("__"):
                    unc = 0.02 if pid.startswith("__bg_") else 0.10
                contrib = disc * idf * unc * self.phrase_boost
                phrase_score += contrib
                if abs(contrib) > _EPS:
                    items.append(EvidenceItem(
                        evidence_type="phrase",
                        source=pid,
                        label=label,
                        contribution=contrib,
                        explanation=(
                            f"Phrase '{pid}' disc={disc:.4f}, "
                            f"idf={idf:.4f}, boost={self.phrase_boost}"
                        ),
                    ))

            # ---- E3: Context evidence (within-document pair co-occurrence) ----
            context_score = 0.0
            n_toks = len(tokens)
            for i in range(n_toks):
                for d in range(1, min(window_size, n_toks - i) + 1):
                    j = i + d
                    if j >= n_toks:
                        break
                    u = tokens[i].text
                    v = tokens[j].text
                    if u == v:
                        continue
                    edge_key = graph.edge_key(u, v)
                    if edge_key not in graph.edges:
                        continue
                    edge     = graph.edges[edge_key]
                    npmi_val = edge.npmi.get(label, 0.0)
                    conf     = edge.confidence.get(label, 0.0)
                    contrib  = npmi_val * conf * (1.0 / d)
                    context_score += contrib
                    if abs(contrib) > _EPS:
                        items.append(EvidenceItem(
                            evidence_type="context",
                            source=f"({u},{v})",
                            label=label,
                            contribution=contrib,
                            explanation=(
                                f"Pair ({u},{v}) dist={d}, "
                                f"NPMI={npmi_val:.4f}, conf={conf:.4f}"
                            ),
                        ))

            # ---- E4: Propagation evidence (non-seed activated nodes) ----
            prop_score = 0.0
            for node_id, strength in activations.items():
                if node_id in token_set or node_id in phrase_ids:
                    continue  # already counted in E1/E2
                if node_id not in graph.nodes:
                    continue
                node   = graph.nodes[node_id]
                disc   = node.discriminativeness.get(label, 0.0)
                idf    = node.idf
                contrib = strength * disc * idf
                prop_score += contrib
                if abs(contrib) > _EPS:
                    items.append(EvidenceItem(
                        evidence_type="propagation",
                        source=node_id,
                        label=label,
                        contribution=contrib,
                        explanation=(
                            f"Propagated '{node_id}' "
                            f"strength={strength:.4f}, disc={disc:.4f}, idf={idf:.4f}"
                        ),
                    ))

            # ---- E5: Interaction evidence (common-mediator triangles) ----
            interact_score = 0.0
            input_list = list(token_set)
            n_inp = len(input_list)
            for ii in range(n_inp):
                u = input_list[ii]
                nbrs_u = set(graph._adjacency.get(u, {}).keys())
                for jj in range(ii + 1, n_inp):
                    v_ = input_list[jj]
                    nbrs_v = set(graph._adjacency.get(v_, {}).keys())
                    common = (nbrs_u & nbrs_v) - token_set - phrase_ids
                    for m in common:
                        ek1 = graph.edge_key(u, m)
                        ek2 = graph.edge_key(v_, m)
                        if ek1 not in graph.edges or ek2 not in graph.edges:
                            continue
                        w1    = graph.edges[ek1].weight.get(label, 0.0)
                        w2    = graph.edges[ek2].weight.get(label, 0.0)
                        disc_m = (
                            graph.nodes[m].discriminativeness.get(label, 0.0)
                            if m in graph.nodes else 0.0
                        )
                        contrib = w1 * w2 * disc_m
                        interact_score += contrib
                        if abs(contrib) > _EPS:
                            items.append(EvidenceItem(
                                evidence_type="interaction",
                                source=f"({u})→[{m}]→({v_})",
                                label=label,
                                contribution=contrib,
                                explanation=(
                                    f"Mediator '{m}': "
                                    f"w({u},{m})={w1:.4f}, "
                                    f"w({v_},{m})={w2:.4f}, "
                                    f"disc={disc_m:.4f}"
                                ),
                            ))

            # ---- Final weighted sum ----
            total = (
                w.get("direct",      1.0) * direct_score +
                w.get("phrase",      1.2) * phrase_score +
                w.get("context",     0.8) * context_score +
                w.get("propagation", 0.6) * prop_score +
                w.get("interaction", 0.4) * interact_score
            )

            per_category[label] = CategoryEvidence(
                label=label,
                direct_score=direct_score,
                phrase_score=phrase_score,
                context_score=context_score,
                propagation_score=prop_score,
                interaction_score=interact_score,
                total_score=total,
                items=items,
            )

        return EvidenceBundle(
            per_category=per_category,
            input_tokens=tokens,
            phrases_found=phrases,
            propagation_results=prop_results,
        )
