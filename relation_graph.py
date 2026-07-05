"""
relation_graph.py — The sole knowledge store for LexiDecay v2.

Every statistic the model uses lives here exclusively.
No parallel counter structures, no shadow dicts exist outside this module.

Mathematical background:
  disc(v,c)  = log[P(v|c) / P(v)]             node log-odds discriminativeness
  NPMI(u,v,c)= PMI(u,v,c) / -log(P(u,v|c))   normalized PMI ∈ [-1,1]
  conf(u,v,c)= 1 - exp(-df(u,v,c)/N_c)        document-frequency confidence
  CS(u,v,c)  = log(cc(u,v,c)/global_cc) - log(1/|C|)   category specificity
  w(u,v,c)   = σ(NPMI) * conf * max(CS, 0)    final edge weight
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Distance Histogram
# ---------------------------------------------------------------------------

@dataclass
class DistanceHistogram:
    """Full distribution of co-occurrence distances for an edge."""
    counts: Dict[int, int] = field(default_factory=dict)

    def add(self, distance: int) -> None:
        self.counts[distance] = self.counts.get(distance, 0) + 1

    @property
    def mean(self) -> float:
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        return sum(d * c for d, c in self.counts.items()) / total

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def mode(self) -> int:
        if not self.counts:
            return 0
        return max(self.counts, key=self.counts.get)


# ---------------------------------------------------------------------------
# NodeData
# ---------------------------------------------------------------------------

@dataclass
class NodeData:
    """
    All knowledge about a single token or phrase node.
    Raw counts are updated incrementally; derived fields are recomputed
    by RelationGraph.recompute_weights() and cached here.
    """
    # Raw counts (incrementally updated)
    category_counts:      Dict[str, int] = field(default_factory=dict)
    document_frequencies: Dict[str, int] = field(default_factory=dict)
    global_count:         int = 0
    global_doc_frequency: int = 0

    # Phrase metadata
    is_phrase:          bool = False
    phrase_components:  Optional[Tuple[str, ...]] = None

    # Derived fields (recomputed by recompute_weights, cached here)
    discriminativeness: Dict[str, float] = field(default_factory=dict)
    uncertainty:        Dict[str, float] = field(default_factory=dict)
    idf:                float = 0.0


# ---------------------------------------------------------------------------
# EdgeData
# ---------------------------------------------------------------------------

@dataclass
class EdgeData:
    """
    All knowledge about a co-occurrence relationship between two nodes.
    The canonical key is always (min(a,b), max(a,b)).
    forward_count records occurrences of the alphabetically-first node
    appearing before the second node in actual text.
    """
    # Raw counts (incrementally updated)
    cooccurrence_count: Dict[str, float] = field(default_factory=dict)
    distance_histogram: DistanceHistogram = field(default_factory=DistanceHistogram)
    forward_count:      int = 0
    backward_count:     int = 0
    document_frequency: Dict[str, int] = field(default_factory=dict)

    # Derived fields (recomputed by recompute_weights, cached here)
    pmi:                  Dict[str, float] = field(default_factory=dict)
    npmi:                 Dict[str, float] = field(default_factory=dict)
    confidence:           Dict[str, float] = field(default_factory=dict)
    category_specificity: Dict[str, float] = field(default_factory=dict)
    weight:               Dict[str, float] = field(default_factory=dict)

    @property
    def average_distance(self) -> float:
        return self.distance_histogram.mean

    @property
    def total_cooccurrence(self) -> float:
        return sum(self.cooccurrence_count.values())

    @property
    def total_doc_frequency(self) -> int:
        return sum(self.document_frequency.values())

    def dominance_ratio(self) -> float:
        """Fraction of co-occurrences where canonical_a appeared before canonical_b."""
        total = self.forward_count + self.backward_count
        return self.forward_count / max(total, 1)


# ---------------------------------------------------------------------------
# RelationGraph
# ---------------------------------------------------------------------------

class RelationGraph:
    """
    The single, unified knowledge store for LexiDecay v2.

    Stores:
      nodes  — Dict[node_id, NodeData]
      edges  — Dict[(min_id, max_id), EdgeData]
      global statistics (also here, not external)

    All statistics used during inference are derived exclusively from
    self.nodes and self.edges. After recompute_weights() the cached
    derived fields on NodeData and EdgeData are current.
    """

    def __init__(self, window_size: int = 5, pmi_smoothing: float = _EPS,
                 uncertainty_lambda: float = 5.0):
        self.window_size:          int   = window_size
        self.pmi_smoothing:        float = pmi_smoothing
        self.uncertainty_lambda:   float = uncertainty_lambda

        # === Primary storage (the ONLY knowledge) ===
        self.nodes: Dict[str, NodeData]                  = {}
        self.edges: Dict[Tuple[str, str], EdgeData]      = {}

        # Global statistics stored IN the graph
        self.total_tokens_per_label: Dict[str, int]   = {}
        self.total_pairs_per_label:  Dict[str, float] = {}
        self.total_docs_per_label:   Dict[str, int]   = {}
        self.total_docs:             int              = 0
        self.labels:                 List[str]        = []

        # Adjacency index (derived, rebuilt during recompute_weights)
        # node → {neighbor: max_weight_across_labels}
        self._adjacency: Dict[str, Dict[str, float]] = {}
        # Set of phrase node IDs for O(1) lookup during inference
        self._phrase_index: Set[str] = set()

        self._weights_dirty: bool = True
        # Track edges/nodes dirtied during partial updates
        self._dirty_edges: Set[Tuple[str, str]] = set()
        self._dirty_nodes: Set[str]             = set()

    # ------------------------------------------------------------------
    # Incremental update — called once per training document
    # ------------------------------------------------------------------

    def update(
        self,
        token_texts: List[str],
        label: str,
        return_dirty: bool = False,
    ) -> Optional[Tuple[Set[Tuple[str, str]], Set[str]]]:
        """
        Incorporate a single tokenized document into the graph.

        Parameters
        ----------
        token_texts : list of normalized token strings
        label       : category label for this document
        return_dirty: if True, return (dirty_edges, dirty_nodes) for partial recompute

        This method updates only raw counts. Call recompute_weights() afterwards.
        """
        n = len(token_texts)
        if n == 0:
            return (set(), set()) if return_dirty else None

        # Register label
        if label not in self.total_docs_per_label:
            self.total_docs_per_label[label] = 0
            self.total_tokens_per_label[label] = 0
            self.total_pairs_per_label[label] = 0.0
            self.labels.append(label)

        dirty_nodes: Set[str] = set()
        dirty_edges: Set[Tuple[str, str]] = set()

        # --- 1. Node counts ---
        seen_in_doc: Set[str] = set()
        for tok in token_texts:
            if tok not in self.nodes:
                self.nodes[tok] = NodeData()
            self.nodes[tok].global_count += 1
            self.nodes[tok].category_counts[label] = (
                self.nodes[tok].category_counts.get(label, 0) + 1
            )
            seen_in_doc.add(tok)
            dirty_nodes.add(tok)

        for tok in seen_in_doc:
            self.nodes[tok].global_doc_frequency += 1
            self.nodes[tok].document_frequencies[label] = (
                self.nodes[tok].document_frequencies.get(label, 0) + 1
            )

        # --- 2. Global statistics ---
        self.total_tokens_per_label[label] += n
        self.total_docs_per_label[label]   += 1
        self.total_docs                    += 1

        # --- 3. Edge co-occurrence (all within-window pairs) ---
        seen_pairs_in_doc: Set[Tuple[str, str]] = set()
        weighted_pairs_this_doc = 0.0

        for i in range(n):
            tok_i = token_texts[i]
            for d in range(1, self.window_size + 1):
                j = i + d
                if j >= n:
                    break
                tok_j = token_texts[j]
                if tok_i == tok_j:
                    continue

                cooc_weight = 1.0 / d
                canon_a, canon_b = (tok_i, tok_j) if tok_i < tok_j else (tok_j, tok_i)
                edge_key = (canon_a, canon_b)

                if edge_key not in self.edges:
                    self.edges[edge_key] = EdgeData()
                edge = self.edges[edge_key]

                edge.cooccurrence_count[label] = (
                    edge.cooccurrence_count.get(label, 0) + cooc_weight
                )
                edge.distance_histogram.add(d)

                if tok_i < tok_j:
                    edge.forward_count  += 1
                else:
                    edge.backward_count += 1

                # Document-frequency: once per pair per document
                if edge_key not in seen_pairs_in_doc:
                    seen_pairs_in_doc.add(edge_key)
                    edge.document_frequency[label] = (
                        edge.document_frequency.get(label, 0) + 1
                    )

                dirty_edges.add(edge_key)
                weighted_pairs_this_doc += cooc_weight

        self.total_pairs_per_label[label] += weighted_pairs_this_doc
        self._weights_dirty = True

        if return_dirty:
            self._dirty_edges.update(dirty_edges)
            self._dirty_nodes.update(dirty_nodes)
            return dirty_edges, dirty_nodes
        return None

    # ------------------------------------------------------------------
    # Weight recomputation — derives all cached statistics from raw counts
    # ------------------------------------------------------------------

    def recompute_weights(
        self,
        only_edges: Optional[Set[Tuple[str, str]]] = None,
        only_nodes: Optional[Set[str]] = None,
    ) -> None:
        """
        Recompute all derived statistics from raw counts.

        If only_edges / only_nodes are given, only those are updated
        (used for incremental partial_fit). Otherwise full recomputation.

        Invariants after this call:
          node.discriminativeness[c] == log(P(v|c) / P(v))
          edge.npmi[c]               == NPMI(u,v|c)
          edge.weight[c]             == σ(NPMI) · conf · max(CS,0)
        """
        ε  = self.pmi_smoothing
        λu = self.uncertainty_lambda
        N  = max(self.total_docs, 1)
        total_tokens_global = max(sum(self.total_tokens_per_label.values()), 1)
        total_pairs_global  = max(sum(self.total_pairs_per_label.values()), _EPS)
        nL = max(len(self.labels), 1)

        nodes_to_process = only_nodes if only_nodes is not None else set(self.nodes.keys())
        edges_to_process = only_edges if only_edges is not None else set(self.edges.keys())

        # --- 1. Node derived statistics ---
        for v in nodes_to_process:
            if v not in self.nodes:
                continue
            node = self.nodes[v]
            node.idf = math.log((N + 1) / (node.global_doc_frequency + 1)) + 1.0
            p_v_global = (node.global_count + ε) / (total_tokens_global + ε)

            for label in self.labels:
                total_label = max(self.total_tokens_per_label.get(label, 1), 1)
                p_v_c = (node.category_counts.get(label, 0) + ε) / (total_label + ε)
                node.discriminativeness[label] = math.log(p_v_c / p_v_global)

                df_c = node.document_frequencies.get(label, 0)
                total_label_docs = max(self.total_docs_per_label.get(label, 1), 1)
                node.uncertainty[label] = 1.0 - math.exp(-df_c / λu)

        # --- 2. Edge derived statistics ---
        for edge_key in edges_to_process:
            if edge_key not in self.edges:
                continue
            a, b = edge_key
            edge = self.edges[edge_key]

            global_cooc = max(sum(edge.cooccurrence_count.values()), ε)

            for label in self.labels:
                total_label_tokens = max(self.total_tokens_per_label.get(label, 1), 1)
                total_label_pairs  = max(self.total_pairs_per_label.get(label, ε), ε)
                total_label_docs   = max(self.total_docs_per_label.get(label, 1), 1)

                p_a_c  = (self.nodes[a].category_counts.get(label, 0) + ε) / (total_label_tokens + ε) if a in self.nodes else ε
                p_b_c  = (self.nodes[b].category_counts.get(label, 0) + ε) / (total_label_tokens + ε) if b in self.nodes else ε
                p_ab_c = (edge.cooccurrence_count.get(label, 0) + ε) / (total_label_pairs + ε)

                # PMI and NPMI
                pmi_val = math.log(p_ab_c / (p_a_c * p_b_c + ε))
                denom   = -math.log(p_ab_c + ε)
                npmi_val = pmi_val / max(denom, ε)
                npmi_val = max(-1.0, min(1.0, npmi_val))   # clamp to [-1,1]

                edge.pmi[label]  = pmi_val
                edge.npmi[label] = npmi_val

                # Document-frequency confidence
                df_c = edge.document_frequency.get(label, 0)
                edge.confidence[label] = 1.0 - math.exp(-df_c / total_label_docs)

                # Category specificity: how much more this pair appears in c vs globally
                cat_cooc = edge.cooccurrence_count.get(label, 0)
                cs = math.log(
                    (cat_cooc / global_cooc + ε) / (1.0 / nL + ε)
                )
                edge.category_specificity[label] = cs

                # Final weight: sigmoid(NPMI) * confidence * max(CS, 0)
                sig_npmi = 1.0 / (1.0 + math.exp(-npmi_val))
                edge.weight[label] = sig_npmi * edge.confidence[label] * max(cs, 0.0)

        # --- 3. Rebuild adjacency index ---
        self._rebuild_adjacency()
        self._weights_dirty = False
        self._dirty_edges.clear()
        self._dirty_nodes.clear()

    def _rebuild_adjacency(self) -> None:
        """O(|E|) adjacency index rebuild. Stores max weight across labels per neighbor."""
        self._adjacency.clear()
        for (a, b), edge in self.edges.items():
            max_w = max(edge.weight.values()) if edge.weight else 0.0
            if max_w <= 0.0:
                continue
            self._adjacency.setdefault(a, {})[b] = max_w
            self._adjacency.setdefault(b, {})[a] = max_w

    # ------------------------------------------------------------------
    # Targeted edge update (used by GraphFineTuner without full recompute)
    # ------------------------------------------------------------------

    def update_edge_weight(
        self,
        edge_key: Tuple[str, str],
        label: str,
        delta_weight: float,
    ) -> None:
        """
        Apply a direct weight delta to edge_key for label.
        Back-propagates to NPMI to keep internal consistency.
        Updates adjacency index for affected nodes.
        """
        if edge_key not in self.edges:
            return
        edge = self.edges[edge_key]
        old_w = edge.weight.get(label, 0.0)
        new_w = max(0.0, old_w + delta_weight)
        edge.weight[label] = new_w

        # Back-propagate: invert w = σ(NPMI) * conf * CS_clamped
        conf = edge.confidence.get(label, _EPS)
        cs   = max(edge.category_specificity.get(label, _EPS), _EPS)
        denominator = conf * cs
        if denominator > _EPS:
            ratio = new_w / denominator
            ratio = min(max(ratio, _EPS), 1.0 - _EPS)
            edge.npmi[label] = math.log(ratio / (1.0 - ratio))  # logit

        # Update adjacency index
        a, b = edge_key
        new_max_w = max(edge.weight.values()) if edge.weight else 0.0
        for node in (a, b):
            neighbor = b if node == a else a
            if node in self._adjacency:
                if new_max_w > 0:
                    self._adjacency[node][neighbor] = new_max_w
                elif neighbor in self._adjacency[node]:
                    del self._adjacency[node][neighbor]

    def update_node_discriminativeness(
        self, node_id: str, label: str, delta: float
    ) -> None:
        """Direct discriminativeness delta for fine-tuning (node-level)."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.discriminativeness[label] = node.discriminativeness.get(label, 0.0) + delta

    # ------------------------------------------------------------------
    # Graph pruning
    # ------------------------------------------------------------------

    def prune(
        self,
        min_weight:   float = 0.01,
        min_cooc:     float = 2.0,
        min_doc_freq: int   = 2,
    ) -> int:
        """
        Remove edges below statistical significance.
        Returns number of edges removed.
        """
        to_delete: List[Tuple[str, str]] = []
        for key, edge in self.edges.items():
            max_w   = max(edge.weight.values()) if edge.weight else 0.0
            max_coc = max(edge.cooccurrence_count.values()) if edge.cooccurrence_count else 0.0
            min_df  = min(edge.document_frequency.values()) if edge.document_frequency else 0
            if max_w < min_weight or max_coc < min_cooc or min_df < min_doc_freq:
                to_delete.append(key)

        for key in to_delete:
            del self.edges[key]

        self._rebuild_adjacency()
        return len(to_delete)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all knowledge. Called at the start of fit()."""
        self.nodes.clear()
        self.edges.clear()
        self.total_tokens_per_label.clear()
        self.total_pairs_per_label.clear()
        self.total_docs_per_label.clear()
        self.total_docs = 0
        self.labels.clear()
        self._adjacency.clear()
        self._phrase_index.clear()
        self._dirty_edges.clear()
        self._dirty_nodes.clear()
        self._weights_dirty = True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Summary statistics for logging and visualization."""
        all_weights = [
            w
            for edge in self.edges.values()
            for w in edge.weight.values()
        ]
        degrees = [len(v) for v in self._adjacency.values()]
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        return {
            "num_nodes":        n_nodes,
            "num_edges":        n_edges,
            "num_phrase_nodes": sum(1 for n in self.nodes.values() if n.is_phrase),
            "num_labels":       len(self.labels),
            "labels":           list(self.labels),
            "total_docs":       self.total_docs,
            "total_tokens_per_label": dict(self.total_tokens_per_label),
            "avg_degree":       sum(degrees) / max(len(degrees), 1),
            "max_degree":       max(degrees, default=0),
            "graph_density":    2 * n_edges / max(n_nodes * (n_nodes - 1), 1),
            "mean_edge_weight": sum(all_weights) / max(len(all_weights), 1),
            "max_edge_weight":  max(all_weights, default=0.0),
            "memory_bytes":     self._estimate_memory(),
        }

    def _estimate_memory(self) -> int:
        """Rough memory estimate in bytes."""
        # Each NodeData: ~200 bytes base + 50 per label
        node_mem = len(self.nodes) * (200 + 50 * len(self.labels))
        # Each EdgeData: ~300 bytes base + 60 per label + distance histogram
        edge_mem = len(self.edges) * (300 + 60 * len(self.labels))
        return node_mem + edge_mem

    def get_top_edges(self, label: str, top_k: int = 20) -> List[Tuple[Tuple[str,str], float]]:
        """Return top-K edges by weight for a given label."""
        scored = [
            (key, edge.weight.get(label, 0.0))
            for key, edge in self.edges.items()
            if edge.weight.get(label, 0.0) > 0
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def get_top_nodes(self, label: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return top-K nodes by discriminativeness for a given label."""
        scored = [
            (nid, node.discriminativeness.get(label, 0.0))
            for nid, node in self.nodes.items()
            if not node.is_phrase
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def neighbors(self, node_id: str, label: Optional[str] = None, top_k: int = 0) -> List[Tuple[str, float]]:
        """
        Return neighbors of node_id sorted by weight.
        If label is given, use per-label weight; else use max-weight adjacency.
        If top_k > 0, return only the top_k neighbors.
        """
        if node_id not in self._adjacency:
            return []

        if label is not None:
            results = []
            for nbr in self._adjacency[node_id]:
                a, b = (node_id, nbr) if node_id < nbr else (nbr, node_id)
                edge_key = (a, b)
                w = self.edges[edge_key].weight.get(label, 0.0) if edge_key in self.edges else 0.0
                if w > 0:
                    results.append((nbr, w))
        else:
            results = list(self._adjacency[node_id].items())

        results.sort(key=lambda x: -x[1])
        return results[:top_k] if top_k > 0 else results

    def edge_key(self, a: str, b: str) -> Tuple[str, str]:
        """Return canonical edge key for pair (a, b)."""
        return (a, b) if a < b else (b, a)
