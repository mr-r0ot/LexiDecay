"""
propagation.py — Stage 4: Depth-Bounded Personalized PageRank Propagation.

Mathematical formulation:
  π_c^(0)(v) = s(v)     where s(v) = 1.0 if v ∈ input_tokens, else 0.0
  π_c^(d)(v) = α·s(v)  +  (1-α)·Σ_{u∈N(v)} [w_c(u,v) / Z_c(u)] · π_c^(d-1)(u)
  Z_c(u) = Σ_{v'∈N(u)} w_c(u,v')   [category-aware out-degree normalization]

Why depth 2–3 suffices:
  Graph average degree ≈ O(W) = O(window_size). After d hops, at most K^d nodes
  are reachable. For K=10, d=2: 100 max. Signal attenuates as (1-α)^d·decay^d,
  so at d=3 with α=0.15, decay=0.85: (0.85)^3·(0.85)^3 ≈ 0.16 of seed strength.
  Below min_threshold (default 0.005) all contributions are pruned.

Anti-drift mechanisms:
  1. Restart probability α: force-pulls activation back to seed nodes.
  2. Decay per hop: reduces transitive activation strength.
  3. top_k fan-out: prevents hub nodes from flooding the propagation.
  4. min_threshold: prunes insignificant activations before next hop.
  5. Seed exclusion from next-hop active set: prevents re-counting seed tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .relation_graph import RelationGraph

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropagationPath:
    """
    One activation path from a seed node to a propagated node.
    Stored for full explainability — every activated node can trace
    back to which input token caused its activation and via which route.
    """
    origin:     str          # input token that seeded this activation chain
    path:       List[str]    # ordered sequence of node IDs: [origin, hop1, hop2, ...]
    final_node: str          # last node in path
    strength:   float        # final activation strength at final_node
    depth:      int          # number of hops from origin
    label:      str          # category for which this propagation was computed


@dataclass
class PropagationResult:
    """
    Complete propagation result for one (input_doc, label) pair.
    activations: node_id → activation strength (includes both seed and propagated)
    paths:       explainability trace for all propagated (non-seed) activations
    label:       which category this result belongs to
    """
    activations: Dict[str, float]
    paths:       List[PropagationPath]
    label:       str
    n_hops_performed: int = 0


# ---------------------------------------------------------------------------
# PropagationEngine
# ---------------------------------------------------------------------------

class PropagationEngine:
    """
    Depth-bounded Personalized PageRank on the RelationGraph.

    For each category label, a separate propagation is run using that
    category's edge weights for neighbor ranking and normalization.

    Parameters
    ----------
    max_depth      : Maximum number of hops from seed nodes (default 2)
    decay          : Multiplicative decay applied per hop to all activations
    top_k          : Maximum neighbors to follow per node per hop
    restart_alpha  : Probability of returning to seed set at each hop
    min_threshold  : Prune activations below this strength (saves computation)
    """

    def __init__(
        self,
        max_depth:     int   = 2,
        decay:         float = 0.85,
        top_k:         int   = 10,
        restart_alpha: float = 0.15,
        min_threshold: float = 0.005,
        min_seed_disc: float = 0.0,
    ):
        self.max_depth     = max_depth
        self.decay         = decay
        self.top_k         = top_k
        self.restart_alpha = restart_alpha
        self.min_threshold = min_threshold
        self.min_seed_disc = min_seed_disc

    def propagate(
        self,
        input_tokens: List[str],
        label: str,
        graph: RelationGraph,
    ) -> PropagationResult:
        """
        Run bounded PPR from input_tokens through the graph for category `label`.

        Only tokens actually present in graph.nodes are seeded.
        When min_seed_disc > 0, only tokens with |disc(v, label)| >= min_seed_disc
        are used as propagation seeds. This prevents weakly-discriminative function
        words (e.g. "to", "for") from propagating noise to spam/ham hub nodes.

        Returns PropagationResult with:
          - activations: all activated nodes (seed + propagated) with strengths
          - paths: full propagation trace for explainability
        """
        seed_set: Set[str] = set()
        for tok in input_tokens:
            if tok in graph.nodes:
                if self.min_seed_disc > 0.0:
                    disc = graph.nodes[tok].discriminativeness.get(label, 0.0)
                    if abs(disc) < self.min_seed_disc:
                        continue
                seed_set.add(tok)

        if not seed_set:
            return PropagationResult(activations={}, paths=[], label=label)

        # Initial activation: all seeds get strength 1.0
        # Track (strength, origin_seed, path_so_far) per node
        active: Dict[str, Tuple[float, str, List[str]]] = {
            t: (1.0, t, [t]) for t in seed_set
        }
        all_activations: Dict[str, float] = {t: 1.0 for t in seed_set}
        all_paths: List[PropagationPath] = []
        hops_done = 0

        for depth in range(1, self.max_depth + 1):
            next_active: Dict[str, Tuple[float, str, List[str]]] = {}

            for node, (strength, origin, path) in active.items():
                # Gather neighbors and their per-label weights
                nbr_weights = self._get_neighbor_weights(node, label, graph)
                if not nbr_weights:
                    continue

                # Degree normalization using category-specific weights
                Z = sum(w for _, w in nbr_weights) + _EPS

                for neighbor, edge_w in nbr_weights:
                    # PPR step: (1-α) * normalized_walk + α * restart
                    walk_contrib   = (1.0 - self.restart_alpha) * (edge_w / Z) * strength
                    restart_contrib = self.restart_alpha * (1.0 if neighbor in seed_set else 0.0)
                    new_strength   = self.decay * walk_contrib + restart_contrib

                    if new_strength < self.min_threshold:
                        continue

                    # Keep maximum activation if node is reachable via multiple paths
                    existing = next_active.get(neighbor, (0.0,))[0]
                    if new_strength > existing:
                        next_active[neighbor] = (new_strength, origin, path + [neighbor])

            # Merge into global activation table (keep maximum per node)
            for node, (s, origin, path_) in next_active.items():
                if node not in seed_set:
                    prev = all_activations.get(node, 0.0)
                    if s > prev:
                        all_activations[node] = s
                        all_paths.append(PropagationPath(
                            origin=origin,
                            path=list(path_),
                            final_node=node,
                            strength=s,
                            depth=depth,
                            label=label,
                        ))

            # Next iteration: only non-seed nodes (no seed re-activation)
            active = {n: v for n, v in next_active.items() if n not in seed_set}
            hops_done = depth
            if not active:
                break

        return PropagationResult(
            activations=all_activations,
            paths=all_paths,
            label=label,
            n_hops_performed=hops_done,
        )

    def propagate_all_labels(
        self,
        input_tokens: List[str],
        graph: RelationGraph,
    ) -> Dict[str, PropagationResult]:
        """
        Run propagation for every label in graph.labels.
        Returns dict mapping label → PropagationResult.
        """
        return {
            label: self.propagate(input_tokens, label, graph)
            for label in graph.labels
        }

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_neighbor_weights(
        self,
        node: str,
        label: str,
        graph: RelationGraph,
    ) -> List[Tuple[str, float]]:
        """
        Return top-K neighbors of `node` sorted by per-label edge weight.
        Uses graph._adjacency for fast neighbor enumeration, then resolves
        per-label weight from graph.edges.
        """
        adjacency = graph._adjacency.get(node)
        if not adjacency:
            return []

        results: List[Tuple[str, float]] = []
        for neighbor in adjacency:
            a, b = graph.edge_key(node, neighbor)
            edge_key = (a, b)
            if edge_key in graph.edges:
                w = graph.edges[edge_key].weight.get(label, 0.0)
                if w > 0.0:
                    results.append((neighbor, w))

        # Sort descending and return top_k
        results.sort(key=lambda x: -x[1])
        return results[: self.top_k]
