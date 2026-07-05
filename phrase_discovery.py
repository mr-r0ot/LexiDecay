"""
phrase_discovery.py — Stage 3: Statistical Multi-Word Expression Detection.

Uses Dunning's Log-Likelihood Ratio (G²) test (1993) plus an NPMI filter
to discover collocations that should become first-class graph nodes.

Why G² instead of raw PMI?
  PMI(A,B) = log(P(A,B) / P(A)·P(B))
  For rare bigrams, P(A,B) is unreliable — PMI explodes for low-count pairs.
  G² is a likelihood-ratio hypothesis test: under H₀ (independence),
  G² ~ χ²(1). The p-value is well-calibrated even for sparse counts.
  Dunning's threshold of G² > 10.83 corresponds to p < 0.001.

Phrase nodes are added directly to the RelationGraph as NodeData objects
with is_phrase=True. There is no separate phrase store.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .relation_graph import NodeData, RelationGraph

_EPS = 1e-12


class PhraseDiscovery:
    """
    Statistical phrase discovery via G² test.

    Discovered phrases are injected into the RelationGraph as first-class
    NodeData objects. No external phrase list is maintained.

    Parameters
    ----------
    g2_threshold       : G² cutoff for accepting a phrase (10.83 → p < 0.001)
    npmi_threshold     : Additional NPMI filter (0.3 is a moderate threshold)
    min_support        : Minimum raw co-occurrence count
    max_phrase_length  : Maximum n-gram length (2 = bigrams, 3 = trigrams)
    """

    PHRASE_SEP = "__"   # separator used in phrase node IDs, e.g. "machine__learning"

    def __init__(
        self,
        g2_threshold:    float = 10.83,
        npmi_threshold:  float = 0.3,
        min_support:     int   = 5,
        max_phrase_length: int = 2,
    ):
        self.g2_threshold    = g2_threshold
        self.npmi_threshold  = npmi_threshold
        self.min_support     = min_support
        self.max_phrase_length = max_phrase_length

    # ------------------------------------------------------------------
    # Main entry point: discovery from graph co-occurrence data
    # ------------------------------------------------------------------

    def discover(
        self,
        graph: RelationGraph,
        incremental: bool = False,
    ) -> List[str]:
        """
        Scan graph edges for statistically significant bigrams/trigrams.
        Adds discovered phrases as NodeData entries in-place.

        Parameters
        ----------
        graph       : the RelationGraph to scan and update
        incremental : if True, only process edges in graph._dirty_edges

        Returns
        -------
        List of newly created phrase node IDs.
        """
        discovered: List[str] = []
        total_tokens = max(sum(graph.total_tokens_per_label.values()), 1)

        # Edges to examine
        edges_to_check = graph._dirty_edges if incremental else set(graph.edges.keys())

        # --- Bigram discovery ---
        new_bigram_ids: Set[str] = set()
        for edge_key in edges_to_check:
            if edge_key not in graph.edges:
                continue
            a, b = edge_key
            edge = graph.edges[edge_key]

            global_cooc = sum(edge.cooccurrence_count.values())
            if global_cooc < self.min_support:
                continue
            if a not in graph.nodes or b not in graph.nodes:
                continue

            count_a = graph.nodes[a].global_count
            count_b = graph.nodes[b].global_count

            # G² contingency table
            C11 = max(int(round(global_cooc)), 0)
            C12 = max(count_a - C11, 0)
            C21 = max(count_b - C11, 0)
            C22 = max(total_tokens - C11 - C12 - C21, 0)

            g2 = self._g2(C11, C12, C21, C22)
            if g2 < self.g2_threshold:
                continue

            # Additional NPMI filter (global, not per-label)
            p_ab = (global_cooc + _EPS) / (total_tokens * graph.window_size + _EPS)
            p_a  = (count_a + _EPS) / (total_tokens + _EPS)
            p_b  = (count_b + _EPS) / (total_tokens + _EPS)
            pmi  = math.log(p_ab / (p_a * p_b + _EPS))
            npmi = pmi / max(-math.log(p_ab + _EPS), _EPS)
            npmi = max(-1.0, min(1.0, npmi))

            if npmi < self.npmi_threshold:
                continue

            phrase_id = f"{a}{self.PHRASE_SEP}{b}"
            if phrase_id not in graph.nodes:
                self._add_phrase_node(graph, a, b, phrase_id, global_cooc)
                new_bigram_ids.add(phrase_id)
                discovered.append(phrase_id)

        # --- Trigram discovery (extend validated bigrams) ---
        if self.max_phrase_length >= 3 and new_bigram_ids:
            for bigram_id in list(new_bigram_ids):
                parts   = bigram_id.split(self.PHRASE_SEP)
                last_part = parts[-1]

                for neighbor, _ in list(graph._adjacency.get(last_part, {}).items()):
                    if neighbor in parts:
                        continue  # no cyclic phrases
                    trigram_id = f"{bigram_id}{self.PHRASE_SEP}{neighbor}"
                    if trigram_id in graph.nodes:
                        continue

                    # Use the last_part→neighbor edge as proxy
                    ek = graph.edge_key(last_part, neighbor)
                    if ek not in graph.edges:
                        continue
                    ext_edge = graph.edges[ek]
                    ext_cooc = sum(ext_edge.cooccurrence_count.values())
                    if ext_cooc < self.min_support:
                        continue

                    # G² test using the bigram node stats as proxy
                    count_bigram = graph.nodes[bigram_id].global_count if bigram_id in graph.nodes else 0
                    count_neighbor = graph.nodes[neighbor].global_count if neighbor in graph.nodes else 0

                    C11 = max(int(round(ext_cooc)), 0)
                    C12 = max(count_bigram - C11, 0)
                    C21 = max(count_neighbor - C11, 0)
                    C22 = max(total_tokens - C11 - C12 - C21, 0)

                    g2 = self._g2(C11, C12, C21, C22)
                    if g2 < self.g2_threshold:
                        continue

                    self._add_phrase_node(graph, bigram_id, neighbor, trigram_id, ext_cooc)
                    discovered.append(trigram_id)

        # Update phrase index
        graph._phrase_index.update(discovered)

        # Recompute weights for new phrase nodes
        if discovered:
            graph.recompute_weights(
                only_nodes=set(discovered),
                only_edges=set(),
            )

        return discovered

    # ------------------------------------------------------------------
    # Phrase node creation
    # ------------------------------------------------------------------

    def _add_phrase_node(
        self,
        graph: RelationGraph,
        comp_a: str,
        comp_b: str,
        phrase_id: str,
        approx_cooc: float,
    ) -> None:
        """
        Add phrase as a first-class NodeData in the graph.

        Category counts are estimated as the min of the two components —
        a conservative estimate: the phrase can appear at most as often
        as the rarer of its two parts.
        """
        if phrase_id in graph.nodes:
            return

        node_a = graph.nodes.get(comp_a)
        node_b = graph.nodes.get(comp_b)

        if node_a is None or node_b is None:
            return

        cat_counts: Dict[str, int] = {}
        doc_freqs:  Dict[str, int] = {}
        for label in graph.labels:
            ca = node_a.category_counts.get(label, 0)
            cb = node_b.category_counts.get(label, 0)
            cat_counts[label] = min(ca, cb)

            dfa = node_a.document_frequencies.get(label, 0)
            dfb = node_b.document_frequencies.get(label, 0)
            doc_freqs[label] = min(dfa, dfb)

        global_count = max(int(round(approx_cooc)), 1)
        global_df    = min(node_a.global_doc_frequency, node_b.global_doc_frequency)

        graph.nodes[phrase_id] = NodeData(
            category_counts=cat_counts,
            document_frequencies=doc_freqs,
            global_count=global_count,
            global_doc_frequency=global_df,
            is_phrase=True,
            phrase_components=tuple(phrase_id.split(self.PHRASE_SEP)),
        )

    # ------------------------------------------------------------------
    # G² (Log-Likelihood Ratio) statistic
    # ------------------------------------------------------------------

    @staticmethod
    def _g2(C11: int, C12: int, C21: int, C22: int) -> float:
        """
        Dunning's (1993) G² log-likelihood ratio statistic.
        Under H₀ (independence), G² ~ χ²(1).
        Critical values: 3.84 (p<0.05), 6.63 (p<0.01), 10.83 (p<0.001).
        """
        N = C11 + C12 + C21 + C22
        if N <= 0:
            return 0.0

        def _term(o: int, e: float) -> float:
            if o <= 0 or e <= _EPS:
                return 0.0
            return o * math.log(o / e)

        # Expected counts under independence
        r1 = C11 + C12
        r2 = C21 + C22
        c1 = C11 + C21
        c2 = C12 + C22

        E11 = r1 * c1 / N
        E12 = r1 * c2 / N
        E21 = r2 * c1 / N
        E22 = r2 * c2 / N

        return 2.0 * (_term(C11, E11) + _term(C12, E12)
                      + _term(C21, E21) + _term(C22, E22))

    # ------------------------------------------------------------------
    # Phrase detection during inference (greedy longest-match)
    # ------------------------------------------------------------------

    def find_phrases(
        self,
        token_texts: List[str],
        graph: RelationGraph,
    ) -> List[Tuple[str, int, int]]:
        """
        Scan token_texts for known phrases using greedy left-to-right
        longest-match. Uses graph._phrase_index for O(1) membership tests.

        Returns
        -------
        List of (phrase_id, start_position, end_position) tuples.
        Positions are indices into token_texts.
        """
        phrases: List[Tuple[str, int, int]] = []
        n = len(token_texts)
        i = 0
        while i < n:
            matched = False
            max_len = min(self.max_phrase_length, n - i)
            for length in range(max_len, 1, -1):
                phrase_id = self.PHRASE_SEP.join(token_texts[i: i + length])
                if phrase_id in graph._phrase_index:
                    phrases.append((phrase_id, i, i + length - 1))
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return phrases

    @staticmethod
    def phrase_id_from_tokens(tokens: List[str]) -> str:
        """Construct canonical phrase ID from component tokens."""
        return PhraseDiscovery.PHRASE_SEP.join(tokens)

    @staticmethod
    def tokens_from_phrase_id(phrase_id: str) -> List[str]:
        """Decompose a phrase node ID into its component tokens."""
        return phrase_id.split(PhraseDiscovery.PHRASE_SEP)
