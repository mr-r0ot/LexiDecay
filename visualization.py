"""
visualization.py — Professional Visualization Suite for LexiDecay v2.

All methods return matplotlib.Figure objects.
plt.show() is NEVER called internally — callers decide to show or save.

Included plots:
  1.  plot_confusion_matrix        heatmap with annotations
  2.  plot_roc_curves              multi-class one-vs-rest overlay
  3.  plot_pr_curves               precision-recall curves
  4.  plot_learning_curve          train vs validation performance vs size
  5.  plot_evaluation_radar        radar/spider chart for metrics
  6.  plot_graph_growth            nodes & edges over training documents
  7.  plot_degree_distribution     log-log degree distribution
  8.  plot_weight_histogram        edge weight distribution per label
  9.  plot_category_distribution   category frequency bar chart
  10. plot_confidence_distribution histogram of prediction confidence
  11. plot_evidence_breakdown      stacked bar: 5 evidence sources per category
  12. plot_propagation_subgraph    network diagram of propagation paths
  13. plot_feature_importance      horizontal bar chart (top positive/negative)
  14. plot_memory_usage            graph memory composition
  15. plot_inference_speed         tokens/ms scatter with trend line
  16. plot_fine_tune_progress      before/after probability comparison
  17. plot_hyperparameter_comparison parallel coordinates for optimizer results
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (works in all environments)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    import numpy as np
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    Figure = object  # type: ignore

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib and numpy are required for visualization. "
            "Install with: pip install matplotlib numpy"
        )


def _new_fig(figsize: Tuple[float, float] = (10, 6)) -> Tuple["Figure", Any]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    return fig, ax


class Visualizer:
    """
    Professional visualization for LexiDecay v2.
    All methods return plt.Figure. Call fig.savefig(path) or fig.show() externally.
    """

    # ------------------------------------------------------------------
    # 1. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        cm:     List[List[int]],
        labels: List[str],
        title:  str = "Confusion Matrix",
        normalize: bool = False,
    ) -> "Figure":
        _require_mpl()
        arr = np.array(cm, dtype=float)
        if normalize:
            row_sums = arr.sum(axis=1, keepdims=True)
            arr = arr / np.where(row_sums == 0, 1, row_sums)

        fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
        im = ax.imshow(arr, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)

        thresh = arr.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = f"{arr[i,j]:.2f}" if normalize else f"{int(arr[i,j])}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if arr[i, j] > thresh else "black", fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. ROC Curves
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        roc_result,   # ROCResult from Evaluator.roc_auc()
        title: str = "ROC Curves (One-vs-Rest)",
    ) -> "Figure":
        _require_mpl()
        fig, ax = _new_fig()
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

        for i, (label, fpr) in enumerate(roc_result.fpr.items()):
            tpr = roc_result.tpr[label]
            auc = roc_result.auc[label]
            color = _PALETTE[i % len(_PALETTE)]
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC={auc:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title}\nMacro AUC = {roc_result.macro_auc:.3f}")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Precision-Recall Curves
    # ------------------------------------------------------------------

    def plot_pr_curves(
        self,
        pr_result,   # PRResult from Evaluator.pr_auc()
        title: str = "Precision-Recall Curves",
    ) -> "Figure":
        _require_mpl()
        fig, ax = _new_fig()

        for i, (label, prec) in enumerate(pr_result.precision.items()):
            rec  = pr_result.recall[label]
            ap   = pr_result.ap[label]
            color = _PALETTE[i % len(_PALETTE)]
            ax.plot(rec, prec, color=color, lw=2,
                    label=f"{label} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}\nMacro AP = {pr_result.macro_ap:.3f}")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. Learning Curve
    # ------------------------------------------------------------------

    def plot_learning_curve(
        self,
        lc_result,   # LearningCurveResult
        metric: str = "F1 Macro",
        title:  str = "Learning Curve",
    ) -> "Figure":
        _require_mpl()
        fig, ax = _new_fig()
        sizes = lc_result.train_sizes
        ax.plot(sizes, lc_result.train_scores, "o-", color=_PALETTE[0],
                lw=2, label="Train")
        ax.plot(sizes, lc_result.val_scores, "s--", color=_PALETTE[1],
                lw=2, label="Validation")
        ax.fill_between(sizes, lc_result.train_scores, lc_result.val_scores,
                        alpha=0.1, color=_PALETTE[2])
        ax.set_xlabel("Training Examples")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.set_ylim([0, 1.05])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 5. Evaluation Radar Chart
    # ------------------------------------------------------------------

    def plot_evaluation_radar(
        self,
        metrics: Dict[str, float],
        title:   str = "Evaluation Metrics",
    ) -> "Figure":
        _require_mpl()
        # Select scalar metrics suitable for radar
        keys   = [k for k, v in metrics.items() if isinstance(v, float) and 0 <= v <= 1]
        values = [metrics[k] for k in keys]
        if not keys:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No valid metrics", ha="center")
            return fig

        n = len(keys)
        angles = [2 * math.pi * i / n for i in range(n)] + [0]
        values_plot = values + [values[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.plot(angles, values_plot, "o-", lw=2, color=_PALETTE[0])
        ax.fill(angles, values_plot, alpha=0.25, color=_PALETTE[0])
        ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], keys)
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=20)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 6. Graph Growth Over Training
    # ------------------------------------------------------------------

    def plot_graph_growth(
        self,
        stats_per_step: List[Dict[str, Any]],
        title: str = "Graph Growth During Training",
    ) -> "Figure":
        _require_mpl()
        fig, ax1 = plt.subplots(figsize=(10, 5))
        steps  = list(range(1, len(stats_per_step) + 1))
        nodes  = [s.get("num_nodes", 0) for s in stats_per_step]
        edges  = [s.get("num_edges", 0) for s in stats_per_step]

        ax1.plot(steps, nodes, color=_PALETTE[0], lw=2, label="Nodes")
        ax1.set_xlabel("Training Documents")
        ax1.set_ylabel("Node Count", color=_PALETTE[0])
        ax1.tick_params(axis="y", labelcolor=_PALETTE[0])

        ax2 = ax1.twinx()
        ax2.plot(steps, edges, color=_PALETTE[1], lw=2, linestyle="--", label="Edges")
        ax2.set_ylabel("Edge Count", color=_PALETTE[1])
        ax2.tick_params(axis="y", labelcolor=_PALETTE[1])

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title(title)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 7. Degree Distribution
    # ------------------------------------------------------------------

    def plot_degree_distribution(
        self,
        graph,   # RelationGraph
        title: str = "Node Degree Distribution",
    ) -> "Figure":
        _require_mpl()
        degrees = [len(v) for v in graph._adjacency.values()]
        if not degrees:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No edges"); return fig

        fig, ax = _new_fig()
        bins = np.logspace(np.log10(max(min(degrees), 1)),
                           np.log10(max(degrees)), 30)
        ax.hist(degrees, bins=bins, color=_PALETTE[0], edgecolor="white", alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (log scale)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(title)
        ax.text(0.7, 0.9, f"Mean: {sum(degrees)/len(degrees):.1f}",
                transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 8. Edge Weight Histogram
    # ------------------------------------------------------------------

    def plot_weight_histogram(
        self,
        graph,
        label:  Optional[str] = None,
        title:  str = "Edge Weight Distribution",
        bins:   int = 50,
    ) -> "Figure":
        _require_mpl()
        if label:
            weights = [e.weight.get(label, 0.0) for e in graph.edges.values()
                       if e.weight.get(label, 0.0) > 0]
        else:
            weights = [w for e in graph.edges.values() for w in e.weight.values() if w > 0]

        if not weights:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No weights"); return fig

        fig, ax = _new_fig()
        ax.hist(weights, bins=bins, color=_PALETTE[2], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Edge Weight")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}" + (f" — {label}" if label else ""))
        ax.axvline(sum(weights)/len(weights), color="red", lw=1.5,
                   linestyle="--", label=f"Mean: {sum(weights)/len(weights):.4f}")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 9. Category Distribution
    # ------------------------------------------------------------------

    def plot_category_distribution(
        self,
        y: List[str],
        title: str = "Category Distribution",
    ) -> "Figure":
        _require_mpl()
        from collections import Counter
        counts = Counter(y)
        labels_ = sorted(counts.keys())
        vals    = [counts[l] for l in labels_]

        fig, ax = _new_fig((max(6, len(labels_)), 5))
        bars = ax.bar(labels_, vals,
                      color=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels_))],
                      edgecolor="white", alpha=0.9)
        ax.bar_label(bars, padding=3)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_title(title)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 10. Confidence Distribution
    # ------------------------------------------------------------------

    def plot_confidence_distribution(
        self,
        predictions: List,   # List[PredictionResult]
        title: str = "Prediction Confidence Distribution",
        bins:  int = 30,
    ) -> "Figure":
        _require_mpl()
        confs = [r.probability for r in predictions]
        if not confs:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No predictions"); return fig

        fig, ax = _new_fig()
        ax.hist(confs, bins=bins, color=_PALETTE[4], edgecolor="white", alpha=0.85)
        ax.axvline(sum(confs)/len(confs), color="red", lw=1.5, linestyle="--",
                   label=f"Mean: {sum(confs)/len(confs):.4f}")
        ax.set_xlabel("Prediction Confidence (probability)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 11. Evidence Breakdown (stacked bars)
    # ------------------------------------------------------------------

    def plot_evidence_breakdown(
        self,
        result,   # PredictionResult
        title: str = "Evidence Breakdown by Category",
    ) -> "Figure":
        _require_mpl()
        evidence = result.evidence
        if not evidence:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No evidence"); return fig

        labels_ = list(evidence.keys())
        ev_types = ["direct", "phrase", "context", "propagation", "interaction"]
        colors   = _PALETTE[:len(ev_types)]

        data = {et: [] for et in ev_types}
        for l in labels_:
            ce = evidence[l]
            data["direct"].append(ce.direct_score)
            data["phrase"].append(ce.phrase_score)
            data["context"].append(ce.context_score)
            data["propagation"].append(ce.propagation_score)
            data["interaction"].append(ce.interaction_score)

        x   = np.arange(len(labels_))
        fig, ax = _new_fig((max(8, len(labels_) * 2), 6))
        bottom = np.zeros(len(labels_))

        for et, color in zip(ev_types, colors):
            vals = np.array(data[et])
            ax.bar(x, vals, bottom=bottom, label=et, color=color, edgecolor="white", alpha=0.85)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.axhline(0, color="black", lw=0.8)

        # Mark predicted label
        pred = result.predicted_label
        if pred in labels_:
            pred_x = labels_.index(pred)
            ax.get_xticklabels()[pred_x].set_fontweight("bold")
            ax.get_xticklabels()[pred_x].set_color(_PALETTE[3])

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 12. Propagation Subgraph
    # ------------------------------------------------------------------

    def plot_propagation_subgraph(
        self,
        result,   # PredictionResult
        graph,    # RelationGraph
        label:    Optional[str] = None,
        max_paths: int = 15,
        title: str = "Propagation Paths",
    ) -> "Figure":
        _require_mpl()
        paths = result.activated_paths
        if label:
            paths = [p for p in paths if p.label == label]
        paths = paths[:max_paths]

        if not paths:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No propagation paths", ha="center", va="center")
            ax.set_title(title)
            return fig

        # Build node set and edges
        nodes_set = set()
        edges_list = []
        for path in paths:
            for i in range(len(path.path) - 1):
                u, v = path.path[i], path.path[i + 1]
                nodes_set.add(u); nodes_set.add(v)
                edges_list.append((u, v, path.strength))

        nodes_list = sorted(nodes_set)
        n = len(nodes_list)

        # Simple circular layout
        pos = {
            node: (math.cos(2 * math.pi * i / max(n, 1)),
                   math.sin(2 * math.pi * i / max(n, 1)))
            for i, node in enumerate(nodes_list)
        }

        # Input tokens (seeds) get special color
        input_texts = set(t.text for t in result.tokens)
        phrase_ids  = set(result.phrases_found)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title + (f" — {label}" if label else ""), pad=10)

        # Draw edges
        max_strength = max((s for _, _, s in edges_list), default=1.0)
        for u, v, strength in edges_list:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            alpha  = max(0.2, strength / max(max_strength, 1e-9))
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#888", alpha=alpha, lw=1.5))

        # Draw nodes
        for node in nodes_list:
            x, y = pos[node]
            if node in input_texts or node in phrase_ids:
                color, size = _PALETTE[3], 300
            else:
                color, size = _PALETTE[0], 150
            ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(
                node[:12], (x, y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, zorder=6,
            )

        # Legend
        patches = [
            mpatches.Patch(color=_PALETTE[3], label="Input token / Phrase"),
            mpatches.Patch(color=_PALETTE[0], label="Propagated node"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 13. Feature Importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        result,     # PredictionResult
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> "Figure":
        _require_mpl()
        pos_feats = result.top_positive_features[:top_n]
        neg_feats = result.top_negative_features[:top_n]

        all_feats = list(neg_feats) + list(pos_feats)
        if not all_feats:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No features", ha="center")
            return fig

        names   = [f.token_or_edge[:25] for f in all_feats]
        contribs = [f.contribution for f in all_feats]
        colors   = [_PALETTE[0] if c >= 0 else _PALETTE[3] for c in contribs]

        fig, ax = plt.subplots(figsize=(10, max(6, len(all_feats) * 0.4)))
        y_pos = range(len(names))
        ax.barh(list(y_pos), contribs, color=colors, edgecolor="white", alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Contribution")
        ax.set_title(f"{title} — {result.predicted_label}")

        pos_patch = mpatches.Patch(color=_PALETTE[0], label="Positive")
        neg_patch = mpatches.Patch(color=_PALETTE[3], label="Negative")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 14. Memory Usage
    # ------------------------------------------------------------------

    def plot_memory_usage(
        self,
        graph,   # RelationGraph
        title: str = "Graph Memory Usage",
    ) -> "Figure":
        _require_mpl()
        stats   = graph.get_stats()
        n_nodes = stats["num_nodes"]
        n_edges = stats["num_edges"]
        n_lab   = max(stats["num_labels"], 1)

        # Approximate per-component memory
        node_base   = n_nodes * 200
        node_labels = n_nodes * n_lab * 50
        edge_base   = n_edges * 300
        edge_labels = n_edges * n_lab * 60
        adjacency   = n_edges * 32

        labels_ = ["Node base", "Node×label", "Edge base", "Edge×label", "Adjacency index"]
        sizes   = [node_base, node_labels, edge_base, edge_labels, adjacency]

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_,
            colors=_PALETTE[:len(labels_)],
            autopct="%1.1f%%", startangle=90,
        )
        ax.set_title(f"{title}\nTotal ≈ {sum(sizes)/1024:.1f} KB")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 15. Inference Speed
    # ------------------------------------------------------------------

    def plot_inference_speed(
        self,
        predictions: List,   # List[PredictionResult]
        title: str = "Inference Speed vs. Document Length",
    ) -> "Figure":
        _require_mpl()
        lengths = [len(r.tokens) for r in predictions]
        times   = [r.inference_time_ms for r in predictions]

        if not lengths:
            fig, ax = plt.subplots(); return fig

        fig, ax = _new_fig()
        ax.scatter(lengths, times, alpha=0.5, color=_PALETTE[0], s=30)

        # Trend line
        if len(lengths) > 2:
            z = np.polyfit(lengths, times, 1)
            p = np.poly1d(z)
            xs = sorted(lengths)
            ax.plot(xs, p(xs), "--", color=_PALETTE[3], lw=1.5, label="Trend")
            ax.legend()

        ax.set_xlabel("Document Length (tokens)")
        ax.set_ylabel("Inference Time (ms)")
        ax.set_title(title)
        mean_t = sum(times) / max(len(times), 1)
        ax.text(0.05, 0.95, f"Mean: {mean_t:.2f} ms",
                transform=ax.transAxes, fontsize=10, va="top")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 16. Fine-tune Progress
    # ------------------------------------------------------------------

    def plot_fine_tune_progress(
        self,
        before,   # PredictionResult (before fine-tune)
        after,    # PredictionResult (after fine-tune)
        title: str = "Fine-Tuning Effect on Probabilities",
    ) -> "Figure":
        _require_mpl()
        labels_  = sorted(set(before.probabilities) | set(after.probabilities))
        x        = np.arange(len(labels_))
        w        = 0.35

        before_p = [before.probabilities.get(l, 0.0) for l in labels_]
        after_p  = [after.probabilities.get(l, 0.0) for l in labels_]

        fig, ax = _new_fig((max(8, len(labels_) * 2), 5))
        ax.bar(x - w/2, before_p, w, label="Before", color=_PALETTE[0], alpha=0.8)
        ax.bar(x + w/2, after_p,  w, label="After",  color=_PALETTE[1], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=20, ha="right")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1.1])
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color="black", lw=0.5)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 17. Hyperparameter Comparison (parallel coordinates)
    # ------------------------------------------------------------------

    def plot_hyperparameter_comparison(
        self,
        opt_result,   # OptimizationResult
        top_n:  int = 20,
        title:  str = "Hyperparameter Comparison",
    ) -> "Figure":
        _require_mpl()
        trials = opt_result.all_trials[:top_n]
        if not trials:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No trials"); return fig

        # Select numeric hyperparameters only
        sample_params = trials[0].params
        num_keys = [k for k, v in sample_params.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not num_keys:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric hyperparameters for parallel plot")
            return fig

        n_axes = len(num_keys)
        scores = [t.score for t in trials]
        max_s  = max(scores) if scores else 1.0
        min_s  = min(scores) if scores else 0.0
        score_range = max(max_s - min_s, 1e-9)

        fig, axes = plt.subplots(1, n_axes, figsize=(3 * n_axes, 6), sharey=False)
        if n_axes == 1:
            axes = [axes]

        # Normalize each parameter column to [0,1]
        def normalize(vals):
            mn, mx = min(vals), max(vals)
            rng = max(mx - mn, 1e-9)
            return [(v - mn) / rng for v in vals]

        col_data = {k: normalize([t.params[k] for t in trials]) for k in num_keys}
        score_norm = [(s - min_s) / score_range for s in scores]

        cmap = plt.get_cmap("RdYlGn")
        for trial_i, trial in enumerate(trials):
            color = cmap(score_norm[trial_i])
            ys = [col_data[k][trial_i] for k in num_keys]
            for ax_i, ax in enumerate(axes[:-1]):
                ax.plot([0, 1], [ys[ax_i], ys[ax_i + 1]], color=color, alpha=0.6, lw=1.2)

        for ax_i, (ax, k) in enumerate(zip(axes, num_keys)):
            raw_vals = [t.params[k] for t in trials]
            mn, mx = min(raw_vals), max(raw_vals)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_xlabel(k, rotation=30, ha="right", fontsize=9)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels([f"{mn:.3g}", f"{(mn+mx)/2:.3g}", f"{mx:.3g}"], fontsize=8)

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(vmin=min_s, vmax=max_s))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[-1], label=opt_result.metric)
        fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()
        return fig
