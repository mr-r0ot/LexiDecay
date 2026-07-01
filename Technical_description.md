# LexiDecay v2 — Technical Description

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Stage 1: Tokenization and Structural Feature Extraction](#stage-1-tokenization-and-structural-feature-extraction)
3. [Stage 2: RelationGraph Construction](#stage-2-relationgraph-construction)
4. [Stage 3: Phrase Discovery](#stage-3-phrase-discovery)
5. [Stage 4: Evidence Aggregation](#stage-4-evidence-aggregation)
6. [Stage 5: Graph Propagation (Bounded PPR)](#stage-5-graph-propagation-bounded-ppr)
7. [Threshold Calibration](#threshold-calibration)
8. [Class Prior Correction](#class-prior-correction)
9. [Hyperparameter Reference](#hyperparameter-reference)

---

## Architecture Overview

LexiDecay v2 is a statistical graph-based text classifier. **The RelationGraph is the sole knowledge store.** All information from training — token statistics, phrase nodes, co-occurrence edges, discriminativeness scores — lives in a single graph. No separate feature vectors, TF-IDF matrices, or model weights are maintained.

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: Structural Extraction │  regex pseudo-tokens injected
│  + Tokenization                 │  before normal tokenization
└─────────────────────────────────┘
    │  TokenizedDocument
    ▼
┌─────────────────────────────────┐
│  Stage 2: RelationGraph         │  nodes: tokens, phrases, structural
│  (single knowledge store)       │  edges: weighted co-occurrence
│  - discriminativeness           │  disc(v,c) = log(P(v|c)/P(v))
│  - uncertainty                  │  unc(v,c)  = 1 - exp(-df_c/λ)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 3: Phrase Discovery      │  G² + NPMI + min_support filter
│  (G² > 10.83, NPMI > 0.3)      │  phrase nodes added to graph
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 4: Evidence Aggregation  │  5 evidence types per class
│  E1: direct     weight=1.0      │  each: disc × idf × uncertainty
│  E2: phrase     weight=1.5      │
│  E3: context    weight=0.25     │
│  E4: propagation weight=0.10    │
│  E5: interaction weight=0.10    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 5: Scoring + Threshold   │  raw scores → softmax → threshold
│  - class prior correction       │  → predicted label
│  - calibrated threshold         │
└─────────────────────────────────┘
```

---

## Stage 1: Tokenization and Structural Feature Extraction

### Structural Feature Extraction

Before standard tokenization, a `StructuralFeatureExtractor` scans the raw text with regular expressions and injects pseudo-tokens. These pseudo-tokens flow through the rest of the pipeline as ordinary tokens, acquire discriminativeness scores during training, and contribute evidence at inference time.

**Patterns and pseudo-tokens:**

| Pseudo-token | Regex | Description |
|---|---|---|
| `__LONG_NUM__` | `\b\d{8,15}\b` | Phone numbers (09061701461) and long codes |
| `__SHORT_CODE__` | `\b\d{4,6}\b` | SMS short codes (87121, 81010) |
| `__MONEY__` | `[£$€¤]\s*[\d,.]+` or `\b\d+p\b` | Currency amounts (£900, 50p) |
| `__URL__` | `https?://\S+` or `www\.\S+` | Hyperlinks |
| `__CAPS_WORD__` | `\b[A-Z]{3,}\b` | All-caps words (WINNER, FREE, URGENT) |
| `__DATE__` | `\b\d{1,2}/\d{1,2}/\d{2,4}\b` | Dates (12/06/2006) |
| `__AT_MENTION__` | `@\w+` | Social media mentions |

**Absence tokens:** When a pattern is not found, `__NO_LONG_NUM__`, `__NO_CAPS_WORD__`, etc. are injected. These become discriminative ham features: ham documents with no phone numbers or all-caps words actively push the score toward ham.

**Empirical discriminativeness (SMS Spam Collection):**

| Pseudo-token | P(present | spam) | P(present | ham) | disc(spam) |
|---|---|---|---|---|
| `__LONG_NUM__` | 0.507 | 0.0007 | +6.58 |
| `__SHORT_CODE__` | 0.515 | 0.0013 | +5.68 |
| `__MONEY__` | 0.218 | 0.0014 | +4.86 |
| `__URL__` | 0.142 | 0.0004 | +5.77 |
| `__CAPS_WORD__` | 0.742 | 0.0783 | +2.25 |

These single structural tokens often provide more discriminative signal than entire sentences of ambiguous vocabulary.

### Standard Tokenization

After structural feature injection, the text passes through a standard tokenizer:
- Lowercase normalization
- Punctuation handling
- Token windowing: for each token at position `i`, the local window `[i - window_size, i + window_size]` is recorded

---

## Stage 2: RelationGraph Construction

### Nodes

Every unique token `v` (including structural pseudo-tokens and phrase nodes) is a graph node with the following statistics maintained per category `c`:

- `df_c(v)`: document frequency of `v` in class `c` training documents
- `tf_c(v)`: total term frequency of `v` in class `c`
- `N_c`: total number of training documents in class `c`
- `N`: total number of training documents

### Edges

An undirected edge `(u, v)` exists if tokens `u` and `v` co-occurred within the same window in at least one document. Edge weight is the total co-occurrence count across all training documents.

After training, edges with weight < `pruning_min_weight`, count < `pruning_min_cooc`, or document frequency < `pruning_min_doc_freq` are removed.

### Discriminativeness

For each node `v` and category `c`:

```
P(v | c) = (df_c(v) + ε) / (N_c + ε)
P(v)     = (df(v)   + ε) / (N    + ε)

disc(v, c) = log( P(v | c) / P(v) )
           = log(P(v | c)) - log(P(v))
```

where `ε = pmi_smoothing` (default 1e-12).

`disc(v, c)` is positive when `v` is more frequent in class `c` than in the corpus overall, and negative when `v` is rarer in class `c` than average. Magnitude reflects strength of association.

### Uncertainty

A node seen in very few class-`c` documents has an unreliable discriminativeness estimate. Uncertainty dampens contributions from such nodes:

```
uncertainty(v, c) = 1 - exp( -df_c(v) / λ )
```

where `λ = uncertainty_lambda` (default 5.0).

**Properties:**
- `df_c(v) = 0` → `uncertainty = 0` → zero contribution (node never seen in this class)
- `df_c(v) = 5` → `uncertainty = 0.63`
- `df_c(v) → ∞` → `uncertainty → 1`

**Minimum uncertainty floor:** To prevent tokens with `uncertainty = 0` from contributing nothing even when structurally important:
- Regular structural tokens (not bigrams) with `uncertainty = 0`: clamped to `min_unc = 0.10`
- Word bigrams (`__bg_*` nodes): clamped to `min_unc = 0.02`

This ensures structural presence tokens that appear for the first time at inference still contribute a small, appropriately downweighted signal.

### IDF

```
idf(v) = log( N / (df(v) + 1) )
```

Standard inverse document frequency. Rare tokens (high IDF) that appear are more informative. Combined with discriminativeness, IDF weights the contribution by both rarity and category specificity.

---

## Stage 3: Phrase Discovery

### Purpose

Individual tokens are often insufficient to capture meaning. "Free" is weakly spam-indicative alone; "free prize" is strongly so. LexiDecay v2 discovers statistically significant multi-token phrases and adds them as graph nodes.

### G² Statistic (Log-Likelihood Ratio)

For a candidate bigram `(u, v)`, the G² statistic tests whether `u` and `v` co-occur significantly more than expected by chance:

```
G²(u, v) = 2 × Σ  O_ij × log(O_ij / E_ij)
```

where the 2×2 contingency table counts documents containing:
- Both `u` and `v`
- `u` but not `v`
- `v` but not `u`
- Neither

A candidate bigram is accepted only if `G² > phrase_g2_threshold` (default 10.83, corresponding to p < 0.001 with df=1).

### NPMI Filter

Normalized Pointwise Mutual Information provides an additional association filter:

```
PMI(u, v)  = log( P(u, v) / (P(u) × P(v)) )
NPMI(u, v) = PMI(u, v) / (-log P(u, v))
```

NPMI is bounded in [-1, +1]. A threshold of 0.3 requires moderate excess co-occurrence beyond independence.

### Min Support Filter

Only bigrams appearing together in at least `phrase_min_support` documents are considered candidates. This prevents hapax-driven spurious phrases.

**Recommended: `phrase_min_support=8` for SMS spam.** At the default (3), low-frequency bigrams like `free__ringtone` (3 occurrences) pass G²/NPMI tests but represent noise. At 8, only well-attested phrase patterns qualify.

### Phrase Nodes

Accepted phrases are added to the RelationGraph as nodes with prefix `__phrase_` or using the underscore-joined form. They accumulate their own `df_c`, `disc`, and `uncertainty` statistics from training documents where they appear. At inference, the tokenizer detects phrase boundaries and activates these nodes.

---

## Stage 4: Evidence Aggregation

The raw score for class `c` given input document `d` is the weighted sum of five evidence types:

```
score(d, c) = w_1 × E1(d,c) + w_2 × E2(d,c) + w_3 × E3(d,c)
            + w_4 × E4(d,c) + w_5 × E5(d,c)
```

### Contribution Formula

For any node `v` contributing to class `c`:

```
contrib(v, c) = disc(v, c) × idf(v) × uncertainty(v, c)
```

### E1: Direct Evidence

Direct token discriminativeness. For each token `t` in the input that exists as a graph node:

```
E1(d, c) = Σ_{t ∈ d ∩ graph}  contrib(t, c)
```

Weight: 1.0 (baseline evidence source)

### E2: Phrase Evidence

For each phrase node `p` matched in the input document:

```
E2(d, c) = phrase_boost × Σ_{p ∈ phrases(d) ∩ graph}  contrib(p, c)
```

Weight: 1.5 (× `phrase_boost`, default 1.5 × 1.5 = 2.25 effective)

Phrase evidence is upweighted because multi-token co-occurrences are more reliable indicators of semantic category than individual tokens.

### E3: Context Evidence

For each input token `t`, its local window tokens `context(t)` also contribute evidence. This captures associations like "call" near "free" being more spam-like than "call" alone:

```
E3(d, c) = Σ_{t ∈ d}  Σ_{u ∈ context(t) ∩ graph}  contrib(u, c)
```

Weight: 0.25 (reduced to prevent phantom connections: ham texts about calling family shouldn't activate spam hub nodes heavily)

### E4: Propagation Evidence

Bounded Personalized PageRank (PPR) activates graph neighbors of input tokens, capturing indirect associations. See [Stage 5](#stage-5-graph-propagation-bounded-ppr) for the propagation algorithm.

```
E4(d, c) = Σ_{(node, strength) ∈ propagation(d, c)}  strength × contrib(node, c)
```

Weight: 0.10 (strongly downweighted to prevent noise from hub nodes that are adjacent to many token types)

### E5: Interaction Evidence

For pairs of input tokens `(t_i, t_j)` that share a graph edge:

```
E5(d, c) = Σ_{(t_i, t_j) ∈ edges(d)}  contrib(edge(t_i,t_j), c)
```

Weight: 0.10

---

## Stage 5: Graph Propagation (Bounded PPR)

### Purpose

Propagation allows the model to activate graph nodes that are not directly present in the input but are strongly connected to input tokens via co-occurrence edges. For example, if "prize" is in the input and "winner" is a strongly connected neighbor, "winner"'s discriminativeness for spam can flow back as evidence.

### Algorithm

LexiDecay v2 uses **bounded Personalized PageRank (PPR)** rather than full random walk PPR for efficiency:

```
Initialize: activation[t] = 1.0  for all t ∈ input_tokens  (seed set S)

For depth = 1 to max_depth:
    For each active node v with activation[v] >= min_threshold:
        neighbors = top_k neighbors of v by edge weight
        For each neighbor u:
            activation[u] += decay × activation[v] × edge_weight(v, u)
        activation[v] × = restart_alpha   (PPR restart dampens seed nodes)

Result: all (node, activation) pairs with activation >= min_threshold
```

**Parameters:**
- `max_depth = 1`: at most 1 hop from input tokens (sufficient for SMS spam; deeper = more noise)
- `decay = 0.85`: activation multiplied by 0.85 per hop
- `top_k = 10`: expand at most 10 neighbors per node
- `restart_alpha = 0.15`: PPR restart probability
- `min_threshold = 0.005`: prune weak activations

### Propagation Seed Filter (`min_seed_disc`)

A critical finding: function words like "to", "for", "the" appear in both spam and ham and may have small positive `disc(spam)` values (e.g. +0.5). These words co-occur in the graph with spam hub nodes like `__free_cta__`. Without filtering, **every document** — including pure ham — activates `__free_cta__` via propagation through function words, inflating spam scores uniformly.

The `min_seed_disc` parameter addresses this:

```
if |disc(v, label)| < min_seed_disc:
    skip v as a propagation seed
```

Only tokens with substantial discriminativeness seed the propagation. Setting `min_seed_disc = 0.5` prevents function words from generating propagation noise while preserving propagation from genuinely discriminative tokens.

---

## Threshold Calibration

For binary classification, the class boundary is determined by a calibrated threshold `τ` rather than the argmax:

```python
predict(d) = positive_label  if  P(positive | d) > τ
             negative_label   otherwise
```

### Calibration Procedure

Given training data `(X_train, y_train)`:

1. Compute `P(positive | d)` for all training documents via `predict_proba()`
2. For each candidate `τ_i = i / 200` where `i ∈ {1, 2, ..., 199}`:
   - Apply threshold to get predictions
   - Compute spam F1 = 2·P·R / (P+R)
3. Store `τ* = argmax_{τ_i} F1(τ_i)` in `model._thresholds["spam"]`

### Why Threshold Calibration Outperforms Prior Correction Alone

With a 6.5:1 ham/spam imbalance, the softmax `P(spam | d)` is naturally suppressed. The calibrated threshold `τ*` effectively re-centers the decision boundary without requiring SMOTE or class weighting. In practice, `τ*` stabilizes around 0.979 across folds (std = 0.032), indicating the model is well-calibrated after softmax.

---

## Class Prior Correction

When `add_class_prior=True`, the log-prior `log(n_c / N)` is added to each class's raw score before softmax:

```
score_corrected(d, c) = score(d, c) + log(n_c / N)
```

where `n_c` is the number of training documents in class `c` and `N` is the total.

With 4,825 ham and 747 spam (N=5,572):
```
log(n_spam / N) = log(747 / 5572) = -2.007   (penalizes spam)
log(n_ham  / N) = log(4825 / 5572) = -0.143  (slight ham advantage)
```

This shifts the raw score by approximately -2 for spam predictions, requiring stronger spam evidence to classify positively. Combined with threshold calibration, this is effective for heavily imbalanced datasets.

---

## Hyperparameter Reference

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `window_size` | 5 | 2–10 | Co-occurrence window radius. Larger = richer edges, slower training. |
| `phrase_min_support` | 3 | 3–20 | Minimum documents for phrase candidacy. Raise for noisy corpora. |
| `phrase_g2_threshold` | 10.83 | 6.63–20 | G² significance cutoff. 10.83 = p<0.001. |
| `phrase_npmi_threshold` | 0.3 | 0.1–0.6 | NPMI strength filter. Higher = only strong phrase associations. |
| `max_phrase_length` | 2 | 2–3 | Maximum phrase tokens. 2 = bigrams only. |
| `propagation_depth` | 1 | 1–3 | PPR hops. 1 = only direct neighbors. |
| `propagation_decay` | 0.85 | 0.5–0.95 | Activation decay per hop. |
| `propagation_top_k` | 10 | 5–50 | Neighbors expanded per node. |
| `propagation_restart` | 0.15 | 0.05–0.5 | PPR restart probability. |
| `propagation_min_seed_disc` | 0.0 | 0.0–2.0 | Min seed discriminativeness. >0 filters function words. |
| `min_threshold` | 0.005 | 0.001–0.05 | Prune weak propagation activations. |
| `uncertainty_lambda` | 5.0 | 1.0–20.0 | Uncertainty saturation rate. |
| `pruning_min_weight` | 0.01 | 0.001–0.1 | Minimum retained edge weight. |
| `pruning_min_cooc` | 2.0 | 1–5 | Minimum co-occurrence count for edge retention. |
| `pruning_min_doc_freq` | 2 | 1–5 | Minimum document frequency for node retention. |
| `phrase_boost` | 1.5 | 1.0–3.0 | Multiplier on phrase evidence (on top of E2 weight). |
| `uncertainty_lambda` | 5.0 | 1.0–20.0 | Uncertainty saturation rate. Higher = nodes need more docs to become confident. |
| `use_structural_features` | True | — | Inject regex pseudo-tokens. Disable for clean prose corpora. |
| `add_class_prior` | False | — | Add log P(c) to scores. Enable for imbalanced datasets. |

---

## Key References

- Almeida, T.A., Gómez Hidalgo, J.M., and Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* DOCENG'11. — Dataset source.
- Manning, C.D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. — G² statistic, IDF, PMI.
- Page, L., Brin, S., Motwani, R., Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web.* Stanford Technical Report. — Personalized PageRank foundation.
- Bouma, G. (2009). *Normalized (Pointwise) Mutual Information in Collocation Extraction.* GSCL. — NPMI formulation.
