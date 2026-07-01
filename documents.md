# LexiDecay v2 — Usage Documentation

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Constructor Parameters](#constructor-parameters)
4. [Training: `fit()` and `partial_fit()`](#training)
5. [Threshold Calibration](#threshold-calibration)
6. [Prediction](#prediction)
7. [Explainability: `classify()`](#explainability)
8. [Persistence: `save()` / `load()`](#persistence)
9. [Recommended Config for SMS Spam](#recommended-config)
10. [Interpreting Evidence Output](#interpreting-evidence-output)
11. [Graph Diagnostics](#graph-diagnostics)

---

## Installation

```bash
pip install LexiDecay
```

**Optional dependencies:**

| Package | Purpose |
|---|---|
| `scikit-learn` | `StratifiedKFold` cross-validation scripts |
| `matplotlib` | Chart generation in evaluation scripts |

---

## Quick Start

```python
from lexidecay import LexiDecayV2

# 1. Create model
model = LexiDecayV2(phrase_min_support=8, add_class_prior=True)

# 2. Train
model.fit(X_train, y_train)  # X_train: List[str], y_train: List[str]

# 3. Calibrate decision threshold (recommended for imbalanced datasets)
model.calibrate_threshold(X_train, y_train, positive_label="spam")

# 4. Predict
labels = model.predict(X_test)              # List[str]
probas = model.predict_proba(X_test)        # List[Dict[str, float]]

# 5. Explain a single prediction
result = model.classify("Win a FREE iPhone now!")
print(result.explanation)
print(result.top_positive_features)

# 6. Save and reload
model.save("spam_model.pkl")
model2 = LexiDecayV2.load("spam_model.pkl")
```

---

## Constructor Parameters

```python
LexiDecayV2(
    window_size:               int   = 5,
    phrase_g2_threshold:       float = 10.83,
    phrase_npmi_threshold:     float = 0.3,
    phrase_min_support:        int   = 3,
    max_phrase_length:         int   = 2,
    propagation_depth:         int   = 1,
    propagation_decay:         float = 0.85,
    propagation_top_k:         int   = 10,
    propagation_restart:       float = 0.15,
    min_threshold:             float = 0.005,
    evidence_weights:          Optional[Dict[str, float]] = None,
    phrase_boost:              float = 1.5,
    pruning_min_weight:        float = 0.01,
    pruning_min_cooc:          float = 2.0,
    pruning_min_doc_freq:      int   = 2,
    pmi_smoothing:             float = 1e-12,
    uncertainty_lambda:        float = 5.0,
    use_structural_features:   bool  = True,
    structural_presence_repeat: int  = 1,
    structural_absence_repeat:  int  = 1,
    use_bigrams:               bool  = True,
    add_class_prior:           bool  = False,
    propagation_min_seed_disc: float = 0.0,
)
```

### Core Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_size` | int | 5 | Co-occurrence window radius. Tokens within ±window_size positions are considered co-occurring. |
| `phrase_min_support` | int | 3 | Minimum number of documents a bigram must appear in to be considered as a phrase. **Raise to 8 for SMS spam to remove spurious low-frequency phrases.** |
| `phrase_g2_threshold` | float | 10.83 | G² (log-likelihood ratio) threshold for phrase acceptance. 10.83 corresponds to p < 0.001 (chi-squared, df=1). |
| `phrase_npmi_threshold` | float | 0.3 | NPMI threshold for phrase acceptance (range: -1 to +1). 0.3 requires moderate co-occurrence excess. |
| `max_phrase_length` | int | 2 | Maximum phrase length in tokens. 2 = bigrams only, 3 = up to trigrams. |
| `add_class_prior` | bool | False | Add `log(n_c / N)` to raw scores before softmax. Corrects for class imbalance. Useful when classes are heavily skewed (e.g. 85% ham / 15% spam). |

### Propagation Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `propagation_depth` | int | 1 | Maximum number of graph hops from seed nodes. 1 is recommended for speed and precision; 2–3 increases recall at risk of noise. |
| `propagation_decay` | float | 0.85 | Activation multiplier per hop. Evidence halves roughly every 5 hops. |
| `propagation_top_k` | int | 10 | Maximum neighbors to expand per node per hop. |
| `propagation_restart` | float | 0.15 | PPR restart probability α. Higher α = stronger anchoring to input tokens. |
| `min_threshold` | float | 0.005 | Prune propagation activations below this strength to reduce noise. |
| `propagation_min_seed_disc` | float | 0.0 | Only propagate from tokens whose `|disc(v, label)| >= min_seed_disc`. Setting > 0 prevents weakly-discriminative function words (e.g. "to", "for") from activating spam hub nodes and inflating spam scores for ham documents. |

### Structural Feature Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_structural_features` | bool | True | Inject regex-based pseudo-tokens before tokenization. Dramatically improves precision on SMS/email spam by capturing structural patterns language-independently. |
| `structural_presence_repeat` | int | 1 | How many times to repeat each pseudo-token when present (amplifies its discriminative signal). |
| `structural_absence_repeat` | int | 1 | How many times to inject the absence pseudo-token (`__NO_LONG_NUM__`) when the pattern is absent. Helps the model learn from absence of spam patterns. |
| `use_bigrams` | bool | True | Whether to compute structural feature bigrams (token pairs including pseudo-tokens). |

**Structural pseudo-tokens extracted:**

| Token | Pattern | Spam frequency | Ham frequency |
|---|---|---|---|
| `__LONG_NUM__` | `\b\d{8,15}\b` | 50.7% | 0.07% |
| `__SHORT_CODE__` | `\b\d{4,6}\b` | 51.5% | 0.13% |
| `__MONEY__` | `[£$€]\d+` or `\d+p` | 21.8% | 0.14% |
| `__URL__` | `https?://` or `www.` | 14.2% | 0.04% |
| `__CAPS_WORD__` | `\b[A-Z]{3,}\b` | 74.2% | 7.83% |
| `__DATE__` | `\d{1,2}/\d{1,2}/\d{2,4}` | — | — |

### Graph Pruning Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pruning_min_weight` | float | 0.01 | Minimum edge weight retained after training. Removes very weak co-occurrence edges. |
| `pruning_min_cooc` | float | 2.0 | Minimum co-occurrence count for edge retention. Removes edges seen in fewer than 2 documents. |
| `pruning_min_doc_freq` | int | 2 | Minimum document frequency for node retention. Removes tokens seen in only 1 document. |

### Evidence Weight Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `evidence_weights` | dict | See below | Per-source weight multipliers. |
| `phrase_boost` | float | 1.5 | Additional multiplier applied to phrase evidence only. |

Default evidence weights (classifier-specific, tuned for precision):
```python
{
    "direct":      1.0,
    "phrase":      1.5,
    "context":     0.25,
    "propagation": 0.10,
    "interaction": 0.10,
}
```

### Other Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `uncertainty_lambda` | float | 5.0 | Controls how quickly uncertainty saturates with document frequency. `uncertainty = 1 - exp(-df_c / λ)`. Higher λ = nodes need more documents before they become fully confident. |
| `pmi_smoothing` | float | 1e-12 | Additive smoothing for PMI/NPMI computation to avoid log(0). |

---

## Training

### `model.fit(X, y)`

Full training from scratch. Resets the graph.

```python
X_train = ["Win a FREE prize now!", "Call me at home tonight", ...]
y_train = ["spam", "ham", ...]
model.fit(X_train, y_train)
```

- Processes all documents through the token engine and builds the RelationGraph
- Runs phrase discovery (G²/NPMI filtering) to add phrase nodes
- Computes discriminativeness and uncertainty for all nodes
- Prunes low-weight edges
- Returns `self` (chainable)

### `model.partial_fit(X, y)`

Incremental online update. Adds new documents to the existing graph without resetting.

```python
model.partial_fit(new_texts, new_labels)
```

Useful for streaming data or retraining on new examples without full retraining. Note: phrase discovery is not re-run after `partial_fit`; only direct token graph statistics are updated.

---

## Threshold Calibration

For binary classification tasks (e.g. spam vs. ham), the default argmax decision rule often produces suboptimal precision/recall balance, especially with imbalanced classes.

### `model.calibrate_threshold(X, y, positive_label, metric)`

```python
best_threshold, best_f1 = model.calibrate_threshold(
    X_train,
    y_train,
    positive_label="spam",   # the class to optimize for
    metric="f1",             # "f1" is the only supported metric
)
print(f"Calibrated threshold: {best_threshold:.3f}")
print(f"Training spam F1:     {best_f1:.4f}")
```

- Scans 200 threshold candidates in (0, 1)
- For each candidate: applies `predict_proba()` and computes spam F1 on the training set
- Stores the best threshold in `model._thresholds["spam"]`
- Subsequent calls to `model.predict()` automatically use this threshold

**Always call `calibrate_threshold()` on the training set before evaluating.** The calibrated threshold is serialized with `model.save()`.

---

## Prediction

### `model.predict(X) -> List[str]`

```python
labels = model.predict(["Win a prize!", "See you tomorrow"])
# ["spam", "ham"]
```

If `calibrate_threshold()` was called, uses the stored threshold for the positive class. Otherwise uses argmax over softmax probabilities.

### `model.predict_proba(X) -> List[Dict[str, float]]`

```python
probas = model.predict_proba(["Win a prize!"])
# [{"spam": 0.9991, "ham": 0.0009}]
```

Returns softmax-normalized probabilities for each class.

### `model.batch_predict(texts, n_jobs) -> List[PredictionResult]`

```python
results = model.batch_predict(texts, n_jobs=4)
```

Parallel batch classification using `ProcessPoolExecutor`. Returns full `PredictionResult` objects with evidence breakdowns. Use `n_jobs=-1` to use all available CPU cores.

---

## Explainability

### `model.classify(text) -> PredictionResult`

Returns a full `PredictionResult` dataclass for a single document:

```python
result = model.classify("WINNER!! Claim your £900 prize. Call 09061701461")
```

**`PredictionResult` fields:**

| Field | Type | Description |
|---|---|---|
| `predicted_label` | str | Final predicted class |
| `probability` | float | Confidence for the predicted label (0–1) |
| `scores` | Dict[str, float] | Raw (pre-softmax) scores per class |
| `probabilities` | Dict[str, float] | Softmax probabilities per class |
| `tokens` | List[Token] | Tokenized input with positions |
| `phrases_found` | List[str] | Phrase node IDs matched in this document |
| `evidence` | Dict[str, CategoryEvidence] | Per-class evidence breakdown |
| `activated_paths` | List[PropagationPath] | Full propagation trace (seed → path → strength) |
| `top_positive_features` | List[FeatureContribution] | Top 10 features supporting predicted label |
| `top_negative_features` | List[FeatureContribution] | Top 10 features opposing predicted label |
| `explanation` | str | Human-readable summary |
| `inference_time_ms` | float | Inference time in milliseconds |

**Example explanation output:**
```
Prediction: 'spam'  (confidence: 99.9%)

Evidence for spam (raw score: +12.47):
  [direct]      winner        disc=+3.21  idf=4.88  unc=0.97
  [structural]  __LONG_NUM__  disc=+5.96  idf=6.10  unc=0.99
  [structural]  __MONEY__     disc=+4.78  idf=5.22  unc=0.98
  [phrase]      free__prize   disc=+2.84  idf=5.11  unc=0.95

Evidence for ham (raw score: -12.47):
  [direct]      winner        disc=-3.21  idf=4.88  unc=0.97
```

### `FeatureContribution` fields

| Field | Description |
|---|---|
| `token_or_edge` | Token, phrase ID, or edge identifier |
| `evidence_type` | `"direct"`, `"phrase"`, `"context"`, `"propagation"`, or `"interaction"` |
| `label` | The class this contribution is for |
| `contribution` | Signed contribution to the raw score |
| `explanation` | Human-readable description |

---

## Persistence

### `model.save(path)`

```python
model.save("models/spam_classifier.pkl")
```

Serializes the full model state to a pickle file, including:
- RelationGraph (all nodes, edges, statistics)
- Calibrated thresholds (`_thresholds`)
- All hyperparameters (`_config`)

### `LexiDecayV2.load(path)`

```python
model = LexiDecayV2.load("models/spam_classifier.pkl")
labels = model.predict(new_texts)
```

---

## Recommended Config

### SMS Spam (SMS Spam Collection dataset)

```python
model = LexiDecayV2(
    window_size=5,
    phrase_min_support=8,         # reduces spurious low-frequency spam phrases
    phrase_g2_threshold=10.83,    # p < 0.001
    phrase_npmi_threshold=0.3,
    propagation_depth=1,          # 1 hop sufficient; deeper = more noise
    pruning_min_weight=0.01,
    pruning_min_cooc=2.0,
    pruning_min_doc_freq=2,
    use_structural_features=True, # __LONG_NUM__, __CAPS_WORD__, __MONEY__ critical
    structural_presence_repeat=1,
    structural_absence_repeat=1,
    use_bigrams=True,
    add_class_prior=True,
)
model.fit(X_train, y_train)
model.calibrate_threshold(X_train, y_train, positive_label="spam")
```

**Why `phrase_min_support=8`:** At the default value of 3, low-frequency bigrams like `free__ringtone` (seen 3 times) get promoted to spam phrase nodes, producing spurious spam signals. Raising to 8 limits phrase nodes to statistically robust patterns, reducing false positives from 3 to 2 on the 80/20 split.

### General Multi-Class Text Classification

```python
model = LexiDecayV2(
    phrase_min_support=5,
    use_structural_features=False,  # disable if text is clean prose
    add_class_prior=True,
    propagation_depth=2,            # deeper propagation for rich text
)
model.fit(X_train, y_train)
# No threshold calibration needed for multi-class (use argmax)
```

---

## Interpreting Evidence Output

The raw score for class `c` is the sum of five evidence components, each weighted:

```
score(c) = Σ_E  weight_E × evidence_E(c)
```

**Evidence types:**

| Type | Weight (default) | What it captures |
|---|---|---|
| `direct` | 1.0 | Token-level discriminativeness: how much does seeing this word shift probability toward class c? |
| `phrase` | 1.5 × phrase_boost | Multi-token phrase discriminativeness (stronger signal, higher weight) |
| `context` | 0.25 | Evidence from tokens in the local window around each input token |
| `propagation` | 0.10 | Evidence from graph neighbors reachable via bounded PPR |
| `interaction` | 0.10 | Evidence from edges between pairs of input tokens |

**Contribution formula (per token v, class c):**
```
contribution(v, c) = disc(v, c) × idf(v) × uncertainty(v, c)
```

- `disc(v, c)` — positive if v is more common in c than average; negative otherwise
- `idf(v)` — IDF weight (rare tokens count more)
- `uncertainty(v, c)` — 0 when the model has seen v in very few class-c documents; 1 when well-attested

**Reading the output:** A large positive raw score means many high-discriminativeness tokens point toward this class. Negative contributions from tokens common in the other class will reduce the score. The softmax converts raw scores to probabilities.

---

## Graph Diagnostics

```python
stats = model.get_graph_stats()
print(stats)
# {
#   "total_docs": 4457,
#   "total_nodes": 9234,
#   "total_edges": 18472,
#   "labels": ["ham", "spam"],
#   "docs_per_label": {"ham": 3868, "spam": 589},
#   "phrase_nodes": 393,
#   "structural_nodes": 12,
# }
```

Use `get_graph_stats()` to verify the graph was built as expected. Key indicators:
- `phrase_nodes` should be in the hundreds (not thousands) with `phrase_min_support=8`
- `total_edges` should be in the tens of thousands for a typical SMS dataset
- `structural_nodes` should match the number of enabled pseudo-token types
