# ⚡️ LexiDecay — The Adaptive Lexical Decay Classifier  
*By Mohammad Taha Gorji*

> **A blazing-fast, semi-supervised text classification algorithm** based on adaptive lexical weighting, frequency decay, and probabilistic scoring — all without any training or labeled dataset.
> **LexiDecay is a semi-supervised lexical weighting model for unstructured text. It classifies content by adaptive word-frequency decay and soft lexical scoring. Fast (O(n·m)), language-flexible, and training-free — ideal for topic classification, semantic filtering, and intent detection.**

---

## 🪶 Web
> See our site [https://mr-r0ot.github.io/LexiDecay](https://mr-r0ot.github.io/LexiDecay)
> 
> Luckily, we support **js** in addition to Python! [Full explanation and explanation](https://github.com/mr-r0ot/LexiDecay/tree/main/lexidecay_js)

## 🌌 Algorithm Philosophy & Core Idea

**LexiDecay** is inspired by the way human cognition evaluates language — not by rigid statistical training, but by dynamically weighting words according to their contextual importance and rarity.  
Instead of “learning” through countless iterations, **LexiDecay** *understands* by **measuring the gravitational pull of words** within conceptual clusters.

The algorithm analyzes each category’s text content, counts and weights its tokens, and applies a **decay function** that reduces the influence of overly common words (like “the”, “of”, “and”).  
During classification, it computes soft lexical similarities using adaptive decay, inverse document frequency, and a softmax-based probability normalization.

> 🧠 *Philosophically, LexiDecay reflects a cognitive model of understanding — flexible, intuitive, and progressively self-balancing.*

---

## 🧩 Scientific Position

| Category | Description |
|-----------|--------------|
| **Learning Type** | Semi-supervised lexical weighting |
| **Data Type** | Unstructured free text |
| **Complexity** | O(n × m) — *n = words in input, m = number of categories* |
| **Core Mechanism** | Adaptive word-frequency decay + soft lexical scoring |
| **Primary Fields** | NLP, cognitive AI, text understanding, knowledge extraction |

---

## 🚀 Real-World Applications

LexiDecay is suitable for a wide variety of language-intelligent systems:

- 🗂 **Topic classification** — Distinguish content across domains (e.g. science, art, politics).  
- 🎯 **Intent detection** — Recognize user intentions from text queries or chatbot messages.  
- 🧭 **Semantic filtering** — Filter or route information based on conceptual meaning.  
- 🪶 **Keyword-based reasoning** — Identify thematic or conceptual similarity.  
- 🧠 **Cognitive AI prototypes** — For lightweight, reasoning-like models without deep networks.  [See here](https://github.com/mr-r0ot/LexiDecay/tree/main/Examples/LexiDecay-Chatbot)

---

## ⚖️ Advantages Over Classical Models

| Feature | LexiDecay | Classical Models (Naive Bayes, TF-IDF, etc.) |
|----------|------------|-----------------------------------------------|
| **Training Required** | ❌ None — works instantly | ✅ Needs training |
| **Computation Speed** | ⚡ Extremely fast (O(n·m)) | 🐢 Often slower (training + inference) |
| **Flexibility** | 🧩 Add or remove categories freely | 🔒 Fixed to trained dataset |
| **Data Requirements** | 🌱 Works with few samples | 📊 Needs many labeled samples |
| **Common Word Handling** | 🪶 Auto frequency decay & adaptive weighting | ⚙️ Manual stopword removal |
| **Language Support** | 🌍 Fully language-independent | ⚠️ Usually language-specific |
| **Explainability** | 🔍 Transparent lexical logic | 🕳 Often black-box statistics |

> 💡 **LexiDecay** combines the interpretability of lexical systems with the adaptability of probabilistic models — no training, no fine-tuning, no waiting.

---

## ⚙️ Installation

```bash
pip install LexiDecay
````

That’s it! 🪄

---

## 🧱 Getting Started

Below is a full example of how to use **LexiDecay** from scratch.

```python
from LexiDecay import LexiDecayModel

# 1️⃣ Create a model
m = LexiDecayModel()

# 2️⃣ Add categories (each category can be a string or list of texts)
m.add_category("science", open("science.txt").read())
m.add_category("philosophy", open("philosophy.txt").read())

# 3️⃣ Classify new input
text = "Quantum theories explore the probabilistic structure of the universe."
result = m.classify(text)

print(result["top"])        # ('science', score, probability)
print(result["probs"])      # Probabilities for all categories
```

---

## 🧠 Function Reference & API Details

### 🔹 `add_category(label, content)`

Adds or replaces a category.

* `label`: `str` → name of the category
* `content`: `str` or `List[str]` → text data belonging to that category

Automatically rebuilds the internal vocabulary and frequency statistics.

---

### 🔹 `classify(input_text, decay=0.5, use_idf=False, auto_common_reduce=True, common_decay=0.7, min_common_mult=0.05, ignore_input_repetitions=False)`

Performs text classification and returns a dictionary with:

```python
{
  "scores": {label: float, ...},
  "probs": {label: float, ...},
  "matches": {label: {matched words, stats...}},
  "top": (best_label, score, probability)
}
```

#### Parameters:

| Parameter                  | Type  | Default | Description                                                                     |
| -------------------------- | ----- | ------- | ------------------------------------------------------------------------------- |
| `decay`                    | float | 0.5     | Controls how strongly frequent words lose influence (0 = linear, 1 = no decay). |
| `use_idf`                  | bool  | False   | Applies inverse-document-frequency weighting.                                   |
| `auto_common_reduce`       | bool  | True    | Automatically detects common words and lowers their impact.                     |
| `common_decay`             | float | 0.7     | Strength of reduction for common words.                                         |
| `min_common_mult`          | float | 0.05    | Minimum multiplier applied to frequent words.                                   |
| `ignore_input_repetitions` | bool  | False   | If True, counts each unique input word only once.                               |

---

### 🔹 `save_model(path)`

Saves the entire model (categories + data) into a `.pkl` file.

```python
m.save_model("lexidecay.pkl")
```

---

### 🔹 `load_model(path)`

Loads a model from a `.pkl` file.

```python
m2 = LexiDecayModel.load_model("lexidecay.pkl")
```


---

## 🌟 Why LexiDecay Feels Different

* **Human-like text perception:** adaptive decay mimics cognitive salience.
* **Instant deployability:** no model training — just plug and classify.
* **Infinite extendability:** add categories anytime, instantly rebuilt.
* **Compact and dependency-light:** only requires NumPy.
* **Transparent math:** pure lexical weighting, fully explainable results.

---

## 🧬 Example: Multi-category Classification

```python
from LexiDecay import LexiDecayModel
m = LexiDecayModel()
m.add_category("tech", ["AI","Model","AI algorithms", "neural networks", "deep learning"])
m.add_category("art", ["painting", "music", "creativity", "aesthetic beauty"])
m.add_category("sports", ["football", "strength", "competition"])

res = m.classify("New AI model beats humans at creative painting tasks.")
print(res)
# Output → ('tech', score, probability)
```

---

## 🧩 Citation

If you use **LexiDecay** in academic work, please cite:

> Mohammad Taha Gorji, *LexiDecay: Semi-supervised Lexical Decay Model for Adaptive Text Classification (2025)*

---

## 🔹 Examples

You can see **LexiDecay Examples** for some examples:

> [See here some examples](https://github.com/mr-r0ot/LexiDecay/tree/main/Examples)
> [See Pypi project](https://pypi.org/project/LexiDecay/)


---

## 🪄 Author

**Mohammad Taha Gorji**
Creator of *LexiDecay*
AI Researcher & Cognitive Systems Developer

---

## 🖤 License

Apache2 License © 2025 — Mohammad Taha Gorji
Open for research, education, and innovation.

---

> “LexiDecay doesn’t learn — it understands.” 🧠✨

---
---





## 🌌 The story behind this algorithm
The story behind this algorithm is that I had a collection of stories with A1 language level and one with C1 and I had to take an input and determine which level it was! And so I thought to myself why not try a simple way instead of very time-consuming, expensive and slow algorithms? And I wrote this simple and really dirty code :). I didn't expect it but it worked really well! And it generally recognized correctly but the more I investigated, the more repeated words like (the, ...) that were repeated a lot hurt the comparison. So I continued this way and developed this algorithm which is really super fast and also has a very high accuracy! I developed this algorithm and added and developed the repetition weight reduction rate + repetition reduction rate + similarity comparisons etc. and now it is this :)
```
data1 = open('A1.txt', encoding='UTF-8').read()
data2 = open('C1.txt', encoding='UTF-8').read()

input='''Beyond the physical cosmos, in dimensions inaccessible to ordinary perception'''
s1=0
s2=0

for i in data1.split():
    if i.lower() in input.lower():
        s1+=1
print(s1)

for i in data2.split():
    if i.lower() in input.lower():
        s2+=1
print(s2)


if s1==0:s1=1
elif s2==0:s2=1
if s1>s2:print(s1/s2)
else:print(s2/s1)
```
