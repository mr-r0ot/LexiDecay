"""
LexiDecay - final implementation
Requirements:
    pip install numpy

coded by Mohammad Taha Gorji
Github: mr-r0ot
"""

import re
import pickle
from collections import Counter, defaultdict
from typing import Dict, Union, List, Any, Optional, Tuple
try:
    import numpy as np
except:
    print('ERROR: Please install Numpy with `pip install numpy`')
    exit()


# ---------------------------
# Tokenizer 
def _tokenize_en(text: str) -> List[str]:
    """Simple English tokenizer: keep words, numbers, apostrophes. Lowercases."""
    if text is None:
        return []
    return re.findall(r"[A-Za-z0-9']+", text.lower())

# ---------------------------
class LexiDecayModel:
    """
    LexiDecay classifier object.

    Usage:
      m = LexiDecayModel()
      m.add_category("side1", raw_text_or_list)
      m.add_category("side2", raw_text_or_list)
      result = m.classify(input_text)
      m.save_model("lexidecay.pkl")
      m2 = LexiDecayModel.load_model("lexidecay.pkl")
      m2.add_category("side3", "more text")
      # tune hyperparams:
      best = m.tune_hyperparams(X_labeled, y_labels)
    """

    def __init__(self):
        # store raw category contents (strings or lists) so we can save/load and extend
        self.categories_raw: Dict[str, Union[str, List[str]]] = {}
        # token counters per category
        self.cat_counters: Dict[str, Counter] = {}
        # doc freq (in how many categories a token appears)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        # global freq (sum of counts across categories)
        self.global_freq: Dict[str, int] = defaultdict(int)
        # labels ordered list (keeps consistent ordering)
        self.labels: List[str] = []
        # vocabulary set
        self.vocab: List[str] = []
        # internal flag to indicate whether stats are up-to-date
        self._built = False

    # ---------------------------
    def add_category(self, label: str, content: Union[str, List[str]]):
        """Add or replace a category. content can be string or list of strings.
           After adding, internal stats are rebuilt on demand."""
        self.categories_raw[label] = content
        self._built = False

    def remove_category(self, label: str):
        """Remove category if exists."""
        if label in self.categories_raw:
            del self.categories_raw[label]
            self._built = False

    def _build_stats(self):
        """(Re)build token counters, doc_freq, global_freq, vocab, labels."""
        self.cat_counters = {}
        self.doc_freq = defaultdict(int)
        self.global_freq = defaultdict(int)

        for label, content in self.categories_raw.items():
            if isinstance(content, list):
                txt = " ".join(map(str, content))
            else:
                txt = str(content)
            toks = _tokenize_en(txt)
            c = Counter(toks)
            self.cat_counters[label] = c
            for w, cnt in c.items():
                self.doc_freq[w] += 1
                self.global_freq[w] += cnt

        self.labels = list(self.cat_counters.keys())
        # stable sorted vocabulary for reproducibility
        self.vocab = sorted(self.global_freq.keys())
        self._built = True

    # ---------------------------
    def classify(
        self,
        input_text: str,
        decay: float = 0.5,
        use_idf: bool = False,
        auto_common_reduce: bool = True,
        common_decay: float = 0.7,
        min_common_mult: float = 0.05,
        ignore_input_repetitions: bool = False,
        eps: float = 1e-9
    ) -> Dict[str, Any]:
        """
        Classify input_text and return scores/probabilities/matches.

        Parameters: same semantics as previous conversation:
          - decay: 0..1  (0 => weight==freq, 1=> weight==1)
          - use_idf: use smoothed IDF across categories
          - auto_common_reduce: detect globally common words and reduce their weights
          - common_decay: intensity of reduction (0..1)
          - min_common_mult: minimum multiplier for common words
          - ignore_input_repetitions: if True, each unique word in input counts once (input freq=1)
        Returns:
          dict with keys: 'scores', 'probs', 'matches', 'top'
        """
        if not self._built:
            self._build_stats()

        if len(self.labels) == 0:
            raise ValueError("No categories available. Add categories with add_category().")

        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be between 0 and 1.")
        if not (0.0 <= common_decay <= 1.0):
            raise ValueError("common_decay must be between 0 and 1.")
        if not (0.0 < min_common_mult <= 1.0):
            raise ValueError("min_common_mult must be in (0,1].")

        L = len(self.labels)

        # IDF
        idf = {}
        if use_idf:
            for w, df in self.doc_freq.items():
                idf[w] = np.log((1.0 + L) / (1.0 + df)) + 1.0
        else:
            for w in self.doc_freq.keys():
                idf[w] = 1.0

        # common-word multiplier
        common_mult = {}
        if auto_common_reduce and len(self.global_freq) > 0:
            max_freq = max(self.global_freq.values()) if len(self.global_freq) > 0 else 1
            max_freq = max(max_freq, 1)
            for w, gf in self.global_freq.items():
                ratio = float(gf) / float(max_freq)
                mult = 1.0 - common_decay * ratio
                if mult < min_common_mult:
                    mult = min_common_mult
                common_mult[w] = mult
        else:
            for w in self.global_freq.keys():
                common_mult[w] = 1.0

        # tokenize input
        toks = _tokenize_en(input_text)
        if ignore_input_repetitions:
            input_counter = Counter(set(toks))
        else:
            input_counter = Counter(toks)

        scores = np.zeros(L, dtype=float)
        matches = {label: {"words": [], "input_count": 0, "cat_freq": 0} for label in self.labels}

        # iterate labels
        for idx, label in enumerate(self.labels):
            cat_counter = self.cat_counters[label]
            s = 0.0
            for w, in_freq in input_counter.items():
                cat_freq = cat_counter.get(w, 0)
                if cat_freq <= 0:
                    continue

                # base weight by category freq and decay
                if decay == 1.0:
                    w_weight = 1.0
                else:
                    w_weight = 1.0 + (1.0 - decay) * (cat_freq - 1.0)

                cm = common_mult.get(w, 1.0)
                idfm = idf.get(w, 1.0)

                contrib = w_weight * float(in_freq) * float(idfm) * float(cm)
                s += contrib

                matches[label]["words"].append(w)
                matches[label]["input_count"] += in_freq
                matches[label]["cat_freq"] += cat_freq

            scores[idx] = s
            if scores[idx] < eps:
                scores[idx] = 0.0

        # softmax -> probabilities
        if np.all(scores == 0):
            probs = np.ones(L) / L
        else:
            ex = np.exp(scores - np.max(scores))
            probs = ex / (np.sum(ex) + eps)

        scores_dict = {label: float(scores[i]) for i, label in enumerate(self.labels)}
        probs_dict = {label: float(probs[i]) for i, label in enumerate(self.labels)}
        best_idx = int(np.argmax(probs))
        top = (self.labels[best_idx], scores_dict[self.labels[best_idx]], probs_dict[self.labels[best_idx]])

        return {"scores": scores_dict, "probs": probs_dict, "matches": matches, "top": top}

    # ---------------------------
    def save_model(self, path: str):
        """
        Save model to a file (pickle). Stores:
          - categories_raw
          - everything necessary to resume (we'll rebuild counters on load)
        """
        payload = {
            "categories_raw": self.categories_raw
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        # NOTE: not saving counters directly to keep format stable; will rebuild on load
        print(f"Model saved to: {path}")

    @staticmethod
    def load_model(path: str) -> "LexiDecayModel":
        """Load model from file saved by save_model. Returns LexiDecayModel instance."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        m = LexiDecayModel()
        m.categories_raw = payload.get("categories_raw", {})
        m._built = False
        m._build_stats()
        print(f"Model loaded from: {path}")
        return m

    # --------------------------