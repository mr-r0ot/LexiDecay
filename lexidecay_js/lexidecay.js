/*
LexiDecay - Browser Version (No NPM)
Original Python algorithm by Mohammad Taha Gorji (mr-r0ot)
Converted to pure JavaScript for browsers.
*/

(function (global) {

  // ---------------------------
  // helper math functions (no external libs)
  function log(x) { return Math.log(x); }
  function exp(x) { return Math.exp(x); }
  function sum(arr) { return arr.reduce((a, b) => a + b, 0); }
  function max(arr) { return Math.max.apply(null, arr); }

  // ---------------------------
  // Tokenizer (English)
  function _tokenize_en(text) {
    if (!text) return [];
    const matches = text.toLowerCase().match(/[A-Za-z0-9']+/g);
    return matches ? matches : [];
  }

  // ---------------------------
  class LexiDecayModel {
    constructor() {
      this.categories_raw = {};   // label → string|list
      this.cat_counters = {};     // label → token frequencies
      this.doc_freq = {};         // token → in how many categories
      this.global_freq = {};      // token → total frequency
      this.labels = [];           // ordered label list
      this.vocab = [];
      this._built = false;
    }

    // ---------------------------
    addCategory(label, content) {
      this.categories_raw[label] = content;
      this._built = false;
    }

    removeCategory(label) {
      if (this.categories_raw[label]) {
        delete this.categories_raw[label];
        this._built = false;
      }
    }

    // ---------------------------
    _buildStats() {
      this.cat_counters = {};
      this.doc_freq = {};
      this.global_freq = {};

      for (const label in this.categories_raw) {
        const content = this.categories_raw[label];
        let txt;
        if (Array.isArray(content)) txt = content.join(" ");
        else txt = String(content);

        const toks = _tokenize_en(txt);
        const counter = {};
        for (const w of toks) counter[w] = (counter[w] || 0) + 1;
        this.cat_counters[label] = counter;

        for (const w in counter) {
          const cnt = counter[w];
          this.doc_freq[w] = (this.doc_freq[w] || 0) + 1;
          this.global_freq[w] = (this.global_freq[w] || 0) + cnt;
        }
      }

      this.labels = Object.keys(this.cat_counters);
      this.vocab = Object.keys(this.global_freq).sort();
      this._built = true;
    }

    // ---------------------------
    classify(input_text, {
      decay = 0.5,
      use_idf = false,
      auto_common_reduce = true,
      common_decay = 0.7,
      min_common_mult = 0.05,
      ignore_input_repetitions = false,
      eps = 1e-9
    } = {}) {

      if (!this._built) this._buildStats();

      if (this.labels.length === 0)
        throw new Error("No categories available. Add categories first.");

      if (!(decay >= 0 && decay <= 1))
        throw new Error("decay must be between 0 and 1.");
      if (!(common_decay >= 0 && common_decay <= 1))
        throw new Error("common_decay must be between 0 and 1.");
      if (!(min_common_mult > 0 && min_common_mult <= 1))
        throw new Error("min_common_mult must be in (0,1].");

      const L = this.labels.length;

      // ----- IDF -----
      const idf = {};
      if (use_idf) {
        for (const w in this.doc_freq)
          idf[w] = log((1.0 + L) / (1.0 + this.doc_freq[w])) + 1.0;
      } else {
        for (const w in this.doc_freq)
          idf[w] = 1.0;
      }

      // ----- common-word multiplier -----
      const common_mult = {};
      if (auto_common_reduce && Object.keys(this.global_freq).length > 0) {
        const max_freq = Math.max(...Object.values(this.global_freq), 1);
        for (const w in this.global_freq) {
          const gf = this.global_freq[w];
          const ratio = gf / max_freq;
          let mult = 1.0 - common_decay * ratio;
          if (mult < min_common_mult) mult = min_common_mult;
          common_mult[w] = mult;
        }
      } else {
        for (const w in this.global_freq)
          common_mult[w] = 1.0;
      }

      // ----- tokenize input -----
      const toks = _tokenize_en(input_text);
      const input_counter = {};
      if (ignore_input_repetitions) {
        const unique = Array.from(new Set(toks));
        for (const w of unique) input_counter[w] = 1;
      } else {
        for (const w of toks)
          input_counter[w] = (input_counter[w] || 0) + 1;
      }

      const scores = new Array(L).fill(0.0);
      const matches = {};
      for (const label of this.labels)
        matches[label] = { words: [], input_count: 0, cat_freq: 0 };

      // ----- iterate labels -----
      this.labels.forEach((label, idx) => {
        const cat_counter = this.cat_counters[label];
        let s = 0.0;
        for (const w in input_counter) {
          const in_freq = input_counter[w];
          const cat_freq = cat_counter[w] || 0;
          if (cat_freq <= 0) continue;

          const w_weight = (decay === 1.0) ? 1.0 : 1.0 + (1.0 - decay) * (cat_freq - 1.0);
          const cm = common_mult[w] ?? 1.0;
          const idfm = idf[w] ?? 1.0;

          const contrib = w_weight * in_freq * idfm * cm;
          s += contrib;

          matches[label].words.push(w);
          matches[label].input_count += in_freq;
          matches[label].cat_freq += cat_freq;
        }
        scores[idx] = s < eps ? 0.0 : s;
      });

      // ----- softmax -----
      const maxScore = max(scores);
      const expScores = scores.map(v => exp(v - maxScore));
      const sumExp = sum(expScores) + eps;
      const probs = expScores.map(v => v / sumExp);

      const scores_dict = {};
      const probs_dict = {};
      this.labels.forEach((label, i) => {
        scores_dict[label] = scores[i];
        probs_dict[label] = probs[i];
      });

      const best_idx = probs.indexOf(Math.max(...probs));
      const top = [this.labels[best_idx], scores_dict[this.labels[best_idx]], probs_dict[this.labels[best_idx]]];

      return { scores: scores_dict, probs: probs_dict, matches, top };
    }

    // ---------------------------
    saveModelToLocalStorage(key = "lexidecay_model") {
      const payload = { categories_raw: this.categories_raw };
      localStorage.setItem(key, JSON.stringify(payload));
      console.log(`Model saved to localStorage key: ${key}`);
    }

    loadModelFromLocalStorage(key = "lexidecay_model") {
      const data = localStorage.getItem(key);
      if (!data) {
        console.error(`No model found in localStorage for key: ${key}`);
        return;
      }
      const payload = JSON.parse(data);
      this.categories_raw = payload.categories_raw || {};
      this._built = false;
      this._buildStats();
      console.log(`Model loaded from localStorage key: ${key}`);
    }
  }

  // Attach to global
  global.LexiDecayModel = LexiDecayModel;

})(typeof window !== "undefined" ? window : this);
