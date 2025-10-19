# 🧠 LexiDecay.js  
### Intelligent Text Classification in Pure JavaScript (No NPM, No Dependencies)

LexiDecay.js is a **browser-based text classification library** originally created in Python by **Mohammad Taha Gorji (mr-r0ot)** and ported to **pure JavaScript** for direct use in the browser — no Node.js, NPM, or external libraries required.

---

## 🚀 Features
- ✅ 100% **Vanilla JavaScript** — works directly in the browser  
- 🧩 **Add / Remove categories** dynamically  
- 📊 **Classify** any input text into the best matching category  
- ⚙️ Built-in **decay**, **IDF weighting**, and **common word reduction**  
- 💾 **Save & load** trained models using `localStorage`  
- 🌐 Supports any modern browser (Chrome, Firefox, Edge, etc.)

---

## 📦 Installation
Simply include the library in your HTML file:

```html
<script src="lexidecay.js"></script>
````

No build tools, no imports — ready to use.

---

## ✨ Basic Usage Example

```html
<script src="lexidecay.js"></script>
<script>
  // Create a new model
  const model = new LexiDecayModel();

  // Add categories with sample training text
  model.addCategory("Technology", [
    "AI, neural networks, JavaScript, programming, computers, innovation"
  ]);

  model.addCategory("Sports", [
    "football, basketball, tennis, championship, player, coach"
  ]);

  model.addCategory("Music", [
    "melody, song, guitar, concert, album, rhythm"
  ]);

  // Classify an input text
  const result = model.classify("I love playing football and watching sports highlights!");

  console.log("Classification result:", result);

  // Access the top predicted category
  console.log("Best match:", result.top[0]);
</script>
```

**Output example:**

```js
Classification result: {
  scores: { Technology: 0.01, Sports: 0.98, Music: 0.01 },
  probs: { Technology: 0.01, Sports: 0.97, Music: 0.02 },
  top: ["Sports", 0.98, 0.97]
}
```

---

## ⚙️ Advanced Options for `.classify()`

You can fine-tune classification behavior using parameters:

```js
model.classify("some text", {
  decay: 0.5,                // token frequency influence (0–1)
  use_idf: true,             // apply inverse document frequency
  auto_common_reduce: true,  // automatically reduce common-word weights
  common_decay: 0.7,         // how aggressively common words are reduced
  min_common_mult: 0.05,     // minimum allowed multiplier for common words
  ignore_input_repetitions: false // ignore repeated tokens in input
});
```

| Parameter                  | Type        | Default | Description                                 |
| -------------------------- | ----------- | ------- | ------------------------------------------- |
| `decay`                    | float (0–1) | `0.5`   | Controls how repeated words influence score |
| `use_idf`                  | bool        | `false` | Uses inverse document frequency weighting   |
| `auto_common_reduce`       | bool        | `true`  | Automatically reduce very common tokens     |
| `common_decay`             | float       | `0.7`   | Strength of common-word penalty             |
| `min_common_mult`          | float       | `0.05`  | Minimum weight for any token                |
| `ignore_input_repetitions` | bool        | `false` | Treat repeated words as one                 |

---

## 💾 Save & Load Model (localStorage)

```js
// Save model to localStorage
model.saveModelToLocalStorage("my_lexidecay");

// Later or on next session
const restored = new LexiDecayModel();
restored.loadModelFromLocalStorage("my_lexidecay");

// You can now classify again
const res = restored.classify("I love neural networks and coding!");
console.log(res.top);
```

---

## 🧠 Internal Logic

* **Tokenization:** Simple regex tokenizer for English text
* **Frequency statistics:** Builds token stats per category
* **Softmax normalization:** Converts raw scores to probabilities
* **Common-word penalty:** Reduces bias toward frequent words
* **Decay factor:** Controls diminishing returns for repetitive tokens

---

## 🧪 Example: Sentiment Analysis (Mini Demo)

```html
<script src="lexidecay.js"></script>
<script>
  const sentiment = new LexiDecayModel();

  sentiment.addCategory("Positive", [
    "happy joy love fantastic good great excellent wonderful amazing"
  ]);

  sentiment.addCategory("Negative", [
    "sad terrible bad horrible awful hate pain depressed"
  ]);

  const text = "This movie was absolutely wonderful and I loved it!";
  const result = sentiment.classify(text);

  alert(`Detected sentiment: ${result.top[0]} (${(result.top[2]*100).toFixed(1)}%)`);
</script>
```

---

## 🧩 API Summary

| Method                            | Description                                |
| --------------------------------- | ------------------------------------------ |
| `addCategory(label, textOrArray)` | Add a labeled category for training        |
| `removeCategory(label)`           | Remove a category                          |
| `classify(text, options)`         | Classify input and return detailed results |
| `saveModelToLocalStorage(key)`    | Save model to browser localStorage         |
| `loadModelFromLocalStorage(key)`  | Load model from localStorage               |
| `_buildStats()`                   | Recalculate token statistics manually      |

---

## 🔖 License

MIT License © 2025 — *Mohammad Taha Gorji (mr-r0ot)*
Free for all personal and commercial use.

---

## 🌟 Credits

Originally developed in **Python**, ported to **JavaScript** by **Mohammad Taha Gorji**.
LexiDecay brings lightweight NLP classification to the browser world — simple, elegant, and dependency-free.