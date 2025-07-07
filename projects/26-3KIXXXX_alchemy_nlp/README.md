
# GPT-2 Fine-Tuning on Historical Alchemical Texts

This project involves fine-tuning a pre-trained GPT-2 language model on a cleaned dataset of historical alchemical writings to generate fluent and stylistically consistent text. The project includes dataset preprocessing, model training, evaluation (cross-entropy loss, perplexity), and prediction analysis.

---

##  Project Structure

```plaintext
├── data/
│   └── alchtexts.csv                  # Original raw dataset
│   └── clean_token_chunks.csv         # Cleaned and tokenized chunks
├── models/
│   └── fine_tuned_gpt2/               # Directory for fine-tuned model checkpoints
├── notebooks/
│   └── training_and_evaluation.ipynb  # Google Colab notebook with code
├── outputs/
│   ├── predictions.txt                # Generated text samples
│   └── metrics.json                   # Perplexity, loss, and accuracy stats
├── README.md                          # Project overview
```

---

##  Objective

To fine-tune GPT-2 on historical alchemical texts and evaluate its ability to generate coherent and stylistically relevant continuations. Emphasis is placed on:
- Cleaning non-linguistic characters (e.g., `√¢¬™`)
- Removing noise (e.g., image markers like “image unnumbered page”)
- Evaluating prediction strength using perplexity and token probabilities

---

##  1. Data Cleaning

### Actions:
- Removed digits, punctuation, URLs, and non-ASCII characters.
- Normalized whitespace and converted all text to lowercase.
- Split cleaned text into 128-token chunks for training.

### Example Code Snippet:
```python
def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove weird unicode chars
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

---

## 2. Model Training

### Model Used:
- `GPT-2 (124M)` from HuggingFace Transformers

### Training Details:
- Trained on cleaned token chunks
- Used `Trainer` API from HuggingFace
- Metrics logged with [Weights & Biases](https://wandb.ai)

---

##  3. Evaluation

### Key Metrics:
- **Cross-Entropy Loss**: 3.99
- **Perplexity**: 54.2  
  Perplexity is calculated as:

  \[
  \text{Perplexity} = e^{\text{loss}} = e^{3.99} \approx 54.2
  \]

### Interpretation:
- A perplexity of 54.2 implies ~1.85% probability for predicting the correct next token on average.
- While this may seem low, it's consistent with results on highly domain-specific and archaic text.

---

##  4. Inference & Prediction

### Example Prompt:
```python
prompt = "the alchemist discovered a way to"
```

### Generated Text:
```
... make and colour glass pastes enamels lakes and other curiosities written in italian by antonio neri...
```

### Top-5 Next Token Probabilities:
```
Token: the,    Probability: 12.89%
Token: he,     Probability: 11.18%
Token: it,     Probability: 5.25%
Token: they,   Probability: 3.82%
Token: there,  Probability: 2.55%
```

---

##  System Usage (WandB)

- GPU Usage: ~40–60%
- Memory Errors: None
- Power Consumption: 100–250W
- Stable Clock Speeds observed

---

##  Visualizations

- Token Chunk Distribution
- Token Length Histogram
- Prediction Probabilities
- System Monitoring Graphs (via wandb)

---

##  Presentation Goals

- Show how cleaning and chunking affect language modeling
- Highlight model’s prediction quality via:
  - Text generation examples
  - Probability ranking
  - Perplexity
- Interpret system resource use (e.g. GPU load)

---

##  Credits

- Original texts: Public domain alchemical writings
- Model: OpenAI GPT-2 via HuggingFace
- Monitoring: wandb.ai


