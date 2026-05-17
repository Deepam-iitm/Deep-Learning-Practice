# 🌏 EmotiCode: Multi-Script Emotion Classification in Low-Resource Languages

> **Kaggle Competition** · Jul 4–7, 2025 · Scored on Macro F1-Score (higher is better)

Fine-tune Google's **Gemma-3-1B-IT** with QLoRA to classify emotions across three linguistically diverse, low-resource Indian language scripts — Santali (Ol Chiki), Kashmiri (Arabic), and Manipuri (Meitei Mayek).


## 🎯 Problem Statement

Classify the **emotion** expressed in short text samples written in three underrepresented Indian language scripts. The corruption challenge here is linguistic — these languages have extremely limited NLP resources, making standard transfer learning difficult.

**6 target emotion classes:**

| Class | Approx. Share |
|---|---|
| `fear` | 23.0% |
| `happy` | 17.2% |
| `surprise` | 15.8% |
| `sad` | 15.7% |
| `anger` | 15.3% |
| `disgust` | 13.4% |

**Evaluation metric:** Macro F1-Score — equal weight to all six classes regardless of frequency. Range: [0.0, 1.0], higher is better.

```
Final Score = Macro F1 = (1/6) × Σ F1_per_emotion_class
```

---

## 🗂️ Dataset

### Files

| File | Description |
|---|---|
| `competition_train.csv` | 7,176 labeled training samples |
| `competition_val.csv` | 2,392 labeled validation samples |
| `competition_test.csv` | 2,392 unlabeled test samples |
| `sample_submission.csv` | Submission template |

### Columns

| Column | Description |
|---|---|
| `id` | Unique integer sample ID |
| `Sentence` | Input text (in native script) |
| `language` | Language tag: `Santali`, `Kashmiri`, or `Manipuri` |
| `emotion` | Target label (absent in test set) |

### Language & Script Distribution

| Language | Script | Samples (all splits) | Share |
|---|---|---|---|
| Santali | Ol Chiki | ~4,252 | 35.6% |
| Kashmiri | Arabic | ~3,945 | 33.0% |
| Manipuri | Meitei Mayek | ~3,763 | 31.4% |

### Text Characteristics

- **Average length:** 101 characters
- **Median length:** 98 characters
- **Range:** 19–659 characters

---

## 🔍 Approach

The solution treats emotion classification as a **generative text task** rather than a traditional classification head approach. The model is prompted to produce a single emotion word, leveraging the instruction-following capability of Gemma-3-1B-IT fine-tuned via QLoRA on the training pairs.

**Pipeline overview:**

```
Raw multilingual text
        │
  [Prompt construction]  ← Sentence + Language tag
        │
  [Gemma-3-1B-IT + LoRA adapter]
        │
  [Generate (max 20 tokens)]
        │
  [Parse last word after prompt suffix]
        │
  Predicted emotion label
```

**Why generative classification?**
- Avoids adding a language-specific classification head
- Leverages the model's existing multilingual understanding
- Prompt engineering naturally constrains output to valid label space
- Works well for low-resource scripts where tokenization is imperfect

---

## 🏗️ Model & Fine-Tuning

### Base Model

```
google/gemma-3-1b-it
```
Gemma 3 1B Instruct — mandatory per competition rules. No other base models permitted.

### Quantization: QLoRA (4-bit NF4)

The model is loaded in 4-bit precision to fit within T4 GPU memory constraints:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # nested quantization for extra memory savings
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### LoRA Adapter

Parameter-efficient fine-tuning via `peft.LoraConfig` applied on top of the quantized base, trained with `trl.SFTTrainer` using a custom formatting function that wraps each training sample into the inference prompt template.

### Training Duration

Training ran for approximately **~4 hours 38 minutes** on a single Tesla T4 GPU (Kaggle Notebook, 15.6 GB VRAM).

---

## 💬 Prompt Design

A consistent prompt template is used for both training (via `formatting_func`) and inference:

```
Sentence: {sentence}, Language: {language}
What is the emotion expressed in this sentence?
Only respond with one word, choosing exactly one of [disgust, anger, sad, happy, fear, surprise].
Answer in lowercase letters only:
```

**Design decisions:**
- Including the `Language` field gives the model explicit script context, aiding cross-lingual transfer
- Constraining the output to a closed set of six words via the prompt reduces hallucination
- The suffix `Answer in lowercase letters only:` acts as a reliable split point during output parsing

**Output parsing:**

```python
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
emotion = decoded.split("Answer in lowercase letters only:")[-1].strip().lower()
```

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `transformers` | Model loading, tokenization, generation |
| `peft` | LoRA adapter — `LoraConfig`, `PeftModel` |
| `trl` | Supervised fine-tuning — `SFTTrainer` |
| `bitsandbytes` | 4-bit quantization (QLoRA) |
| `accelerate` | Device management and distributed support |
| `datasets` | HuggingFace dataset utilities |
| `torch` | Core deep learning framework |
| `pandas` | CSV I/O and data manipulation |
| `scikit-learn` | `f1_score`, `classification_report` |
| `tqdm` | Progress bars during inference |

**Hardware:** Single NVIDIA Tesla T4 (15.6 GB VRAM) via Kaggle Notebooks with GPU acceleration.
