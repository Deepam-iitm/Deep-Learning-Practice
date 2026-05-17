# 🎙️ NPPE-2: Uyghur Automatic Speech Recognition (ASR) Challenge

> **Kaggle Competition** · Jul 26–29, 2025 · Scored on Character Error Rate (lower is better)

Transcribe Uyghur-language audio clips using a pre-trained **Whisper Small** model fine-tuned on Uyghur speech, with Unicode-aware post-processing to clean the output text.

---

## 🎯 Problem Statement

Build an **Automatic Speech Recognition (ASR)** system that transcribes audio clips in the **Uyghur language** — a Turkic language written in the Arabic script and severely underrepresented in modern ASR research.

**Evaluation metric:** Character Error Rate (CER) — lower is better, 0.0 is perfect.

```
CER = (S + D + I) / N
```

| Symbol | Meaning |
|---|---|
| S | Substitutions |
| D | Deletions |
| I | Insertions |
| N | Total characters in ground-truth reference |

CER is computed as the Levenshtein (edit) distance at the **character level**, making it especially sensitive to script-specific errors — critical for Arabic-script Uyghur where diacritics and character forms matter.

---

## 🗂️ Dataset

### Files

| File / Folder | Description |
|---|---|
| `wavs/` | 9,468 `.wav` audio clips, named by UUID |
| `train.csv` | 7,574 labeled training samples |
| `test.csv` | 1,894 unlabeled test samples |
| `sample.csv` | Submission template with correct format |

### CSV Schemas

**`train.csv`**

| Column | Description |
|---|---|
| `ID` | Unique identifier for the audio clip |
| `filepath` | Relative path to the `.wav` file in `wavs/` |
| `transcription` | Ground-truth Uyghur text transcription |

**`test.csv`**

| Column | Description |
|---|---|
| `ID` | Unique identifier |
| `filepath` | Relative path to the `.wav` file in `wavs/` |

### Audio Characteristics

| Property | Value |
|---|---|
| Format | WAV |
| Channels | Mono (single-channel) |
| Sample rate | 16,000 Hz |
| Total duration | ~23.95 hours |
| Train samples | 7,574 |
| Test samples | 1,894 |
| Total clips | 9,468 |

---

## 🔍 Approach

This solution uses a **zero-shot / transfer inference** strategy: load a community Whisper checkpoint already fine-tuned on Uyghur speech, then run inference directly on the test set — no additional training required on the competition data.

**Why this works:**
- Uyghur is a low-resource language but has been addressed by community fine-tunes of OpenAI's Whisper
- The checkpoint `ixxan/whisper-small-uyghur-thugy20` was trained specifically on Uyghur audio, giving strong out-of-the-box CER
- The competition places **no restrictions** on models, compute, or external data, making pre-trained community checkpoints a valid and powerful strategy

**Pipeline at a glance:**

```
Audio file (.wav)
      │
[torchaudio.load]  →  waveform tensor
      │
[Stereo → Mono]    →  mean over channels (if needed)
      │
[WhisperProcessor] →  log-Mel spectrogram features
      │
[WhisperForConditionalGeneration.generate]
      │
[processor.batch_decode]  →  raw transcription string
      │
[Unicode post-processing]  →  cleaned Uyghur text
      │
submission.csv
```

---

## 🏗️ Model Architecture

**Checkpoint:** `ixxan/whisper-small-uyghur-thugy20`

A community fine-tune of OpenAI's **Whisper Small** on the THUGY-20 Uyghur speech dataset.

### Whisper Small — Architecture Summary

```
Input: log-Mel spectrogram (80 mel bins)

Encoder
├── Conv1d stem: 80 → 768 (kernel 3, stride 1)
├── Conv1d downsample: 768 → 768 (kernel 3, stride 2)
├── Positional embedding: 1500 positions × 768
└── 12 × WhisperEncoderLayer
    ├── SdpaAttention (Q/K/V: 768 → 768)
    ├── LayerNorm
    └── FFN: 768 → 3072 → 768

Decoder
├── Token embedding: 51865 vocab × 768
├── Positional embedding: 448 positions × 768
└── 12 × WhisperDecoderLayer
    ├── Masked self-attention (768)
    ├── Cross-attention to encoder output (768)
    └── FFN: 768 → 3072 → 768

Output projection: 768 → 51865 (vocab logits)
```

**Key specs:**

| Parameter | Value |
|---|---|
| Hidden size | 768 |
| Encoder layers | 12 |
| Decoder layers | 12 |
| FFN dimension | 3,072 |
| Vocabulary size | 51,865 |
| Max audio length | ~30 seconds (1,500 spectrogram frames) |
| Max decode length | 448 tokens |

---

## 🧹 Post-Processing

Raw Whisper output may contain encoding artifacts and punctuation that hurt CER. A Unicode-aware cleaning step is applied after initial decoding:

```python
def clean_text(text):
    # Fix any latin1/utf-8 mojibake
    fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    # Remove punctuation (keep word characters and whitespace)
    fixed = re.sub(r"[^\w\s]", "", fixed)
    # NFKC normalization: unify Unicode variants of the same character
    fixed = unicodedata.normalize("NFKC", fixed)
    # Lowercase and strip leading/trailing whitespace
    fixed = fixed.lower().strip()
    return fixed
```

**Why each step matters for Uyghur Arabic script:**

| Step | Reason |
|---|---|
| `latin1 → utf-8` re-encoding | Guards against mojibake from mixed-encoding audio metadata |
| Punctuation removal | CER penalizes extra characters; Uyghur ASR output often includes stray Arabic punctuation |
| NFKC normalization | Collapses compatibility variants (e.g., Arabic letter forms, ligatures) to canonical forms, preventing false character mismatches |
| Lowercase | Ensures case-insensitive comparison where applicable |

---

## 📊 Evaluation

CER is computed between the cleaned predicted transcription and the hidden ground-truth text. The character-level metric is particularly meaningful for Uyghur because:

- Arabic script encodes vowels as diacritics — a single wrong diacritic counts as one character error
- Connected letter forms mean segmentation errors cascade into multiple character errors
- Correct NFKC normalization is essential to avoid spurious mismatches on equivalent Unicode representations

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `transformers` | `WhisperProcessor`, `WhisperForConditionalGeneration` |
| `torchaudio` | Audio file loading and resampling |
| `torch` | GPU inference |
| `pandas` | CSV I/O |
| `tqdm` | Progress bars |
| `unicodedata` | NFKC Unicode normalization (stdlib) |
| `re` | Punctuation removal (stdlib) |

**Hardware:** Single NVIDIA Tesla T4 via Kaggle Notebooks with GPU acceleration enabled.
