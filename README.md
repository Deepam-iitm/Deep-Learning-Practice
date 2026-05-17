# 🧠 Deep Learning Practice

A collection of competitive deep learning projects spanning computer vision, natural language processing, and speech recognition — each built and evaluated under real Kaggle competition constraints.

This repository is organized into dedicated branches, one per competition, each containing the solution notebook and a detailed write-up.

---

## 📂 Repository Structure

```
Deep-Learning-Practice/
├── main          ← You are here — repo overview
├── NPPE-1        ← Multi-Script Emotion Classification (NLP)
├── NPPE-2        ← Uyghur Speech Recognition (ASR)
└── NPPE-3        ← Low-Light Image Super-Resolution (CV)
```

Each branch contains:
- 📓 A Kaggle solution notebook (`.ipynb`)
- 📄 A `README.md` with full technical documentation

---

## 🏆 Projects

### [NPPE-1 — EmotiCode: Multi-Script Emotion Classification](https://github.com/Deepam-iitm/Deep-Learning-Practice/tree/NPPE-1)
> **Domain:** Natural Language Processing · **Metric:** Macro F1-Score

Fine-tuned **Google Gemma-3-1B-IT** with QLoRA to classify emotions across three linguistically diverse, low-resource Indian language scripts — Santali (Ol Chiki), Kashmiri (Arabic), and Manipuri (Meitei Mayek). Covers 6 emotion classes: `fear`, `happy`, `surprise`, `sad`, `anger`, `disgust`.

| Detail | Value |
|---|---|
| Base model | `google/gemma-3-1b-it` |
| Fine-tuning | QLoRA (4-bit NF4) via `trl.SFTTrainer` |
| Dataset | 11,960 multilingual text samples |
| Competition period | Jul 4–7, 2025 |

---

### [NPPE-2 — Uyghur Automatic Speech Recognition](https://github.com/Deepam-iitm/Deep-Learning-Practice/tree/NPPE-2)
> **Domain:** Speech Recognition · **Metric:** Character Error Rate (CER, lower is better)

Built an ASR pipeline for **Uyghur** — a low-resource Turkic language in Arabic script — using a community Whisper Small checkpoint fine-tuned on the THUGY-20 dataset, with Unicode-aware post-processing (NFKC normalization, punctuation stripping) to minimize CER.

| Detail | Value |
|---|---|
| Model | `ixxan/whisper-small-uyghur-thugy20` |
| Architecture | Whisper Small (12-layer encoder-decoder, 768d) |
| Dataset | ~23.95 hours of audio, 9,468 clips at 16kHz |
| Competition period | Jul 26–29, 2025 |

---

### [NPPE-3 — Low-Light Image Denoising & 4× Super-Resolution](https://github.com/Deepam-iitm/Deep-Learning-Practice/tree/NPPE-3)
> **Domain:** Computer Vision · **Metric:** PSNR (higher is better)

Trained an **RRDBNet** (Residual-in-Residual Dense Block Network — the ESRGAN backbone) from scratch to jointly denoise and upscale noisy low-light images by 4× using two sequential PixelShuffle stages. Trained end-to-end with L1 loss over 40 epochs on paired LR/HR image sets.

| Detail | Value |
|---|---|
| Model | RRDBNet (`nf=64`, `nb=23`, `gc=32`) |
| Scale factor | 4× (2× PixelShuffle × 2 stages) |
| Loss | L1 (Mean Absolute Error) |
| Competition period | Aug 20–23, 2025 |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, Hugging Face Transformers, PEFT, TRL |
| Computer Vision | torchvision, Pillow, scikit-image |
| Speech | torchaudio, Whisper |
| NLP | BitsAndBytes, Accelerate, Datasets |
| Evaluation | scikit-learn, skimage.metrics |
| Platform | Kaggle Notebooks · NVIDIA Tesla T4 GPU |

---

## 👤 Author

**Deepam** — [github.com/Deepam-iitm](https://github.com/Deepam-iitm)
