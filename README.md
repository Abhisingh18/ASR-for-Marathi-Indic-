<div align="center">

# 🎙️ Marathi ASR — Encoder & LLM Comparison

### SLAM-ASR for Marathi + English code-switched speech recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Marathi%20%2B%20English-orange)](docs/DATASET.md)
[![Encoders](https://img.shields.io/badge/Encoders%20compared-6-blue)](docs/ENCODERS.md)
[![Test sets](https://img.shields.io/badge/Test%20sets-7-blueviolet)](docs/DATASET.md)
[![Framework](https://img.shields.io/badge/Framework-SLAM--LLM-lightgrey)](https://github.com/ddlBoJack/SLAM-LLM)

A controlled comparison of **6 speech-encoder / LLM combinations** for Marathi+English
code-switched ASR, fine-tuned with LoRA on top of frozen encoders and frozen LLMs, evaluated on
**7 benchmark test sets**.

</div>

---

## 📋 Table of contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Models compared](#-models-compared)
- [Results](#-results)
- [Repository layout](#-repository-layout)
- [Quickstart](#-quickstart)
- [Documentation](#-documentation)
- [License & acknowledgements](#-license--acknowledgements)

## 🔭 Overview

Every model here follows the same **SLAM-ASR** recipe (frozen encoder → linear projector → frozen
LLM + LoRA), trained on the same 526k-utterance Marathi+English bilingual dataset, with the same
optimizer, batch size, and LoRA config. Only the **speech encoder** — and, for one run, the
**LLM** — changes between runs, so any WER difference is attributable to that swap alone.

| | |
|---|---|
| 🗣️ Languages | Marathi, English (code-switched) |
| 🧩 Encoders compared | data2vec-AQC (FT & SSL), ccc-wav2vec2.0 (FT), Whisper large-v3, XEUS |
| 🤖 LLMs compared | Gemma-3-4B-IT, Sarvam-1 |
| 📊 Test sets | CommonVoice, FLEURS, IndicTTS, Kathbath (+noisy), MUCS, R12 |
| 🏋️ Training data | 526,134 utterances, ~1,180 hours |

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Marathi+English audio] --> B[Frozen Speech Encoder]
    B --> C[Linear Projector<br/>downsample x5]
    C --> D[LLM + LoRA<br/>r=8, alpha=32]
    P["Prompt: transcribe + prev_context<br/>+ code-switch language note"] --> D
    D --> E[Marathi/English transcript]
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## 🧩 Models compared

| # | Encoder | LLM | Marathi fine-tuned encoder? | Card |
|---|---|---|:---:|---|
| 1 | data2vec-AQC | Gemma-3-4B-IT | ✅ | [`results/data2vec-FT_Gemma3`](results/data2vec-FT_Gemma3/README.md) |
| 2 | ccc-wav2vec2.0 | Gemma-3-4B-IT | ✅ | [`results/ccc-wav2vec2-FT_Gemma3`](results/ccc-wav2vec2-FT_Gemma3/README.md) |
| 3 | data2vec-AQC | Sarvam-1 | ✅ | [`results/data2vec-FT_Sarvam1`](results/data2vec-FT_Sarvam1/README.md) |
| 4 | data2vec-AQC (SSL) | Gemma-3-4B-IT | ❌ | [`results/data2vec-SSL_Gemma3`](results/data2vec-SSL_Gemma3/README.md) |
| 5 | Whisper large-v3 | Gemma-3-4B-IT | ❌ | [`results/Whisper_Gemma3`](results/Whisper_Gemma3/README.md) |
| 6 | XEUS | Gemma-3-4B-IT | ❌ | [`results/XEUS_Gemma3`](results/XEUS_Gemma3/README.md) |

Full encoder specs: [`docs/ENCODERS.md`](docs/ENCODERS.md)

## 📊 Results

**WER% on 6 complete test sets** (checkpoint: epoch 1 / step 32,000 for every model — controlled comparison):

| Test set | data2vec-FT+Gemma3 | ccc-wav2vec2+Gemma3 | Sarvam-1+FT | data2vec-SSL+Gemma3 | Whisper+Gemma3 | XEUS+Gemma3 |
|---|---|---|---|---|---|---|
| commonvoice | **22.49** | 24.16 | 23.53 | 25.23 | 29.27 | 35.80 |
| fleurs | **23.88** | 26.57 | 26.77 | 26.57 | 29.57 | 30.43 |
| indictts | **16.15** | 18.26 | 17.26 | 19.27 | 18.97 | 21.90 |
| kathbath | **21.29** | 23.74 | 22.73 | 25.04 | 26.99 | 31.96 |
| kathbath_noisy | **22.14** | 23.97 | 23.22 | 27.25 | 28.94 | 33.38 |
| mucs | 29.24 | 27.96 | 43.62 | 34.82 | **26.99** | 35.28 |

**Bold = best (lowest WER)**. A 7th, larger test set (R12, 6,282 utterances) is still decoding —
see [`results/SUMMARY.md`](results/SUMMARY.md) for its preliminary numbers and full takeaways.

## 📁 Repository layout

```
├── scripts/
│   ├── training/     6 training scripts (one per encoder/LLM combo)
│   └── decoding/     6 decode scripts + WER scorer + path-rewrite helper
├── src_patches/      2 small patches on top of the SLAM-LLM framework
├── results/          per-model WER files, config snapshot, and README card
├── docs/             architecture, encoders, dataset, training, decoding, resume-fix write-ups
└── CHANGELOG.md
```

## 🚀 Quickstart

```bash
# 1. Train (pick a script from scripts/training/)
GPU_INCLUDE=0,1,2,3 bash scripts/training/finetune_data2vec_gemma3_marathi_cs.sh

# 2. Rewrite test-set audio paths to your local layout
python3 scripts/decoding/rewrite_marathi_testset_paths.py

# 3. Decode
GPU=6 bash scripts/decoding/inference_data2vec_gemma3_marathi_7testsets.sh

# 4. Score
bash scripts/decoding/compute_wer_marathi_7testsets.sh
```

Full walkthrough: [`docs/REPRODUCE.md`](docs/REPRODUCE.md)

## 📚 Documentation

| Doc | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Model diagram, shared design |
| [`docs/ENCODERS.md`](docs/ENCODERS.md) | Every encoder's params, pretraining, fine-tune status |
| [`docs/DATASET.md`](docs/DATASET.md) | Train/dev/test data schema and stats |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Shared hyperparameters |
| [`docs/RESUME.md`](docs/RESUME.md) | The DeepSpeed checkpoint-resume fix |
| [`docs/DECODING.md`](docs/DECODING.md) | Decode pipeline + cross-host data transfer |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Software/hardware environment |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | End-to-end reproduction steps |
| [`docs/testsets/`](docs/testsets/) | One page per test set |

## 📄 License & acknowledgements

MIT licensed — see [`LICENSE`](LICENSE). This work builds on the
[SLAM-LLM](https://github.com/ddlBoJack/SLAM-LLM) framework and SPRING-INX encoder checkpoints;
full credits in [`docs/ACKNOWLEDGEMENTS.md`](docs/ACKNOWLEDGEMENTS.md).
