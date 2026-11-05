# Telugu Creator ASR v2: Domain Adaptation & Edge CPU Deployment

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=flat&logo=huggingface)](https://huggingface.co/)
[![RTX-5070Ti](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%205070%20Ti-76B900.svg?style=flat&logo=nvidia)](https://nvidia.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An enterprise-grade, student-teacher speech recognition pipeline that solves **Acoustic Domain Collapse** on conversational Telugu YouTube speech (slang, 180–240 WPM speech rate, English tech loanwords) and fulfills the academic directive:
> *"The model is too heavy, make the model small so it should be able to run in CPU itself."*

---

## 🎯 Executive Summary & Breakthroughs

1. **Domain Collapse Quantified**: Open-source state-of-the-art models (*Swecha*, *EkStep Vakyansh*, *Meta XLSR-53*) fail drastically on real YouTube creators, producing **87.12% to 102.15% Word Error Rates (WER)**. Vakyansh completely destabilizes into infinite `<s>` token hallucination loops ($S+D+I > N$).
2. **Teacher-Student Knowledge Distillation**: Harvested **66.97 hours** across 4 top Telugu creator channels (`film_companion_south`, `prasad_tech`, `telugu_geeks`, `idream_post`). Labeled by Deepgram Nova-3 at 49.9 clips/s to build 46,987 verified training pairs.
3. **Rapid Domain Adaptation**: Fine-tuning Swecha Gonthuka dropped WER from **87.12% → 52.77%** (↓34.35% absolute) in just 500 steps.
4. **Edge CPU INT8 Quantization**: Dynamic post-training quantization (`torch.qint8`) shrunk the model from **360.27 MB → 116.08 MB (3.1× reduction)**, achieving **725 ms latency on a basic laptop CPU (6.9× faster than real-time)** with only a 1.9% quantization delta.
5. **Two-Stage Hybrid Engine**: Combines the fast INT8 acoustic model on CPU with an LLM Post-Corrector to restore English tech loanwords ("Tenglish") and grammar punctuation.

---

## 📊 Empirical Results Table

| Architecture | Model Family | Parameters | Pre-Train Zero-Shot WER | Fine-Tuned WER | Post-INT8 Quantization | CPU Latency (ms) | CPU Real-Time Factor (RTF) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Swecha Gonthuka** | Wav2Vec2 Base | 94.4M | **87.12%** | **52.77%** | **54.74%** | **725.3 ms** | **0.1441 (6.9× RT)** |
| **EkStep Vakyansh** | Wav2Vec2 Base | 94.4M | **102.15%** (Loops) | *Ready to Train* | — | — | — |
| **Meta XLSR-53** | Wav2Vec2 Large| 315.5M | **96.60%** | *Ready to Train* | — | — | — |
| **IndicConformer** | Conformer CTC | 129.3M | *Ready to Train* | *Ready to Train* | — | — | — |
| **IISc SraVaani** | FastConformer TDT| 110.0M | *Access Granted* | *Ready to Train* | — | — | — |

---

## 📁 Repository Structure

```
forge/
├── README.md                      # High-level architecture & results
├── A_TO_Z_PROJECT_GUIDE.md        # Comprehensive operational manual for AI agents & engineers
├── requirements.txt               # Pinned Python dependencies
├── manifests/
│   ├── benchmark_50_eval.jsonl    # 48 strictly held-out gold evaluation clips
│   ├── baseline_benchmark_results.json # Empirical zero-shot baseline metrics
│   ├── valid_nova3.jsonl          # 2,470 validation clips (3.36 hours)
│   └── train_nova3.jsonl          # 46,987 training clips (64.54 hours)
└── scripts/
    ├── download_models.py         # Automated HF model weights downloader
    ├── train_5070ti.py            # High-throughput training pipeline optimized for RTX 5070 Ti (16GB)
    ├── evaluate_benchmark.py      # Gold benchmark evaluator (WER, CER, RTF)
    ├── quantize_cpu.py            # INT8 dynamic quantization & CPU latency benchmark
    └── llm_postprocessor.py       # Two-stage Acoustic + LLM Code-Switching engine
```

---

## ⚡ Quickstart on NVIDIA RTX 5070 Ti (or Modern GPU)

### 1. Environment Setup
```bash
git clone https://github.com/tharun77714/forge.git
cd forge
pip install -r requirements.txt
```

### 2. Download Baseline Foundation Models
```bash
python scripts/download_models.py --model all
```

### 3. Fine-Tune Models (Optimized for 16GB VRAM)
```bash
# Fine-tune EkStep Vakyansh (Model #2)
python scripts/train_5070ti.py --model vakyansh --batch_size 16 --steps 2500 --audio_dir /path/to/segments

# Fine-tune Meta XLSR-53 (Model #3)
python scripts/train_5070ti.py --model xlsr_53_telugu --batch_size 16 --steps 2500 --lr 5e-5 --audio_dir /path/to/segments
```

### 4. Quantize to INT8 & Benchmark CPU Latency
```bash
python scripts/quantize_cpu.py --checkpoint checkpoints/vakyansh_finetuned --name vakyansh_creator --audio_dir /path/to/segments
```

### 5. Run Two-Stage Transcription (ASR + LLM Polish)
```bash
python scripts/llm_postprocessor.py --audio /path/to/sample.wav --model checkpoints/swecha_gonthuka_finetuned
```

---

## 📖 For External AI Agents / Collaborators

Read [`A_TO_Z_PROJECT_GUIDE.md`](./A_TO_Z_PROJECT_GUIDE.md) for full architectural theory, gradient accumulation math, hyperparameter breakdowns, and step-by-step reproduction instructions.
