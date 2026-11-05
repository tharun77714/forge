# A-TO-Z MASTER OPERATIONAL GUIDE FOR AI AGENTS & RESEARCHERS
**Project**: Telugu Creator Automatic Speech Recognition (ASR) v2  
**Target Hardware**: NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM) & CPU Deployment Target  
**Repository**: `https://github.com/tharun77714/forge`  

---

## 1. Executive Context & Mission

### The Core Problem: Acoustic Domain Collapse
Pre-trained open-source Indian ASR models (*Swecha*, *EkStep Vakyansh*, *Meta XLSR-53*) are trained exclusively on clean, read, textbook speech (*Common Voice*, *Kathbath*, *Shruti*). When deployed on real-world **Telugu YouTube Creators**, they suffer catastrophic failure:
- **Baseline WER**: **87.12% to 102.15%**.
- **Failure Modes**:
  1. High conversational speech rate (180–240 WPM vs 120 WPM read speech).
  2. Heavy colloquial slang, contractions, and audio banter.
  3. Extensive English technical loanwords (*"unboxing"*, *"processor"*, *"camera"*, *"battery"*).
  4. Background studio music and reverberant room acoustics.
  5. EkStep Vakyansh gets stuck in infinite `<s>` token hallucination loops where substitutions + deletions + insertions exceed total reference words ($S+D+I > N$), producing >100% WER!

### The Academic / Faculty Constraint
> *"The model is too heavy, make the model small so it should be able to run in CPU itself."*
You cannot simply leave the models as bulky FP32 GPU weights. You must quantize the fine-tuned winner to **PyTorch INT8 (`torch.qint8`)**, verify sub-second latency on a standard laptop CPU, and prove that CPU inference runs faster than real-time playback (RTF < 0.2).

### The Distillation Solution
We harvested **66.97 hours** (48,632 valid clips) of raw Telugu creator audio across 4 top channels (`film_companion_south`, `prasad_tech`, `telugu_geeks`, `idream_post`). We used **Deepgram Nova-3** as an oracle teacher to transcribe the entire dataset at 49.9 clips/s with zero rate limits, creating a high-fidelity ground truth training corpus.

---

## 2. Hardware Advantage: NVIDIA RTX 5070 Ti (16 GB GDDR7)

The original pilot run was performed on an RTX 4050 Laptop GPU (6 GB VRAM). Moving to an **RTX 5070 Ti (16 GB VRAM)** provides massive compute headroom:

| Parameter | RTX 4050 (Pilot Machine) | RTX 5070 Ti (Your Machine) |
| :--- | :--- | :--- |
| **VRAM** | 6 GB (4.8 GB usable) | **16 GB GDDR7** |
| **Native Batch Size** | 2 clips (requires grad accum 8) | **16 clips directly in VRAM** |
| **Mixed Precision** | FP16 | **FP16 / BF16 (Blackwell Tensor Cores)** |
| **Training Throughput** | ~13–15 clips/second | **55–70 clips/second (4.5× faster!)** |
| **1 Full Epoch (47,000 clips)** | ~55 minutes | **~15–18 minutes** |
| **Total Time per Model** | ~1 hour 15 min | **~25–30 minutes** |

---

## 3. Dataset & Manifest Structure

The dataset contains **51,661 WAV audio clips (~7.59 GB)** sampled at 16,000 Hz mono 16-bit PCM.

The repository includes pre-built manifest files in `./manifests/`:
- `manifests/train_nova3.jsonl`: **46,987 training clips (64.54 hours)**.
- `manifests/valid_nova3.jsonl`: **2,470 validation clips (3.36 hours)**.
- `manifests/benchmark_50_eval.jsonl`: **48 strictly held-out gold test clips (0.07 hours)**.
- `manifests/baseline_benchmark_results.json`: Empirical zero-shot baseline metrics.

### Dynamic Audio Path Resolution
Audio paths in `train_nova3.jsonl` originally recorded Windows paths from the donor machine (e.g. `C:\Users\Kotha\...`).  
**Do NOT edit the manifest files!**  
All training and evaluation scripts implement `resolve_audio_path()`:
If the recorded absolute path does not exist on your machine, the script automatically takes the filename (e.g. `film_companion_south_...wav`) and searches inside `--audio_dir` or local fallback folders (`./segments`, `../segments`, `./datasets/segments`).

---

## 4. The 5 Candidate Architectures

| ID | Model Name | Architecture | Parameters | Baseline WER | Training Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `swecha_gonthuka` | Swecha Gonthuka ASR | Wav2Vec2 Base | 94.4M | **87.12%** | *Pilot complete (dropped to 52.77%)* |
| `vakyansh` | EkStep Vakyansh Telugu | Wav2Vec2 Base | 94.4M | **102.15%** | Freeze CNN, train Transformer + CTC head (`lr=1e-4`) |
| `xlsr_53_telugu` | Wav2Vec2 XLSR-53 Telugu | Wav2Vec2 Large| 315.5M | **96.60%** | Freeze CNN, train Transformer + CTC head (`lr=5e-5`) |
| `indicconformer` | AI4Bharat IndicConformer| Conformer CTC | 129.3M | Pending eval | Conformer blocks with Cosine schedule (`lr=1e-4`) |
| `sravaani` | IISc ARTPARK SraVaani-1.0 | FastConformer TDT| 110.0M | Access Unlocked | HuggingFace `SraVaaniTDTModel` with Token-Duration Transducer |

---

## 5. Step-by-Step Execution Workflow

Follow these exact steps on the RTX 5070 Ti machine:

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/tharun77714/forge.git
cd forge
pip install -r requirements.txt
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0), '| VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

### Step 2: Download / Mount the Audio Clips
You can obtain the ~7.6 GB audio clips in one of two ways:

**Method A (Automated Cloud Download from HuggingFace):**
```bash
python scripts/download_dataset.py
```
*(This automatically downloads and extracts the entire `segments` audio archive directly into `forge/segments/`).*

**Method B (Local / USB Transfer):**
Place the `segments` folder directly inside `forge/segments/` (or specify `--audio_dir <path>`).

Verify audio resolution:
```bash
python -c "from scripts.train_5070ti import resolve_audio_path; print('Resolved:', resolve_audio_path('dummy_film_companion_south_0WxUj-uWjiQ_seg0041_234.525_238.335.wav', audio_dir='segments'))"
```

### Step 3: Download Model Weights
Download the pre-trained weights from HuggingFace Hub:
```bash
python scripts/download_models.py --model all
```
*(If prompted for gated models like SraVaani, run `huggingface-cli login` and paste your HuggingFace token).*

---

### Step 4: Fine-Tune Model #2 (EkStep Vakyansh)
EkStep Vakyansh had a 102.15% baseline WER due to token hallucination loops. Fine-tuning will eliminate this bug.

Run on RTX 5070 Ti:
```bash
python scripts/train_5070ti.py \
    --model vakyansh \
    --batch_size 16 \
    --grad_accum 2 \
    --steps 2500 \
    --lr 1e-4 \
    --audio_dir ./segments
```
- **Batch Size**: 16 clips directly in VRAM.
- **Grad Accum**: 2 (effective batch size = 32 clips).
- **Time Expected**: ~20–25 minutes.
- **Output**: Best checkpoint automatically saved to `./checkpoints/vakyansh_finetuned`.

---

### Step 5: Fine-Tune Model #3 (Meta AI Wav2Vec2 XLSR-53)
XLSR-53 is a large 315.5M parameter multilingual model. We use a lower learning rate (`5e-5`) to avoid catastrophic forgetting:

```bash
python scripts/train_5070ti.py \
    --model xlsr_53_telugu \
    --batch_size 16 \
    --grad_accum 2 \
    --steps 2500 \
    --lr 5e-5 \
    --audio_dir ./segments
```
- **Time Expected**: ~30–35 minutes.
- **Output**: Checkpoint saved to `./checkpoints/xlsr_53_telugu_finetuned`.

---

### Step 6: Evaluate All Models on Gold Benchmark
Run full comparative evaluation across the 48 held-out gold test clips:
```bash
python scripts/evaluate_benchmark.py --model all --audio_dir ./segments
```
This generates `benchmark_results.json` containing exact Word Error Rates (WER), Character Error Rates (CER), and Real-Time Factors (RTF).

---

### Step 7: Dynamic INT8 Quantization for CPU Deployment (Professor Directive)
Fulfill the academic requirement by compressing the fine-tuned models to INT8 and measuring CPU latency:

```bash
# Quantize EkStep Vakyansh
python scripts/quantize_cpu.py \
    --checkpoint checkpoints/vakyansh_finetuned \
    --name vakyansh_creator \
    --audio_dir ./segments

# Quantize Swecha Gonthuka
python scripts/quantize_cpu.py \
    --checkpoint checkpoints/swecha_gonthuka_finetuned \
    --name swecha_creator \
    --audio_dir ./segments
```

#### Expected Quantization Metrics:
- **Size Reduction**: ~360 MB → ~116 MB (**68% smaller / 3.1× compression**).
- **CPU Latency**: ~650–750 ms per 4-second clip.
- **Real-Time Factor (RTF)**: ~0.14–0.16 (**6.5× to 7× faster than real-time playback on standard CPU**).
- **Quantization Delta**: < 2.0% WER trade-off.

---

### Step 8: Two-Stage Hybrid Engine (ASR + LLM Post-Processor)
To handle code-switching ("Tenglish") where creators mix English tech terms into Telugu speech:

```bash
# Run with local heuristic loanword dictionary:
python scripts/llm_postprocessor.py \
    --audio ./segments/sample_tech_clip.wav \
    --model checkpoints/vakyansh_finetuned

# Run with LLM API (Gemini / OpenAI):
export GEMINI_API_KEY="your-api-key"
python scripts/llm_postprocessor.py \
    --audio ./segments/sample_tech_clip.wav \
    --model checkpoints/vakyansh_finetuned \
    --use_llm
```

#### What the Two-Stage Engine Achieves:
- **Stage 1 (Acoustic)**: Generates raw phonetic Telugu in <200ms on CPU (`ఈ ఫోన్ లో ప్రాసెసర్ స్పీడ్ చాలా బాగుంది`).
- **Stage 2 (Semantic LLM)**: Restores English loanwords and applies punctuation (`ఈ ఫోన్ లో Processor speed చాలా బాగుంది, Camera 4K support చేస్తుంది.`).

---

## 6. Critical Engineering Rules (DO NOT BREAK)

1. **NO FAKE SCRIPTS OR MOCK DATA**: Every metric must be computed using `jiwer.wer()` and `time.perf_counter()` on real audio files.
2. **FREEZE CNN FEATURE ENCODER**: Never unfreeze the low-level convolutional audio feature extractor on Wav2Vec2 models (`model.freeze_feature_encoder()`). Unfreezing it corrupts fundamental audio wave filters and wastes 40% VRAM.
3. **DO NOT RESAMPLE MANUALLY**: All models expect 16,000 Hz single-channel mono PCM. The dataset is already standardized to 16 kHz.
4. **SAVE CHECKPOINTS ON VALIDATION WER**: Always save when `val_wer < best_wer`. Never rely solely on training loss (CTC loss can decrease while WER overfits).
5. **KEEP INT8 MODELS FOR CPU ONLY**: PyTorch dynamic quantization (`torch.quantization.quantize_dynamic`) is optimized for x86 CPU SIMD instructions (AVX-512/VNNI). Do not move quantized INT8 models back to CUDA.

---

## 7. Expected Final Benchmark Table for Presentation

When all 5 models are evaluated, populate this master table for your slide deck and final thesis report:

```markdown
| Architecture | Model ID | Parameters | Baseline WER | Fine-Tuned WER | Absolute Drop (↓) | INT8 Size | CPU Latency | CPU RTF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wav2Vec2 Base** | Swecha Gonthuka | 94.4M | 87.12% | 52.77% | ↓34.35% | 116 MB | 725 ms | 0.1441 |
| **Wav2Vec2 Base** | EkStep Vakyansh | 94.4M | 102.15% | [Trained WER] | [Drop] | 116 MB | ~700 ms | ~0.1400 |
| **Wav2Vec2 Large** | Meta XLSR-53 | 315.5M | 96.60% | [Trained WER] | [Drop] | 380 MB | ~1800 ms| ~0.3500 |
| **Conformer CTC** | IndicConformer | 129.3M | [Eval WER] | [Trained WER] | [Drop] | 160 MB | ~950 ms | ~0.1900 |
| **FastConformer TDT**| IISc SraVaani | 110.0M | [Eval WER] | [Trained WER] | [Drop] | 140 MB | ~850 ms | ~0.1700 |
```

---

*This guide contains everything required for any AI agent or engineer to execute, reproduce, and extend the Telugu Creator ASR v2 pipeline.*
