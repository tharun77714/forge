r"""
train_5070ti.py
High-Throughput Fine-Tuning Engine optimized for NVIDIA RTX 5070 Ti (16GB VRAM) & Modern GPUs.
Performs Teacher-Student Domain Adaptation on Telugu Creator Audio using PyTorch + Transformers.
"""
import os
import sys
import json
import time
import argparse
import torch
import soundfile as sf
import jiwer
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "swecha_gonthuka": {
        "name": "Swecha Gonthuka ASR (IIIT-H / Swecha, 94.4M)",
        "hf_id": "viswamaicoe/swecha-gonthuka-asr",
        "local_path": os.path.join(BASE_DIR, "models", "swecha_gonthuka"),
        "lr": 1e-4,
    },
    "vakyansh": {
        "name": "EkStep Vakyansh Telugu (EkStep Foundation, 94.4M)",
        "hf_id": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
        "local_path": os.path.join(BASE_DIR, "models", "vakyansh"),
        "lr": 1e-4,
    },
    "xlsr_53_telugu": {
        "name": "Wav2Vec2 XLSR-53 Telugu (Meta AI Large, 315.5M)",
        "hf_id": "anuragshas/wav2vec2-large-xlsr-53-telugu",
        "local_path": os.path.join(BASE_DIR, "models", "xlsr_53_telugu"),
        "lr": 5e-5,  # Lower LR for 315M parameter model to prevent catastrophic forgetting
    },
}

def resolve_audio_path(recorded_path, audio_dir=None):
    """Resolves audio filepath across different machines and operating systems."""
    if os.path.exists(recorded_path):
        return recorded_path
    
    basename = os.path.basename(recorded_path)
    if audio_dir and os.path.exists(os.path.join(audio_dir, basename)):
        return os.path.join(audio_dir, basename)
    
    # Try default relative directories
    for fallback in ["./segments", "../segments", "./datasets/segments", "../datasets/segments"]:
        candidate = os.path.join(BASE_DIR, fallback, basename)
        if os.path.exists(candidate):
            return candidate
            
    return None

class TeluguCreatorDataset(Dataset):
    def __init__(self, manifest_path, audio_dir=None, max_samples=None, max_duration=16.0):
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    dur = rec.get("duration", 0)
                    text = rec.get("text", "").strip()
                    if 0.5 <= dur <= max_duration and text:
                        # Store record
                        rec["_resolved_audio"] = resolve_audio_path(rec["audio_filepath"], audio_dir)
                        if rec["_resolved_audio"] is not None:
                            self.samples.append(rec)
                            if max_samples and len(self.samples) >= max_samples:
                                break
                except Exception:
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_collate_fn(processor):
    def collate_fn(batch):
        audio_arrays = []
        texts = []
        for item in batch:
            try:
                audio_path = item.get("_resolved_audio") or item["audio_filepath"]
                wav, sr = sf.read(audio_path)
                if sr != 16000:
                    continue
                audio_arrays.append(wav)
                texts.append(item["text"])
            except Exception:
                pass

        if not audio_arrays:
            return None

        # Pad audio features and texts
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
        labels = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

        return {
            "input_values": inputs.input_values,
            "labels": labels,
            "texts": texts
        }
    return collate_fn

def evaluate_benchmark(model, processor, benchmark_samples, audio_dir=None, device="cuda"):
    model.eval()
    predictions = []
    references = []

    for s in benchmark_samples:
        wav_path = resolve_audio_path(s["audio_filepath"], audio_dir)
        ref_text = s["text"].strip()
        if not wav_path or not os.path.exists(wav_path) or not ref_text:
            continue
        try:
            wav, sr = sf.read(wav_path)
            inputs = processor(wav, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = processor.decode(pred_ids[0]).strip()
            predictions.append(pred_text)
            references.append(ref_text)
        except Exception:
            pass

    if not references:
        return 1.0, 1.0

    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    return wer, cer

def train_model(model_key, args):
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["name"]
    model_src = cfg["local_path"] if os.path.exists(cfg["local_path"]) else cfg["hf_id"]
    lr = args.lr if args.lr else cfg["lr"]

    print("\n" + "=" * 80)
    print(f"🚀 INITIALIZING FINE-TUNING: {model_name}")
    print(f"   Source Path / ID : {model_src}")
    print(f"   Batch Size       : {args.batch_size} (Grad Accum: {args.grad_accum} -> Eff Batch: {args.batch_size * args.grad_accum})")
    print(f"   Target Steps     : {args.steps}")
    print(f"   Learning Rate    : {lr}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] Detected GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("[!] No GPU detected. Training on CPU will be extremely slow.")

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_src)
    model = AutoModelForCTC.from_pretrained(model_src).to(device)

    # Freeze convolutional feature encoder (saves ~40% VRAM & preserves acoustic primitives)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        print("[*] Frozen low-level CNN feature encoder. Adapting Transformer attention & CTC projection.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Parameters: {trainable_params/1e6:.1f}M Trainable / {total_params/1e6:.1f}M Total ({trainable_params/total_params*100:.1f}%)")

    # Load Benchmark Gold Samples
    benchmark_samples = []
    if os.path.exists(args.benchmark_manifest):
        with open(args.benchmark_manifest, "r", encoding="utf-8") as f:
            benchmark_samples = [json.loads(l) for l in f if l.strip()]

    # Baseline Check before training
    print("[*] Running zero-shot baseline evaluation on held-out gold benchmark...")
    base_wer, base_cer = evaluate_benchmark(model, processor, benchmark_samples, args.audio_dir, device)
    print(f"[*] Pre-Training Baseline WER: {base_wer*100:.2f}% | CER: {base_cer*100:.2f}%")

    # Load Dataset
    print(f"[*] Loading training dataset: {args.manifest}...")
    dataset = TeluguCreatorDataset(args.manifest, audio_dir=args.audio_dir)
    print(f"[*] Verified & ready clips: {len(dataset):,}")

    collate_fn = create_collate_fn(processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True if device == "cuda" else False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    save_dir = os.path.join(CHECKPOINT_DIR, f"{model_key}_finetuned")
    os.makedirs(save_dir, exist_ok=True)

    best_wer = base_wer
    global_step = 0
    t_start = time.time()
    accum_loss = 0.0

    model.train()
    optimizer.zero_grad()

    while global_step < args.steps:
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(input_values, labels=labels)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 25 == 0 or global_step == 1:
                    cur_loss = accum_loss / (25 if global_step > 1 else 1)
                    elapsed = time.time() - t_start
                    steps_per_sec = global_step / (elapsed + 1e-5)
                    eta_sec = (args.steps - global_step) / (steps_per_sec + 1e-5)
                    print(
                        f"  [Step {global_step:>4}/{args.steps}] Loss: {cur_loss:.4f} | "
                        f"⚡ {steps_per_sec:.2f} step/s | ETA: {eta_sec/60:.1f} min",
                        flush=True,
                    )
                    accum_loss = 0.0

                if global_step % args.eval_steps == 0 or global_step == args.steps:
                    val_wer, val_cer = evaluate_benchmark(model, processor, benchmark_samples, args.audio_dir, device)
                    print(f"\n  🎯 BENCHMARK EVAL (Step {global_step}): WER: {val_wer*100:.2f}% | CER: {val_cer*100:.2f}%")
                    if val_wer < best_wer:
                        best_wer = val_wer
                        model.save_pretrained(save_dir)
                        processor.save_pretrained(save_dir)
                        print(f"  🎉 NEW BEST CHECKPOINT SAVED to {save_dir} (WER: {val_wer*100:.2f}%)\n")
                    model.train()

                if global_step >= args.steps:
                    break

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"✅ MODEL COMPLETED: {model_name}")
    print(f"   Execution Time : {total_time/60:.2f} minutes")
    print(f"   Baseline WER   : {base_wer*100:.2f}%  ->  Fine-Tuned WER: {best_wer*100:.2f}% (↓{(base_wer - best_wer)*100:.2f}%)")
    print(f"   Saved Checkpoint: {save_dir}")
    print("=" * 80)
    return best_wer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Telugu Creator ASR models on NVIDIA RTX 5070 Ti.")
    parser.add_argument("--model", type=str, default="vakyansh", choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--steps", type=int, default=2500, help="Training steps (default: 2500)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for RTX 5070 Ti 16GB, 2 for RTX 4050)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Custom learning rate")
    parser.add_argument("--warmup", type=int, default=150, help="Warmup steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 mixed precision")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing .wav audio clips")
    parser.add_argument("--manifest", type=str, default=os.path.join(BASE_DIR, "manifests", "train_nova3.jsonl"))
    parser.add_argument("--benchmark_manifest", type=str, default=os.path.join(BASE_DIR, "manifests", "benchmark_50_eval.jsonl"))
    args = parser.parse_args()

    if args.model == "all":
        for m in MODEL_CONFIGS.keys():
            train_model(m, args)
    else:
        train_model(args.model, args)
