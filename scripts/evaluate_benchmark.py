r"""
evaluate_benchmark.py
Empirical Evaluation of Telugu Speech Recognition Models on Held-out Creator Audio.
Computes true Word Error Rate (WER) and Character Error Rate (CER) with zero hallucinations or mock data.
"""
import os
import sys
import json
import time
import argparse
import torch
import soundfile as sf
import jiwer
from transformers import AutoModelForCTC, AutoProcessor

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_MANIFEST = os.path.join(BASE_DIR, "manifests", "benchmark_50_eval.jsonl")

MODELS = {
    "swecha_baseline": {
        "name": "Swecha Gonthuka (Baseline)",
        "path": os.path.join(BASE_DIR, "models", "swecha_gonthuka"),
        "hf_id": "viswamaicoe/swecha-gonthuka-asr"
    },
    "swecha_finetuned": {
        "name": "Swecha Gonthuka (Fine-Tuned)",
        "path": os.path.join(BASE_DIR, "checkpoints", "swecha_gonthuka_finetuned"),
        "hf_id": None
    },
    "vakyansh_baseline": {
        "name": "EkStep Vakyansh (Baseline)",
        "path": os.path.join(BASE_DIR, "models", "vakyansh"),
        "hf_id": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100"
    },
    "vakyansh_finetuned": {
        "name": "EkStep Vakyansh (Fine-Tuned)",
        "path": os.path.join(BASE_DIR, "checkpoints", "vakyansh_finetuned"),
        "hf_id": None
    },
    "xlsr_53_baseline": {
        "name": "Wav2Vec2 XLSR-53 (Baseline)",
        "path": os.path.join(BASE_DIR, "models", "xlsr_53_telugu"),
        "hf_id": "anuragshas/wav2vec2-large-xlsr-53-telugu"
    },
    "xlsr_53_finetuned": {
        "name": "Wav2Vec2 XLSR-53 (Fine-Tuned)",
        "path": os.path.join(BASE_DIR, "checkpoints", "xlsr_53_telugu_finetuned"),
        "hf_id": None
    }
}

def resolve_audio(recorded_path, audio_dir=None):
    if os.path.exists(recorded_path):
        return recorded_path
    basename = os.path.basename(recorded_path)
    if audio_dir and os.path.exists(os.path.join(audio_dir, basename)):
        return os.path.join(audio_dir, basename)
    for fallback in ["./segments", "../segments", "./datasets/segments", "../datasets/segments"]:
        candidate = os.path.join(BASE_DIR, fallback, basename)
        if os.path.exists(candidate):
            return candidate
    return None

def evaluate_model(model_key, audio_dir=None, device="cuda"):
    info = MODELS[model_key]
    model_path = info["path"]
    if not os.path.exists(model_path) and info["hf_id"]:
        model_path = info["hf_id"]

    if not os.path.exists(model_path) and not info["hf_id"]:
        print(f"[!] Checkpoint for {model_key} not found at {model_path}. Skipping.")
        return None

    print(f"\n[*] Evaluating: {info['name']} on {device.upper()}...")
    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path).to(device)
    model.eval()

    samples = []
    with open(BENCHMARK_MANIFEST, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    predictions = []
    references = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for s in samples:
        wav_path = resolve_audio(s["audio_filepath"], audio_dir)
        ref = s["text"].strip()
        if not wav_path or not os.path.exists(wav_path) or not ref:
            continue

        wav, sr = sf.read(wav_path)
        total_audio_sec += len(wav) / sr

        t0 = time.perf_counter()
        inputs = proc(wav, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = proc.decode(pred_ids[0]).strip()
        total_infer_sec += (time.perf_counter() - t0)

        predictions.append(pred_text)
        references.append(ref)

    if not references:
        print("[!] No benchmark audio files could be loaded! Check --audio_dir.")
        return None

    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    rtf = total_infer_sec / total_audio_sec

    result = {
        "model_key": model_key,
        "name": info["name"],
        "wer": round(wer * 100, 2),
        "cer": round(cer * 100, 2),
        "rtf": round(rtf, 4),
        "speedup": round(1 / (rtf + 1e-6), 1),
        "evaluated_clips": len(predictions)
    }

    print(f"  --> WER: {result['wer']}% | CER: {result['cer']}% | RTF: {result['rtf']} ({result['speedup']}x real-time)")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Telugu creator gold benchmark.")
    parser.add_argument("--model", type=str, default="all", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "benchmark_results.json"))
    args = parser.parse_args()

    results = []
    if args.model == "all":
        for k in MODELS.keys():
            res = evaluate_model(k, audio_dir=args.audio_dir, device=args.device)
            if res:
                results.append(res)
    else:
        res = evaluate_model(args.model, audio_dir=args.audio_dir, device=args.device)
        if res:
            results.append(res)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Results saved to {args.output}")
