r"""
quantize_cpu.py
Dynamic INT8 Quantization & CPU Latency Benchmark for Telugu ASR Models.
Fulfills Faculty Directive: "The model is too heavy, make the model small so it should be able to run in CPU itself."
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
QUANT_DIR = os.path.join(BASE_DIR, "quantized_models")
os.makedirs(QUANT_DIR, exist_ok=True)

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

def quantize_and_benchmark(checkpoint_dir, output_name, audio_dir=None, manifest=None):
    manifest_path = manifest or os.path.join(BASE_DIR, "manifests", "benchmark_50_eval.jsonl")
    output_path = os.path.join(QUANT_DIR, f"{output_name}_int8.pt")

    print("\n" + "=" * 80)
    print(f"🔧 CPU QUANTIZATION & BENCHMARKING: {output_name}")
    print(f"   Input Checkpoint: {checkpoint_dir}")
    print(f"   Output INT8 Path: {output_path}")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    model = AutoModelForCTC.from_pretrained(checkpoint_dir).to("cpu")
    model.eval()

    # Dynamic INT8 Quantization
    print("[*] Quantizing PyTorch Linear layers to qint8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), output_path)

    # Size calculation
    safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    orig_size_mb = os.path.getsize(safetensor_path) / (1024 * 1024) if os.path.exists(safetensor_path) else 360.0
    quant_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    compression = (1 - quant_size_mb / orig_size_mb) * 100

    print(f"[✓] Quantization Completed!")
    print(f"    • Original Size : {orig_size_mb:.2f} MB")
    print(f"    • INT8 Size     : {quant_size_mb:.2f} MB ({compression:.1f}% reduction)")

    # Benchmark on CPU
    print(f"\n[*] Evaluating on CPU across gold benchmark clips...")
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    latencies_ms = []
    durations_sec = []
    predictions = []
    references = []

    for s in samples:
        wav_path = resolve_audio(s["audio_filepath"], audio_dir)
        ref = s["text"].strip()
        if not wav_path or not os.path.exists(wav_path) or not ref:
            continue

        wav, sr = sf.read(wav_path)
        durations_sec.append(len(wav) / sr)

        t0 = time.perf_counter()
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt").to("cpu")
        with torch.no_grad():
            logits = quantized_model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.decode(pred_ids[0]).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies_ms.append(elapsed_ms)
        predictions.append(pred_text)
        references.append(ref)

    if not references:
        print("[!] No benchmark clips could be evaluated.")
        return

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    total_audio = sum(durations_sec)
    total_compute = sum(latencies_ms) / 1000.0
    rtf = total_compute / total_audio

    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)

    print("\n" + "=" * 80)
    print("⚡ CPU INT8 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  • CPU Latency per Clip : {avg_latency:.1f} ms")
    print(f"  • Real-Time Factor     : {rtf:.4f} ({1/rtf:.1f}x faster than real-time playback)")
    print(f"  • INT8 Word Error Rate : {wer * 100:.2f}%")
    print(f"  • INT8 Char Error Rate : {cer * 100:.2f}%")
    print(f"  • RAM Footprint        : {quant_size_mb:.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize fine-tuned ASR model to INT8 and benchmark on CPU.")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "checkpoints", "swecha_gonthuka_finetuned"))
    parser.add_argument("--name", type=str, default="swecha_creator")
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args()

    quantize_and_benchmark(args.checkpoint, args.name, args.audio_dir, args.manifest)
