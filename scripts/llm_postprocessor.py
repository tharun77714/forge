r"""
llm_postprocessor.py
Two-Stage Hybrid ASR Engine:
  Stage 1: Fine-Tuned INT8 Acoustic Model (Fast sub-second transcription on CPU)
  Stage 2: LLM Code-Switching & Punctuation Post-Corrector ("Tenglish" Restoration)

Solves the core failure of acoustic models on Telugu tech creator speech:
Restores English loanwords (e.g., 'processor', 'battery', 'unboxing'),
applies grammatical punctuation, and formats numbers/specifications.
"""
import os
import sys
import json
import argparse
import torch
import soundfile as sf
from transformers import AutoModelForCTC, AutoProcessor

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SYSTEM_PROMPT = """You are an expert Telugu-English bilingual ASR post-processor.
You receive raw, unpunctuated phonetic transcriptions from an acoustic speech recognition model listening to Telugu YouTube creators.
The creators speak colloquial Telugu heavily mixed with English technical loanwords (Tenglish).

Your task:
1. Identify phonetically transcribed English loanwords (e.g. 'స్మార్ట్ ఫోన్' -> 'smartphone', 'ప్రాసెసర్' -> 'processor', 'అన్ బాక్సింగ్' -> 'unboxing', 'కామెరా' -> 'camera') and write them in standard English if appropriate for tech creator subtitles, or clean up the phonetic Telugu.
2. Fix speech recognition stuttering or character repetitions.
3. Add proper punctuation (commas, periods, question marks).
4. Do NOT hallucinate new facts or change the meaning. Return ONLY the polished transcript."""

def rule_based_loanword_restore(text):
    """Fast offline heuristic dictionary for common Telugu tech creator loanwords."""
    tech_map = {
        "స్మార్ట్‌ఫోన్": "smartphone",
        "స్మార్ట్ ఫోన్": "smartphone",
        "ప్రాసెసర్": "processor",
        "ప్రాసెస్సర్": "processor",
        "డిస్‌ప్లే": "display",
        "డిస్ప్లే": "display",
        "బ్యాటరీ": "battery",
        "కెమెరా": "camera",
        "కామెరా": "camera",
        "అన్‌బాక్సింగ్": "unboxing",
        "అన్ బాక్సింగ్": "unboxing",
        "రివ్యూ": "review",
        "సబ్‌స్క్రైబ్": "subscribe",
        "గేమింగ్": "gaming",
        "పెర్ఫార్మెన్స్": "performance",
        "పెర్ఫార్మన్స్": "performance",
        "ఫీచర్స్": "features",
        "అప్‌డేట్": "update",
    }
    tokens = text.split()
    corrected = [tech_map.get(t, t) for t in tokens]
    return " ".join(corrected)

def run_two_stage_transcription(audio_path, model_path, use_llm_api=False, api_key=None):
    print("=" * 80)
    print("🎙️ TWO-STAGE ASR PIPELINE")
    print(f"   Audio: {audio_path}")
    print(f"   Model: {model_path}")
    print("=" * 80)

    # Stage 1: Acoustic Model Inference
    print("\n[*] STAGE 1: Acoustic Model Inference (Pure Telugu Phonetics)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path).to(device)
    model.eval()

    wav, sr = sf.read(audio_path)
    inputs = proc(wav, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    raw_acoustic_transcript = proc.decode(pred_ids[0]).strip()

    print(f"  [Stage 1 Output]: {raw_acoustic_transcript}")

    # Stage 2: Code-Switching & Loanword Restoration
    print("\n[*] STAGE 2: Semantic Polish & English Loanword Restoration...")
    if use_llm_api and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            llm = genai.GenerativeModel("gemini-1.5-flash")
            response = llm.generate_content(
                f"{SYSTEM_PROMPT}\n\nRaw ASR text: {raw_acoustic_transcript}\nCleaned transcript:"
            )
            final_transcript = response.text.strip()
            print(f"  [Stage 2 LLM Output]: {final_transcript}")
        except Exception as e:
            print(f"  [!] LLM API failed ({e}). Falling back to rule-based dictionary...")
            final_transcript = rule_based_loanword_restore(raw_acoustic_transcript)
            print(f"  [Stage 2 Heuristic Output]: {final_transcript}")
    else:
        final_transcript = rule_based_loanword_restore(raw_acoustic_transcript)
        print(f"  [Stage 2 Heuristic Output]: {final_transcript}")

    print("\n" + "=" * 80)
    print("✨ COMPARATIVE TRANSCRIPTION SUMMARY")
    print("=" * 80)
    print(f"  Raw Acoustic Output : {raw_acoustic_transcript}")
    print(f"  Stage 2 Final Polish: {final_transcript}")
    print("=" * 80)
    return raw_acoustic_transcript, final_transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage Telugu creator ASR with LLM post-processing.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("--model", type=str, default=os.path.join(BASE_DIR, "checkpoints", "swecha_gonthuka_finetuned"))
    parser.add_argument("--use_llm", action="store_true", help="Use LLM API for stage 2")
    parser.add_argument("--api_key", type=str, default=os.environ.get("GEMINI_API_KEY", None))
    args = parser.parse_args()

    run_two_stage_transcription(args.audio, args.model, args.use_llm, args.api_key)
