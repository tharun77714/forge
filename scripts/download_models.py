r"""
download_models.py
Downloads the Telugu ASR Foundation Models into ./models/
Supports single or multi-model downloading with automatic caching.
"""
import os
import sys
import time
import argparse
from huggingface_hub import snapshot_download

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "swecha_gonthuka": {
        "name": "1. Swecha Gonthuka ASR (IIIT-H / Swecha, 94.4M)",
        "repo_id": "viswamaicoe/swecha-gonthuka-asr",
        "folder": os.path.join(MODELS_DIR, "swecha_gonthuka"),
        "allow_patterns": None,
    },
    "vakyansh": {
        "name": "2. EkStep Vakyansh Telugu (EkStep Foundation, 94.4M)",
        "repo_id": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
        "folder": os.path.join(MODELS_DIR, "vakyansh"),
        "allow_patterns": None,
    },
    "xlsr_53_telugu": {
        "name": "3. Wav2Vec2 XLSR-53 Telugu (Meta AI Base, 315.5M)",
        "repo_id": "anuragshas/wav2vec2-large-xlsr-53-telugu",
        "folder": os.path.join(MODELS_DIR, "xlsr_53_telugu"),
        "allow_patterns": None,
    },
    "indicconformer": {
        "name": "4. AI4Bharat IndicConformer (IIT Madras, 129.3M)",
        "repo_id": "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large",
        "folder": os.path.join(MODELS_DIR, "indicconformer"),
        "allow_patterns": ["*.nemo", "README.md", "config.yaml"],
    },
    "sravaani": {
        "name": "5. IISc ARTPARK SraVaani-1.0 (IISc Bangalore, 110M)",
        "repo_id": "ARTPARK-IISc/SraVaani-1.0",
        "folder": os.path.join(MODELS_DIR, "sravaani"),
        "allow_patterns": None,
    },
}

def download_model(key):
    info = MODELS[key]
    name = info["name"]
    repo = info["repo_id"]
    out_dir = info["folder"]
    patterns = info["allow_patterns"]

    print(f"\n[*] Fetching: {name}")
    print(f"    • Repo ID: {repo}")
    print(f"    • Target:  {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    try:
        if patterns:
            snapshot_download(
                repo_id=repo,
                local_dir=out_dir,
                allow_patterns=patterns,
                local_dir_use_symlinks=False,
            )
        else:
            snapshot_download(
                repo_id=repo,
                local_dir=out_dir,
                local_dir_use_symlinks=False,
            )
        elapsed = time.time() - t0
        print(f"[✓] Downloaded {name} in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"[!] Error downloading {name}: {e}")
        if "403" in str(e) or "gated" in str(e).lower():
            print("    --> Note: This model is gated. Ensure you ran `huggingface-cli login`.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Telugu ASR models from HuggingFace.")
    parser.add_argument("--model", type=str, default="all", choices=list(MODELS.keys()) + ["all"],
                        help="Model key to download or 'all'")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 TELUGU CREATOR ASR: MODEL DOWNLOADER")
    print(f"   Destination Directory: {MODELS_DIR}")
    print("=" * 80)

    if args.model == "all":
        for k in MODELS.keys():
            download_model(k)
    else:
        download_model(args.model)

    print("\n[✓] All requested downloads finished.")
