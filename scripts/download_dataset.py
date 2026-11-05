r"""
download_dataset.py
Downloads and extracts the verified Telugu Creator Audio Dataset (~7.6 GB)
from the private HuggingFace dataset repository directly into ./segments/
"""
import os
import sys
import zipfile
from huggingface_hub import hf_hub_download

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DIR = os.path.join(BASE_DIR, "segments")
REPO_ID = "jenxsjeixj/telugu-creator-speech"

def download_and_extract():
    print("=" * 80)
    print(f"🚀 DOWNLOADING TELUGU CREATOR DATASET FROM HUGGINGFACE")
    print(f"   Repository : {REPO_ID}")
    print(f"   Target Dir : {TARGET_DIR}")
    print("=" * 80)

    try:
        print("[*] Fetching segments.zip from HuggingFace Hub (this may take a few minutes)...")
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="segments.zip",
            repo_type="dataset",
        )
        print(f"[✓] Downloaded archive to: {zip_path}")
        print(f"[*] Extracting clips to: {BASE_DIR}...")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
            
        print(f"[✓] Dataset successfully extracted into: {TARGET_DIR}")
        print(f"[*] Total clips ready: {len(os.listdir(TARGET_DIR)):,}")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"[!] Error downloading dataset: {e}")
        print("    --> Note: If access is denied, run `huggingface-cli login` on this machine first.")
        return False

if __name__ == "__main__":
    download_and_extract()
