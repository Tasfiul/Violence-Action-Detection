#!/usr/bin/env python
"""
Automated Model Weights Downloader for SentinAI Violence Action Detection.
Downloads missing YOLOv8 and OpenCLIP CoCa ViT-L-14 model weights into models/.
Usage:
    python scripts/download_models.py
"""
import os
import sys
import urllib.request
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


MODELS = {
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "description": "YOLOv8 Nano Person Detection Weights (~6.5 MB)",
        "expected_min_bytes": 6_000_000,
    },
    "coca_l14.bin": {
        "url": "https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin",
        "description": "OpenCLIP CoCa ViT-L-14 Action Recognition Weights (~2.55 GB)",
        "expected_min_bytes": 2_500_000_000,
    }
}


def download_with_progress(url: str, output_path: Path, desc: str):
    """Downloads a file with a dynamic progress bar."""
    print(f"\n[DOWNLOADING] {desc}")
    print(f"  Source: {url}")
    print(f"  Target: {output_path}")

    temp_path = output_path.with_suffix(".tmp")

    class DownloadProgressBar:
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if self.pbar is None and tqdm is not None and total_size > 0:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=output_path.name)
            if self.pbar:
                downloaded = block_num * block_size
                if downloaded < total_size:
                    self.pbar.update(block_size)
                else:
                    self.pbar.n = total_size
                    self.pbar.refresh()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=DownloadProgressBar() if tqdm else None)
        if temp_path.exists():
            os.replace(temp_path, output_path)
        print(f"[SUCCESS] Downloaded: {output_path.name} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)\n")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[ERROR] Failed to download {output_path.name}: {e}")
        print(f"  Manual download link: {url}")
        raise e


def main():
    print("=" * 65)
    print("   SENTINAI // MODEL WEIGHTS VERIFICATION & DOWNLOADER")
    print("=" * 65)
    print(f"Target directory: {MODELS_DIR}\n")

    all_present = True
    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size >= info["expected_min_bytes"]:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [OK] {filename:<15} present ({size_mb:.1f} MB) - {info['description']}")
        else:
            all_present = False
            print(f"  [MISSING] {filename:<15} - {info['description']}")
            download_with_progress(info["url"], dest, info["description"])

    if all_present:
        print("\nAll required model weights are verified and ready to run.")
    print("=" * 65)


if __name__ == "__main__":
    main()
