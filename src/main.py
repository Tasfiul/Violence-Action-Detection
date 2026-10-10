import os
import sys
from pathlib import Path
import argparse

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import cfg
from src.pipelines.rapid_flow import RapidPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Action Detection Pipeline")
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Path to video file (.mp4) or webcam device index (e.g. 0). Defaults to input_source in config.yaml."
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip Phase 3 VLM (Ollama) entirely."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless without opening OpenCV GUI display window."
    )
    args = parser.parse_args()

    # Determine video source
    video_source = args.source if args.source is not None else cfg['paths']['input_source']
    
    # Initialize and Run Pipeline
    app = RapidPipeline(
        skip_vlm=True if args.skip_vlm else None,
        headless=True if args.headless else None
    )
    app.run(video_source)