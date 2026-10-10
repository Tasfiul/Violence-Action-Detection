#!/usr/bin/env python
"""
Launcher for SentinAI Web Surveillance Dashboard.
Run: python run_web.py [--port 5000] [--host 0.0.0.0]
"""
import sys
import os
import argparse

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.web.app import app

if __name__ == "__main__":
    default_port = cfg.get("web", {}).get("port", 5000)
    default_host = cfg.get("web", {}).get("host", "0.0.0.0")

    parser = argparse.ArgumentParser(description="SentinAI Web Surveillance Interface")
    parser.add_argument("--port", "-p", type=int, default=default_port, help=f"Web server port (default: {default_port})")
    parser.add_argument("--host", type=str, default=default_host, help=f"Web server host (default: {default_host})")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("   SENTINAI // VIOLENCE ACTION DETECTION WEB DASHBOARD")
    print("=" * 65)
    print(f"   * Local Access:    http://localhost:{args.port}")
    print(f"   * Network Access:  http://{args.host}:{args.port}")
    print("   * Zero GPU Overhead: Web server runs on CPU threads")
    print("=" * 65 + "\n")

    logger.info(f"SentinAI Web Dashboard starting on http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
