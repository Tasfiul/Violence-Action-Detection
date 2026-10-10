# test_env.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_loader import cfg
from src.utils.logger import logger
import torch

def smoke_test():
    logger.info("--- Starting System Smoke Test & Hardware Profile Check ---")
    
    # 1. Test Config
    logger.info(f"Project: {cfg['project_name']} v{cfg['version']}")
    
    # 2. Test Torch/CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.warning("No GPU detected. System will run in CPU mode (Slow).")

    # 3. Display Performance Profile
    det = cfg.get('detection', {})
    act = cfg.get('action', {})
    vlm = cfg.get('vlm', {})
    sys_cfg = cfg.get('system', {})

    logger.info("=== ACTIVE PERFORMANCE PROFILE (.env) ===")
    logger.info(f"  System Device:    {sys_cfg.get('device', 'cpu')} (Benchmark: {sys_cfg.get('torch_benchmark', False)})")
    logger.info(f"  Phase 1 (YOLO):   device={det.get('device', sys_cfg.get('device'))}, model={det.get('model_weights')}, imgsz={det.get('imgsz', 640)}, fp16={det.get('half')}, frame_skip={det.get('frame_skip', 1)}")
    logger.info(f"  Phase 2 (Action): device={act.get('device', 'cpu')}, model={act.get('model_name')}, sample_frames={act.get('sample_frames')}, cooldown={act.get('analysis_cooldown')}s, queue_size={act.get('queue_size', 1)}")
    logger.info(f"  Phase 3 (VLM):    enabled={vlm.get('enabled', False)}, provider={vlm.get('provider')}, model={vlm.get('model_id')}")
    logger.info("===========================================")

    logger.info("=== ACTIVE DETECTION THRESHOLDS (thresholds.env) ===")
    logger.info(f"  Phase 1 Thresholds: conf={det.get('confidence_threshold')}, iou={det.get('iou_threshold')}, track_act={det.get('track_activation_threshold')}, match={det.get('minimum_matching_threshold')}, lost_buf={det.get('lost_track_buffer')}")
    logger.info(f"  Phase 2 Thresholds: motion_gate={act.get('motion_threshold')}, violence_conf={act.get('threshold')}, osr_gate={act.get('osr_threshold')}, strike_trigger={act.get('alert_trigger_count')}, crop_margin={act.get('crop_context_ratio')}")
    logger.info(f"  State Persistence:  memory_timeout={cfg.get('state', {}).get('timeout_seconds', 10.0)}s")
    logger.info("====================================================")

    mongo = cfg.get('mongodb', {})
    logger.info("=== DATABASE ARCHIVE CONFIGURATION (MongoDB) ===")
    logger.info(f"  Enabled:        {mongo.get('enabled', False)}")
    logger.info(f"  URI:            {mongo.get('uri', 'mongodb://localhost:27017')}")
    logger.info(f"  Upload Delay:   {mongo.get('upload_delay_seconds', 2.0)}s (Paced background transfers)")
    logger.info(f"  Media Upload:   videos={mongo.get('upload_videos', True)}, screenshots={mongo.get('upload_screenshots', True)}")
    logger.info("================================================")
    logger.info("--- Environment Check Passed ---")

if __name__ == "__main__":
    smoke_test()

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121