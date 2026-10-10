import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import time
from src.core.perception.detector import Detector
from src.utils.logger import logger

def run_test():
    logger.info("--- Phase 1 (Detector & ByteTrack) Test Starting ---")
    detector = Detector()
    
    # Create a dummy frame (720, 1280, 3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    logger.info("Running detection on dummy frame...")
    start = time.time()
    detections = detector.process_frame(frame)
    elapsed = time.time() - start
    
    logger.info(f"Detection processed in {elapsed:.4f}s")
    logger.info(f"Detections found: {len(detections)}")
    logger.info("--- Phase 1 Test Complete ---")

if __name__ == "__main__":
    run_test()
