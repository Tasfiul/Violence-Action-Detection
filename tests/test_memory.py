import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import supervision as sv
from src.core.memory.evidence import EvidenceManager
from src.core.memory.state_manager import SecurityStateManager
from src.utils.logger import logger

def run_test():
    logger.info("--- Testing EvidenceManager & SecurityStateManager ---")
    memory = EvidenceManager()
    state_mgr = SecurityStateManager(alert_trigger_count=3)
    
    # Create fake frame and detection
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    xyxy = np.array([[100, 100, 200, 300]], dtype=np.float32)
    tracker_id = np.array([1])
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=tracker_id
    )
    
    # Update memory for 8 frames (window_size)
    ready = {}
    for _ in range(8):
        ready = memory.update(frame, detections)
        
    logger.info(f"Clips ready after 8 frames: {list(ready.keys())}")
    assert 1 in ready, "Tracker ID 1 should be ready with a clip!"
    assert len(ready[1]) == 8, f"Clip should have 8 frames, got {len(ready[1])}"
    
    # Test State Manager updates
    state_mgr.update_phase2(1, is_violent=False, action_label="normal behavior", confidence=0.95)
    color, text = state_mgr.get_ui_data(1)
    logger.info(f"Safe state UI data: color={color}, text={text}")
    assert color == (0, 255, 0), "Color should be green for safe state"
    
    # Test strikes escalation
    for i in range(3):
        state_mgr.update_phase2(1, is_violent=True, action_label="punching", confidence=0.88)
    color, text = state_mgr.get_ui_data(1)
    logger.info(f"After 3 strikes UI data: color={color}, text={text}")
    assert color == (0, 165, 255), "Color should be orange after 3 strikes"
    
    # Test confirm threat
    state_mgr.confirm_threat(1, "CONFIRMED: PUNCHING")
    color, text = state_mgr.get_ui_data(1)
    logger.info(f"After confirm_threat UI data: color={color}, text={text}")
    assert color == (0, 0, 255), "Color should be red after confirmation"
    
    logger.info("--- Memory & State Manager Test Passed Successfully ---")

if __name__ == "__main__":
    run_test()
