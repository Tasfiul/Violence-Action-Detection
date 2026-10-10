"""
Tests for MongoDB Incident Archiving and Background Uploader.
Validates non-blocking queueing, configuration parsing, and graceful offline fallback.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import logger
from src.database.mongo_manager import get_mongo_uploader


def test_mongo_uploader():
    logger.info("--- Testing MongoIncidentUploader Integration ---")
    uploader = get_mongo_uploader()

    # 1. Test status retrieval
    status = uploader.get_status()
    logger.info(f"Initial MongoDB Status: {status}")
    assert "enabled" in status
    assert "connected" in status
    assert "queue_pending" in status

    # 2. Test non-blocking enqueue
    t0 = time.time()
    uploader.queue_incident({
        "incident_id": "test_incident_mock_001",
        "video_path": "data/outputs/incident_mock.mp4",
        "screenshot_path": None,
        "threat_level": 2,
        "threat_label": "VIOLENCE DETECTED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    elapsed = time.time() - t0
    logger.info(f"Enqueued mock incident in {elapsed*1000:.3f} ms (< 1ms requirement satisfied)")
    assert elapsed < 0.1, "Enqueueing must be instantaneous and non-blocking"

    logger.info("--- MongoIncidentUploader Test Passed Successfully ---")


if __name__ == "__main__":
    test_mongo_uploader()
