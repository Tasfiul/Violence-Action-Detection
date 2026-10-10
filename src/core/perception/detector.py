import time
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
import supervision as sv
from src.utils.config_loader import cfg
from src.utils.logger import logger

# Prevent Ultralytics from executing blocking global CUDA device synchronizations on each frame
ops.Profile.time = lambda self: time.perf_counter()

class Detector:
    def __init__(self):
        # 1. Load Settings
        desired_device = cfg.get('detection', {}).get('device', cfg.get('system', {}).get('device', 'cuda'))
        if desired_device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested for Detector but not available. Falling back to CPU.")
            self.device = 'cpu'
        else:
            self.device = desired_device

        model_path = cfg['detection']['model_weights']
        self.conf_thresh = cfg['detection']['confidence_threshold']
        self.target_classes = cfg['detection'].get('target_classes', [0])
        self.imgsz = cfg['detection'].get('imgsz', 640)
        self.half = cfg['detection'].get('half', True) if ('cuda' in str(self.device)) else False
        self.frame_skip = max(1, int(cfg['detection'].get('frame_skip', 1)))
        self.frame_counter = 0
        self.last_detections = sv.Detections.empty()

        logger.info(f"Loading YOLO model from {model_path} to {self.device} (imgsz={self.imgsz}, fp16={self.half}, frame_skip={self.frame_skip})...")
        
        # 2. Initialize YOLO
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            if self.half and 'cuda' in str(self.device):
                self.model.model.half()
            
            # Warmup (prevents lag on first frame)
            warmup_tensor = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
            if self.half and self.device == 'cuda':
                warmup_tensor = warmup_tensor.half()
            self.model(warmup_tensor, verbose=False)
            
        except Exception as e:
            logger.critical(f"Failed to load YOLO model: {e}")
            raise e

        # 3. Initialize ByteTrack (via Supervision)
        track_thresh = cfg['detection'].get('track_activation_threshold', 0.25)
        lost_buf = cfg['detection'].get('lost_track_buffer', 30)
        match_thresh = cfg['detection'].get('minimum_matching_threshold', 0.8)
        trk_fps = cfg['detection'].get('tracker_frame_rate', 30)

        logger.info(f"Initializing ByteTrack (activation={track_thresh}, lost_buffer={lost_buf}, match={match_thresh})...")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=lost_buf,
            minimum_matching_threshold=match_thresh,
            frame_rate=trk_fps
        )

    def reset_tracker(self):
        """Resets ByteTrack tracks and frame state."""
        if hasattr(self, 'tracker') and hasattr(self.tracker, 'reset'):
            try:
                self.tracker.reset()
            except Exception:
                pass
        self.frame_counter = 0
        self.last_detections = None

    def process_frame(self, frame):
        """
        Input: Raw Frame (numpy array)
        Output: Tracked Detections (Supervision object)
        """
        # Frame Skip optimization (if enabled, skips inference on intermediate frames)
        self.frame_counter += 1
        if self.frame_skip > 1 and (self.frame_counter % self.frame_skip != 1):
            return self.last_detections

        # A. Inference
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device
        )[0]

        # B. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # C. Filter (Keep only Persons - Class ID 0)
        detections = detections[detections.class_id == 0]

        # D. Update Tracker
        detections = self.tracker.update_with_detections(detections)
        self.last_detections = detections

        return detections