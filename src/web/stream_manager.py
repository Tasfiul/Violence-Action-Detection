import os
import cv2
import time
import json
import glob
import threading
import numpy as np
from datetime import datetime

from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.pipelines.rapid_flow import RapidPipeline
from src.database.mongo_manager import get_mongo_uploader


class WebStreamManager:
    """
    Thread-safe stream manager that integrates the RapidPipeline into a web application.
    Controls live webcam/RTSP feeds and video file uploads with 0% extra GPU overhead.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.pipeline = None
        self.is_streaming = False
        self.current_source = None
        self.current_source_name = "None"
        self.source_type = "idle"  # 'camera', 'video', 'idle'
        self.status = "IDLE"      # 'IDLE', 'INITIALIZING', 'STREAMING', 'ANALYZING', 'COMPLETED', 'ERROR'
        self.status_message = "System idle. Ready to monitor."

        self.latest_frame_bytes = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stream_thread = None

        self.fps = 0.0
        self.frame_count = 0
        self.start_time = 0.0
        self.last_frame_time = time.time()
        self.fps_history = []
        self.last_client_request_time = time.time()
        self.last_encode_time = 0.0

        # Placeholder frame when camera is not running
        self._standby_frame_bytes = self._create_standby_frame()

    def _ensure_pipeline(self):
        """Initializes the AI pipeline lazily in headless mode."""
        if self.pipeline is None:
            self.status = "INITIALIZING"
            self.status_message = "Loading AI models (YOLO & CoCa)..."
            logger.info("WebStreamManager: Initializing shared RapidPipeline in headless mode...")
            self.pipeline = RapidPipeline(headless=True)
            self.status = "IDLE"
            self.status_message = "AI models loaded. Ready to stream."

    def _create_standby_frame(self):
        """Generates a high-tech dark surveillance standby card with grid and radar HUD."""
        w, h = 1280, 720
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Subtle dark blueprint background
        frame[:] = (18, 14, 20)

        # Draw grid pattern
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (26, 22, 30), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (26, 22, 30), 1)

        # Center target reticle
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 140, (50, 40, 60), 2)
        cv2.circle(frame, (cx, cy), 180, (40, 32, 50), 1)
        cv2.line(frame, (cx - 200, cy), (cx + 200, cy), (45, 38, 55), 1)
        cv2.line(frame, (cx, cy - 200), (cx, cy + 200), (45, 38, 55), 1)

        # Corner HUD brackets
        b_len = 30
        corners = [(60, 60), (w - 60, 60), (60, h - 60), (w - 60, h - 60)]
        for (bx, by) in corners:
            dx = b_len if bx == 60 else -b_len
            dy = b_len if by == 60 else -b_len
            cv2.line(frame, (bx, by), (bx + dx, by), (0, 200, 255), 2)
            cv2.line(frame, (bx, by), (bx, by + dy), (0, 200, 255), 2)

        # HUD Text
        cv2.putText(frame, "SENTINAI // NEURAL SURVEILLANCE ENGINE", (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "STATUS: STANDBY", (cx - 110, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "PRESS 'START LIVE STREAM' OR 'UPLOAD VIDEO' TO BEGIN", (cx - 290, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 150), 1, cv2.LINE_AA)

        # Timestamp
        cv2.putText(frame, "[MONITORING OFFLINE]", (80, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 110), 1, cv2.LINE_AA)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def _frame_callback(self, out_frame):
        """Called whenever the pipeline produces an annotated frame."""
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 20:
                self.fps_history.pop(0)
            self.fps = round(sum(self.fps_history) / len(self.fps_history), 1)

        self.frame_count += 1

        # Only encode JPEG if web client recently pulled a frame (within 4s)
        # and throttle encoding to max 30 FPS to leave 100% compute for AI inference
        if (now - self.last_client_request_time < 4.0) and (now - self.last_encode_time >= 0.033):
            self.last_encode_time = now
            h, w = out_frame.shape[:2]
            if w > 1280:
                # Fast downscale for web preview
                stream_frame = cv2.resize(out_frame, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_NEAREST)
            else:
                stream_frame = out_frame

            jpeg_quality = cfg.get("web", {}).get("jpeg_quality", 75)
            ret, buf = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ret:
                with self.frame_lock:
                    self.latest_frame_bytes = buf.tobytes()

    def start_stream(self, source):
        """
        Starts processing a video source in a background thread.
        source: can be an int (0 for webcam), numeric string ('0'), or video file path.
        """
        self.stop_stream()  # Ensure previous stream is cleanly terminated

        self._ensure_pipeline()
        self.stop_event.clear()

        # Determine source type & display name
        is_camera = isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit())
        if is_camera:
            self.source_type = "camera"
            self.current_source = int(source)
            self.current_source_name = f"Camera #{source}"
            self.status = "STREAMING"
            self.status_message = "Live camera monitoring active."
        else:
            self.source_type = "video"
            self.current_source = str(source)
            self.current_source_name = os.path.basename(str(source))
            self.status = "ANALYZING"
            self.status_message = f"Analyzing video: {self.current_source_name}"

        self.frame_count = 0
        self.fps_history = []
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.is_streaming = True

        # Reset pipeline session trackers and buffers
        self.pipeline.reset_session()

        def _runner():
            logger.info(f"WebStreamManager: Starting background execution on {self.current_source}...")
            try:
                self.pipeline.run(
                    source_path=self.current_source,
                    stop_event=self.stop_event,
                    frame_callback=self._frame_callback,
                    auto_shutdown=False
                )
                if not self.stop_event.is_set():
                    self.status = "COMPLETED"
                    self.status_message = f"Finished analyzing {self.current_source_name}."
            except Exception as e:
                logger.error(f"WebStreamManager stream error: {e}")
                self.status = "ERROR"
                self.status_message = f"Execution error: {str(e)}"
            finally:
                self.is_streaming = False
                logger.info(f"WebStreamManager: Finished stream on {self.current_source}")

        self.stream_thread = threading.Thread(target=_runner, daemon=True)
        self.stream_thread.start()
        return True

    def stop_stream(self):
        """Stops the active stream and resets state to IDLE."""
        if self.is_streaming or (self.stream_thread and self.stream_thread.is_alive()):
            self.status_message = "Stopping stream..."
            self.stop_event.set()
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
            self.is_streaming = False
            self.status = "IDLE"
            self.status_message = "Stream stopped. System in standby."
            logger.info("WebStreamManager: Stream successfully stopped.")
        else:
            self.status = "IDLE"
            self.status_message = "System in standby."

    def reset_feed(self):
        """
        Clears all tracking memory, threat alarms, motion buffers, and frame caches.
        If a stream is running, it resets tracking on the fly without interrupting camera capture.
        """
        with self.frame_lock:
            self.latest_frame_bytes = None
            self.fps_history = []
            self.frame_count = 0
            self.start_time = time.time()
            self.last_frame_time = time.time()

        if self.pipeline is not None:
            self.pipeline.reset_session()
            logger.info("WebStreamManager: Pipeline tracking memory, threat states, and buffers successfully reset.")

        return True

    def get_latest_frame(self):
        """Returns the latest JPEG frame bytes, or the standby HUD if idle."""
        self.last_client_request_time = time.time()
        with self.frame_lock:
            if self.is_streaming and self.latest_frame_bytes is not None:
                return self.latest_frame_bytes
            return self._standby_frame_bytes

    def get_status(self):
        """Returns live system telemetry and active threat state."""
        highest_threat_level = 0
        highest_threat_label = "NORMAL"
        active_tracks = 0

        if self.pipeline and hasattr(self.pipeline, 'state_manager'):
            states = self.pipeline.state_manager.states
            active_tracks = len(states)
            for state in states.values():
                if state.level > highest_threat_level:
                    highest_threat_level = state.level
                    highest_threat_label = state.label

        return {
            "is_streaming": self.is_streaming,
            "status": self.status,
            "status_message": self.status_message,
            "source": self.current_source_name,
            "source_type": self.source_type,
            "fps": self.fps,
            "frames_processed": self.frame_count,
            "active_tracks": active_tracks,
            "threat_level": highest_threat_level,
            "threat_label": highest_threat_label,
            "vlm_enabled": self.pipeline.vlm_enabled if self.pipeline else cfg.get('vlm', {}).get('enabled', False),
            "mongodb": get_mongo_uploader().get_status()
        }

    def get_incidents(self):
        """
        Scans data/outputs/ and groups incident mp4s, screenshots, and json reports.
        Returns a sorted list of incidents (newest first).
        """
        out_dir = cfg['paths'].get('output_dir', 'data/outputs')
        if not os.path.exists(out_dir):
            return []

        # Find all files starting with incident_
        all_files = os.listdir(out_dir)
        incident_map = {}

        for fname in all_files:
            if not fname.startswith("incident_"):
                continue
            
            fpath = os.path.join(out_dir, fname)
            mtime = os.path.getmtime(fpath)

            # Extract incident base name (e.g. incident_20261004-123522)
            base = fname
            if fname.endswith("_screenshot.jpg"):
                base = fname[:-15]
            elif fname.endswith("_report.json"):
                base = fname[:-12]
            elif fname.endswith(".mp4"):
                base = fname[:-4]

            if base not in incident_map:
                # Parse timestamp from base if possible
                ts_str = base.replace("incident_", "").split("-ezgif")[0]
                formatted_time = ts_str
                try:
                    dt = datetime.strptime(ts_str[:15], "%Y%m%d-%H%M%S")
                    formatted_time = dt.strftime("%b %d, %Y - %I:%M:%S %p")
                except Exception:
                    pass

                incident_map[base] = {
                    "id": base,
                    "timestamp": formatted_time,
                    "mtime": mtime,
                    "video": None,
                    "screenshot": None,
                    "report": None,
                    "video_size_mb": 0.0,
                    "threat_level": 2, # Default red for confirmed alert
                    "threat_label": "VIOLENCE DETECTED",
                    "report_summary": None
                }

            entry = incident_map[base]
            entry["mtime"] = max(entry["mtime"], mtime)

            if fname.endswith(".mp4"):
                entry["video"] = fname
                entry["video_size_mb"] = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            elif fname.endswith("_screenshot.jpg") or fname.endswith(".jpg"):
                entry["screenshot"] = fname
            elif fname.endswith("_report.json"):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        entry["report"] = report_data
                        if "description" in report_data:
                            entry["report_summary"] = report_data["description"]
                        if report_data.get("threat_detected", False):
                            entry["threat_label"] = "CONFIRMED FORENSIC THREAT"
                except Exception as e:
                    logger.debug(f"Error reading report {fname}: {e}")

        # Convert to list and sort by modification time descending
        incidents = list(incident_map.values())
        incidents.sort(key=lambda x: x["mtime"], reverse=True)
        return incidents

    def delete_incident(self, incident_id):
        """
        Deletes all files associated with an incident ID (.mp4, _screenshot.jpg, _report.json).
        """
        out_dir = cfg['paths'].get('output_dir', 'data/outputs')
        if not os.path.exists(out_dir):
            return False, "Output directory not found."

        clean_id = os.path.basename(incident_id).strip()
        if not clean_id:
            return False, "Invalid incident ID."

        deleted_files = []
        for fname in os.listdir(out_dir):
            # Match incident base name
            if fname.startswith(clean_id):
                fpath = os.path.join(out_dir, fname)
                try:
                    os.remove(fpath)
                    deleted_files.append(fname)
                    logger.info(f"Deleted incident artifact: {fpath}")
                except Exception as e:
                    logger.error(f"Failed to delete {fpath}: {e}")

        if deleted_files:
            return True, f"Deleted {len(deleted_files)} evidence files."
        return False, f"No files found matching {clean_id}."

