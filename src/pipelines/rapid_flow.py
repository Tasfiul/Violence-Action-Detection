import cv2
import queue
import threading
import os
import time
import json
import torch

from collections import deque

from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.utils.visualization import Visualizer

from src.core.perception.detector import Detector
from src.core.memory.evidence import EvidenceManager
from src.core.memory.state_manager import SecurityStateManager
from src.core.analysis.action_rec import ActionRecognizer
from src.core.analysis.vlm import VisionReasonerFactory


class CameraStream:
    """
    Zero-latency threaded camera capture.
    Constantly drains the driver buffer in the background and delivers the absolute latest frame,
    eliminating accumulated webcam display lag.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class RapidPipeline:
    """
    The main pipeline that orchestrates the entire system.
    """
    def __init__(self, skip_vlm=None, headless=None):
        """Initializes the pipeline."""

        logger.info("Initializing Asynchronous Pipeline...")
        
        if headless is not None:
            self.headless = headless
        else:
            self.headless = cfg['system'].get('headless', False)
        
        self.detector = Detector()
        self.memory = EvidenceManager()
        self.visualizer = Visualizer()
        self.brain = ActionRecognizer()
        
        # Check if Phase 3 VLM should run
        if skip_vlm is not None:
            self.vlm_enabled = not skip_vlm
        else:
            self.vlm_enabled = cfg.get('vlm', {}).get('enabled', False)

        if self.vlm_enabled:
            logger.info("Initializing Phase 3 VLM Reasoner...")
            self.reasoner = VisionReasonerFactory.create()
            self.vlm_queue = queue.Queue()
            self.vlm_worker_thread = threading.Thread(target=self._vlm_worker, daemon=True)
            self.vlm_worker_thread.start()
        else:
            logger.info("Phase 3 VLM (Ollama) is DISABLED. Running in detection & recording mode.")
            self.reasoner = None
            self.vlm_queue = None
            self.vlm_worker_thread = None
        
        self.conf_threshold = cfg['action']['threshold']
        self.alert_trigger_count = cfg['action'].get('alert_trigger_count', 2)
        
        self.safe_actions = cfg['action'].get('safe_prompts', []) + ['unknown_benign_activity']    
        self.ui_labels = cfg['action'].get('ui_labels', {})

        self.state_manager = SecurityStateManager(self.alert_trigger_count)
        
        queue_size = cfg['action'].get('queue_size', 1)
        self.analysis_queue = queue.Queue(maxsize=queue_size)
        self.running = True
        self.next_target_index = 0  
        self.last_phase2_dispatch = {}
        
        # Enable CuDNN benchmark if configured
        if cfg['system'].get('torch_benchmark', False) and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("Torch CuDNN benchmark mode enabled.")

        p2_dev = cfg['action'].get('device', cfg['system'].get('device', 'cpu'))
        self.p2_stream = torch.cuda.Stream() if (p2_dev == 'cuda' or 'cuda' in str(p2_dev)) and torch.cuda.is_available() else None
        
        # Start Background Worker
        self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker_thread.start()

        # --- Incident Recording & Screenshot Settings ---
        self.record_incidents = cfg['system'].get('record_incident', cfg['system'].get('record_indident', True))
        self.save_screenshots = cfg['system'].get('save_screenshot', True)
        self.fps_estimate = 30 # Default, will be updated in run()
        self.pre_buffer_size = cfg['system'].get('pre_event_seconds', 2) * self.fps_estimate
        self.post_buffer_size = cfg['system'].get('post_event_seconds', 3) * self.fps_estimate

        self.frame_buffer = deque(maxlen=self.pre_buffer_size)
        self.is_recording_incident = False
        self.post_alert_counter = 0
        self.incident_frames = []
        self.current_threat_id = None
        self.current_incident_path = None

    def reset_session(self):
        """Resets per-session tracking, ByteTrack identities, AI memory, and incident buffers."""
        if hasattr(self, 'eye') and hasattr(self.eye, 'reset_tracker'):
            self.eye.reset_tracker()
        self.memory = EvidenceManager()
        self.state_manager = SecurityStateManager(self.alert_trigger_count)
        self.frame_buffer.clear()
        self.last_phase2_dispatch.clear()
        self.next_target_index = 0
        self.incident_frames = []
        self.is_recording_incident = False
        self.post_alert_counter = 0
        self.current_threat_id = None
        self.current_incident_path = None
        self._screenshot_saved_for_event = False

        # Flush any pending tasks in the analysis queue
        while not self.analysis_queue.empty():
            try:
                self.analysis_queue.get_nowait()
                self.analysis_queue.task_done()
            except Exception:
                break

    def run(self, source_path, stop_event=None, frame_callback=None, auto_shutdown=True):
        """
        Runs the pipeline on the given source path using a clean, phase-based execution loop.
        Supports graceful stopping and streaming frame callbacks for web interfaces.
        """
        cap, w, h, out_dir = self._setup_environment(source_path)
        if not cap:
            return

        is_live = isinstance(source_path, int) or (isinstance(source_path, str) and source_path.strip().isdigit())
        playback_realtime = cfg['system'].get('video_playback_realtime', True)
        target_frame_time = (1.0 / max(1, self.fps_estimate)) if (not is_live and playback_realtime) else 0.0

        try:
            while self.running:
                if stop_event is not None and stop_event.is_set():
                    logger.info("Pipeline stopped by external signal.")
                    break

                frame_start = time.time()
                ret, frame = cap.read()
                if not ret: 
                    break
                
                # 0. Context Maintenance
                self.frame_buffer.append(frame.copy())
                
                # 1. Phase 1: Spatial Perception & Memory
                detections, ready_clips = self._run_phase1_perception(frame)
                
                # 2. Phase 2: Action Recognition Dispatch
                self._dispatch_phase2_analysis(ready_clips)
                
                # 3. Visualization Mapping
                out_frame = self.visualizer.draw(frame, detections, state_manager=self.state_manager)
                
                # 4. Phase 3 & Recording: Evidence Management
                self._handle_incident_recording(out_frame, w, h, out_dir)
                
                # 5. Web streaming callback
                if frame_callback is not None:
                    try:
                        frame_callback(out_frame)
                    except Exception as cb_err:
                        logger.debug(f"Frame callback error: {cb_err}")

                # 6. UI Rendering
                if not self._render_ui(out_frame):
                    break
                
                # For recorded video files, regulate playback to match original FPS
                if target_frame_time > 0:
                    elapsed = time.time() - frame_start
                    sleep_time = target_frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
        finally:
            if auto_shutdown:
                self._shutdown_pipeline(cap)
            else:
                self._close_active_stream(cap)

    def _analysis_worker(self):
        """
        Worker function to analyze clips in the background.
        """
        while self.running:
            try:
                tracker_id, clip = self.analysis_queue.get(timeout=0.1)
                if self.p2_stream is not None:
                    with torch.cuda.stream(self.p2_stream):
                        scores = self.brain.get_action_score(clip)
                else:
                    scores = self.brain.get_action_score(clip)
                
                if scores is None:
                    self.analysis_queue.task_done()
                    continue

                top_action = max(scores, key=scores.get)
                top_score = scores[top_action]                
                is_violent = (top_action not in self.safe_actions) and (top_score >= self.conf_threshold)
                
                display_name = self.ui_labels.get(top_action, top_action)
                
                # Delegate to the Brain
                self.state_manager.update_phase2(tracker_id, is_violent, display_name, top_score)
                
                # Trigger recording if the state manager escalated to Orange (Level 1) or higher
                current_state = self.state_manager.states.get(tracker_id)
                if current_state and current_state.level >= 1 and not self.is_recording_incident:
                    if self.record_incidents or self.save_screenshots:
                        self.is_recording_incident = True
                        self.post_alert_counter = self.post_buffer_size
                        self.current_threat_id = tracker_id # Remember who caused the recording
                    
                    # If Phase 3 VLM is disabled, confirm threat and lock to Red
                    if not self.vlm_enabled:
                        self.state_manager.confirm_threat(tracker_id, f"CONFIRMED: {display_name.upper()}")

                self.analysis_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker Error: {e}")

    def _vlm_worker(self):
        """
        Background thread that processes saved incident videos through the Ollama VLM.
        """
        while self.running:
            try:
                if self.vlm_queue is None or self.reasoner is None:
                    break
                # Sleep and wait for a completed video path to arrive
                video_path = self.vlm_queue.get(timeout=1.0)
                logger.info(f"Phase 3 Worker analyzing new evidence: {video_path}")
                
                # Send to Ollama (This takes a few seconds, but won't block the camera)
                report = self.reasoner.analyze_incident(video_path)
                
                # Save the JSON report right next to the video file
                report_path = video_path.replace('.mp4', '_report.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=4)

                # Upgrade: Feedback Loop to the UI
                threat_detected = report.get('threat_detected', False)
                summary = report.get('description', '')
                if hasattr(self, 'current_threat_id') and self.current_threat_id is not None:
                    self.state_manager.update_phase3(self.current_threat_id, threat_detected, summary)

                logger.info(f"Official Incident Report generated: {report_path}")
                self.vlm_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Phase 3 Worker Error: {e}")

    def _setup_environment(self, source_path):
        # Support numeric webcam string or integer
        if isinstance(source_path, int) or (isinstance(source_path, str) and source_path.strip().isdigit()):
            cam_idx = int(source_path)
            logger.info(f"Opening camera index: {cam_idx} with threaded zero-latency grabber...")
            cap = CameraStream(cam_idx)
        else:
            if not os.path.exists(source_path):
                logger.error(f"Input video file does not exist: {source_path}")
                return None, 0, 0, ""
            logger.info(f"Opening video file: {source_path}...")
            cap = cv2.VideoCapture(source_path)

        if not cap.isOpened(): 
            logger.error("Failed to open input video or camera source.")
            return None, 0, 0, ""

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.fps_estimate = fps if fps > 0 else 30
        
        out_dir = cfg['paths']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        disp_w, disp_h = cfg['system'].get('display_resolution', [1280, 720])
        logger.info(f"Pipeline started. Monitoring for incidents (Headless: {self.headless})...")
        
        if not self.headless:
            cv2.namedWindow("SentinAI Async System", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("SentinAI Async System", disp_w, disp_h)
        
        return cap, w, h, out_dir

    def _run_phase1_perception(self, frame):
        detections = self.detector.process_frame(frame)
        ready_clips = self.memory.update(frame, detections)
        return detections, ready_clips

    def _dispatch_phase2_analysis(self, ready_clips):
        if not ready_clips or self.is_recording_incident:
            return

        now = time.time()
        cooldown = cfg['action'].get('analysis_cooldown', 1.5)

        # Only analyze IDs that:
        # 1) Are not yet confirmed threats (level < 2)
        # 2) Have respected the cooldown timer
        available_ids = [
            tid for tid in ready_clips.keys()
            if (self.state_manager.states.get(tid) is None or self.state_manager.states[tid].level < 2)
            and (now - self.last_phase2_dispatch.get(tid, 0) >= cooldown)
        ]
        self.state_manager.cleanup(list(ready_clips.keys()))

        # Clean up stale dispatch timestamps for inactive IDs
        for tid in list(self.last_phase2_dispatch.keys()):
            if tid not in ready_clips:
                del self.last_phase2_dispatch[tid]

        if available_ids:
            idx = self.next_target_index % len(available_ids)
            target_id = available_ids[idx]
            try:
                self.analysis_queue.put_nowait((target_id, ready_clips[target_id]))
                self.last_phase2_dispatch[target_id] = now
                self.next_target_index += 1
            except queue.Full:
                pass

    def _handle_incident_recording(self, out_frame, w, h, out_dir):
        if not self.is_recording_incident:
            return

        # Start of incident: initialize path and copy pre-event buffer (instant in-memory copy)
        if not self.incident_frames and self.record_incidents:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.current_incident_path = os.path.join(out_dir, f"incident_{timestamp}.mp4")
            self.incident_frames = list(self.frame_buffer)
            logger.info(f"Incident triggered. Buffering evidence clip for {self.current_incident_path}")

        # Capture and save incident screenshot (dispatched to CPU thread to avoid disk I/O lag)
        if self.save_screenshots and not getattr(self, '_screenshot_saved_for_event', False):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = os.path.join(out_dir, f"incident_{timestamp}_screenshot.jpg")
            threading.Thread(
                target=cv2.imwrite,
                args=(screenshot_path, out_frame.copy()),
                daemon=True
            ).start()
            logger.info(f"Incident screenshot captured: {screenshot_path}")
            self._screenshot_saved_for_event = True
        
        # Append current frame to in-memory buffer (instant append, 0ms overhead)
        if self.record_incidents:
            self.incident_frames.append(out_frame.copy())
        self.post_alert_counter -= 1
        
        # Finalize clip when aftermath window closes
        if self.post_alert_counter <= 0:
            frames_to_save = self.incident_frames
            incident_path = self.current_incident_path
            fps = self.fps_estimate

            self.is_recording_incident = False
            self.incident_frames = []
            self._screenshot_saved_for_event = False

            if incident_path and frames_to_save:
                # Offload video encoding strictly to a background CPU worker
                threading.Thread(
                    target=self._save_incident_video_cpu,
                    args=(frames_to_save, incident_path, fps, w, h),
                    daemon=True
                ).start()

    def _save_incident_video_cpu(self, frames, incident_path, fps, w, h):
        """
        Encodes and writes the incident video file entirely on a background CPU thread.
        Uses PyAV with libx264 (ultrafast, yuv420p, +faststart) for instant browser compatibility
        and 0% GPU pressure. Zero impact on camera monitoring framerate.
        """
        logger.info(f"Background CPU worker encoding {len(frames)} frames to {incident_path}...")
        t0 = time.time()
        success = False

        # Attempt 1: PyAV libx264 with faststart (web-safe, moov at front, ~200 FPS on CPU)
        try:
            import av
            container = av.open(incident_path, mode='w', options={'movflags': '+faststart'})
            stream = container.add_stream('libx264', rate=max(1, int(fps)))
            stream.width = w
            stream.height = h
            stream.pix_fmt = 'yuv420p'
            stream.options = {'preset': 'ultrafast', 'crf': '23'}

            for f in frames:
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                av_frame = av.VideoFrame.from_ndarray(f, format='bgr24')
                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
            container.close()
            success = True
            logger.info(f"Incident clip saved with CPU PyAV H.264 (faststart) in {time.time()-t0:.2f}s: {incident_path}")
        except Exception as av_err:
            logger.warning(f"PyAV encoding encountered error: {av_err}. Falling back to OpenCV mp4v writer...")

        # Attempt 2: Fallback to OpenCV mp4v if PyAV failed
        if not success:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(incident_path, fourcc, max(1, fps), (w, h))
                for f in frames:
                    if f.shape[1] != w or f.shape[0] != h:
                        f = cv2.resize(f, (w, h))
                    writer.write(f)
                writer.release()
                success = True
                logger.info(f"Incident clip saved with OpenCV mp4v fallback: {incident_path}")
            except Exception as cv_err:
                logger.error(f"Failed to write incident video with OpenCV fallback: {cv_err}")

        # Handoff to Phase 3 VLM if enabled
        if success and self.vlm_enabled and self.vlm_queue is not None:
            self.vlm_queue.put(incident_path)

        # Non-blocking handoff to MongoDB Cloud Uploader (runs in background with delay)
        if success:
            try:
                from src.database.mongo_manager import get_mongo_uploader
                get_mongo_uploader().queue_incident({
                    "incident_id": os.path.splitext(os.path.basename(incident_path))[0],
                    "video_path": incident_path,
                    "screenshot_path": incident_path.replace('.mp4', '_screenshot.jpg'),
                    "report_path": incident_path.replace('.mp4', '_report.json'),
                    "threat_level": 2,
                    "threat_label": "VIOLENCE DETECTED",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as mongo_err:
                logger.debug(f"MongoDB queue handoff error: {mongo_err}")

    def _render_ui(self, out_frame):
        if self.headless:
            return True
        status_color = (0, 165, 255) if not self.analysis_queue.empty() else (0, 255, 0)
        if self.is_recording_incident: 
            status_color = (0, 0, 255) # Red for recording
        
        cv2.circle(out_frame, (30, 30), 10, status_color, -1) 
        cv2.imshow("SentinAI Async System", out_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False # Signal to break the loop
        return True

    def _close_active_stream(self, cap):
        """Releases the capture device and finalizes any active recording without stopping background workers."""
        if cap is not None:
            cap.release()

        # Graceful recording handoff if a stream was stopped mid-incident
        if self.is_recording_incident and self.incident_frames and self.current_incident_path:
            frames_to_save = self.incident_frames
            incident_path = self.current_incident_path
            fps = self.fps_estimate
            fw = frames_to_save[0].shape[1]
            fh = frames_to_save[0].shape[0]
            self.is_recording_incident = False
            self.incident_frames = []
            threading.Thread(
                target=self._save_incident_video_cpu,
                args=(frames_to_save, incident_path, fps, fw, fh),
                daemon=True
            ).start()

        if not self.headless:
            cv2.destroyAllWindows()

    def shutdown(self):
        """Terminates background worker threads and flushes final queues for full app exit."""
        self.running = False
        if self.vlm_enabled and self.vlm_queue is not None:
            logger.info("Waiting for Phase 3 VLM to finish generating final reports...")
            try:
                self.vlm_queue.join()
            except Exception:
                pass
        
        if not self.analysis_queue.empty():
            try:
                self.analysis_queue.join()
            except Exception:
                pass
        
        logger.info("System shutdown complete.")

    def _shutdown_pipeline(self, cap):
        self._close_active_stream(cap)
        self.shutdown()