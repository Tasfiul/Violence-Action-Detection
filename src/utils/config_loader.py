import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _parse_int(val, default=None):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _parse_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class ConfigLoader:
    """
    Singleton class to load and access configuration settings.
    Merges configs/config.yaml with machine-specific .env environment variables.
    """
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "configs" / "config.yaml"
        thresholds_path = base_dir / "thresholds.env"
        env_path = base_dir / ".env"

        if not config_path.exists():
            raise FileNotFoundError(f"[CRITICAL] Config file missing at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        # 1. Load detection thresholds from thresholds.env if available
        thresholds_loaded = False
        if thresholds_path.exists():
            load_dotenv(dotenv_path=thresholds_path, override=True)
            thresholds_loaded = True

        # 2. Load machine/hardware settings from .env if available
        env_loaded = False
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            env_loaded = True
        else:
            load_dotenv(override=True)

        # 3. Overlay environment variables onto YAML configuration
        overrides = self._apply_env_overrides()

        # 4. Inform user of active configuration
        print(f"[SYSTEM] Base configuration loaded from {config_path}")
        if thresholds_loaded:
            print(f"[SYSTEM] Detection thresholds loaded from {thresholds_path}")
        if env_loaded:
            print(f"[SYSTEM] Machine-specific environment loaded from {env_path}")
        if overrides:
            print(f"[SYSTEM] Applied {len(overrides)} configuration overrides")

    def _apply_env_overrides(self):
        """
        Maps environment variables to the nested config dictionary.
        Returns a list of keys that were overridden.
        """
        overridden = []

        def set_nested(section, key, val):
            if section not in self._config or not isinstance(self._config[section], dict):
                self._config[section] = {}
            self._config[section][key] = val
            overridden.append(f"{section}.{key} = {val}")

        # System settings
        if "SYSTEM_DEVICE" in os.environ:
            set_nested("system", "device", os.environ["SYSTEM_DEVICE"])
        if "LOG_LEVEL" in os.environ:
            set_nested("system", "log_level", os.environ["LOG_LEVEL"].upper())
        if "HEADLESS" in os.environ:
            set_nested("system", "headless", _parse_bool(os.environ["HEADLESS"]))
        if "RECORD_INCIDENT" in os.environ:
            set_nested("system", "record_incident", _parse_bool(os.environ["RECORD_INCIDENT"]))
        if "SAVE_SCREENSHOT" in os.environ:
            set_nested("system", "save_screenshot", _parse_bool(os.environ["SAVE_SCREENSHOT"]))
        if "PRE_EVENT_SECONDS" in os.environ:
            set_nested("system", "pre_event_seconds", _parse_int(os.environ["PRE_EVENT_SECONDS"], 2))
        if "POST_EVENT_SECONDS" in os.environ:
            set_nested("system", "post_event_seconds", _parse_int(os.environ["POST_EVENT_SECONDS"], 3))
        if "VIDEO_PLAYBACK_REALTIME" in os.environ:
            set_nested("system", "video_playback_realtime", _parse_bool(os.environ["VIDEO_PLAYBACK_REALTIME"]))
        if "TORCH_BENCHMARK" in os.environ:
            set_nested("system", "torch_benchmark", _parse_bool(os.environ["TORCH_BENCHMARK"]))

        # Display resolution
        disp_w = os.environ.get("DISPLAY_WIDTH")
        disp_h = os.environ.get("DISPLAY_HEIGHT")
        if disp_w or disp_h:
            curr_res = self._config.get("system", {}).get("display_resolution", [1280, 720])
            w = _parse_int(disp_w, curr_res[0] if isinstance(curr_res, (list, tuple)) else 1280)
            h = _parse_int(disp_h, curr_res[1] if isinstance(curr_res, (list, tuple)) and len(curr_res) > 1 else 720)
            set_nested("system", "display_resolution", [w, h])

        # Paths
        if "INPUT_SOURCE" in os.environ:
            set_nested("paths", "input_source", os.environ["INPUT_SOURCE"])
        if "OUTPUT_DIR" in os.environ:
            set_nested("paths", "output_dir", os.environ["OUTPUT_DIR"])
        if "MODEL_DIR" in os.environ:
            set_nested("paths", "model_dir", os.environ["MODEL_DIR"])

        # Phase 1: Detection & Tracking
        if "DETECTION_DEVICE" in os.environ:
            set_nested("detection", "device", os.environ["DETECTION_DEVICE"])
        if "DETECTION_MODEL_WEIGHTS" in os.environ:
            set_nested("detection", "model_weights", os.environ["DETECTION_MODEL_WEIGHTS"])
        if "DETECTION_CONF_THRESHOLD" in os.environ:
            set_nested("detection", "confidence_threshold", _parse_float(os.environ["DETECTION_CONF_THRESHOLD"], 0.5))
        if "DETECTION_IOU_THRESHOLD" in os.environ:
            set_nested("detection", "iou_threshold", _parse_float(os.environ["DETECTION_IOU_THRESHOLD"], 0.45))
        if "DETECTION_IMGSZ" in os.environ:
            set_nested("detection", "imgsz", _parse_int(os.environ["DETECTION_IMGSZ"], 640))
        if "DETECTION_HALF_PRECISION" in os.environ:
            set_nested("detection", "half", _parse_bool(os.environ["DETECTION_HALF_PRECISION"]))
        if "DETECTION_FRAME_SKIP" in os.environ:
            set_nested("detection", "frame_skip", max(1, _parse_int(os.environ["DETECTION_FRAME_SKIP"], 1)))
        if "TRACKER_ACTIVATION_THRESHOLD" in os.environ:
            set_nested("detection", "track_activation_threshold", _parse_float(os.environ["TRACKER_ACTIVATION_THRESHOLD"], 0.25))
        if "TRACKER_LOST_BUFFER" in os.environ:
            set_nested("detection", "lost_track_buffer", _parse_int(os.environ["TRACKER_LOST_BUFFER"], 30))
        if "TRACKER_MIN_MATCHING_THRESHOLD" in os.environ:
            set_nested("detection", "minimum_matching_threshold", _parse_float(os.environ["TRACKER_MIN_MATCHING_THRESHOLD"], 0.8))
        if "TRACKER_FRAME_RATE" in os.environ:
            set_nested("detection", "tracker_frame_rate", _parse_int(os.environ["TRACKER_FRAME_RATE"], 30))

        # Phase 2: Action Recognition (CoCa ViT-L-14)
        if "ACTION_DEVICE" in os.environ:
            set_nested("action", "device", os.environ["ACTION_DEVICE"])
        if "ACTION_MODEL_NAME" in os.environ:
            set_nested("action", "model_name", os.environ["ACTION_MODEL_NAME"])
        if "ACTION_WEIGHTS_PATH" in os.environ:
            set_nested("action", "weights_path", os.environ["ACTION_WEIGHTS_PATH"])
        if "ACTION_SAMPLE_FRAMES" in os.environ:
            set_nested("action", "sample_frames", _parse_int(os.environ["ACTION_SAMPLE_FRAMES"], 1))
        if "ACTION_WINDOW_SIZE" in os.environ:
            set_nested("action", "window_size", _parse_int(os.environ["ACTION_WINDOW_SIZE"], 8))
        if "ACTION_STRIDE" in os.environ:
            set_nested("action", "stride", _parse_int(os.environ["ACTION_STRIDE"], 4))
        if "ACTION_MOTION_THRESHOLD" in os.environ:
            set_nested("action", "motion_threshold", _parse_float(os.environ["ACTION_MOTION_THRESHOLD"], 15.0))
        if "ACTION_ANALYSIS_COOLDOWN" in os.environ:
            set_nested("action", "analysis_cooldown", _parse_float(os.environ["ACTION_ANALYSIS_COOLDOWN"], 1.5))
        if "ACTION_CONFIDENCE_THRESHOLD" in os.environ:
            set_nested("action", "threshold", _parse_float(os.environ["ACTION_CONFIDENCE_THRESHOLD"], 0.60))
        if "ACTION_ALERT_TRIGGER_COUNT" in os.environ:
            set_nested("action", "alert_trigger_count", _parse_int(os.environ["ACTION_ALERT_TRIGGER_COUNT"], 2))
        if "ACTION_OSR_THRESHOLD" in os.environ:
            osr_val = _parse_float(os.environ["ACTION_OSR_THRESHOLD"], 0.25)
            set_nested("action", "osr_threshold", osr_val)
            set_nested("action", "ors_threshold", osr_val)
        if "ACTION_QUEUE_SIZE" in os.environ:
            set_nested("action", "queue_size", _parse_int(os.environ["ACTION_QUEUE_SIZE"], 1))
        if "ACTION_CROP_CONTEXT_RATIO" in os.environ:
            set_nested("action", "crop_context_ratio", _parse_float(os.environ["ACTION_CROP_CONTEXT_RATIO"], 0.20))
        if "ACTION_HALF_PRECISION" in os.environ:
            set_nested("action", "half_precision", _parse_bool(os.environ["ACTION_HALF_PRECISION"]))

        # State Manager
        if "STATE_TIMEOUT_SECONDS" in os.environ:
            set_nested("state", "timeout_seconds", _parse_float(os.environ["STATE_TIMEOUT_SECONDS"], 10.0))

        # Phase 3: Vision-Language Model
        if "VLM_ENABLED" in os.environ:
            set_nested("vlm", "enabled", _parse_bool(os.environ["VLM_ENABLED"]))
        if "VLM_PROVIDER" in os.environ:
            set_nested("vlm", "provider", os.environ["VLM_PROVIDER"])
        if "VLM_MODEL_ID" in os.environ:
            set_nested("vlm", "model_id", os.environ["VLM_MODEL_ID"])
        if "VLM_CLOUD_MODEL_ID" in os.environ:
            set_nested("vlm", "cloud_model_id", os.environ["VLM_CLOUD_MODEL_ID"])
        if "VLM_NUM_FRAMES" in os.environ:
            if "extraction" not in self._config.setdefault("vlm", {}):
                self._config["vlm"]["extraction"] = {}
            self._config["vlm"]["extraction"]["num_frames"] = _parse_int(os.environ["VLM_NUM_FRAMES"], 8)
            overridden.append(f"vlm.extraction.num_frames = {os.environ['VLM_NUM_FRAMES']}")
        if "VLM_RESIZE_WIDTH" in os.environ:
            if "extraction" not in self._config.setdefault("vlm", {}):
                self._config["vlm"]["extraction"] = {}
            self._config["vlm"]["extraction"]["resize_width"] = _parse_int(os.environ["VLM_RESIZE_WIDTH"], 480)
            overridden.append(f"vlm.extraction.resize_width = {os.environ['VLM_RESIZE_WIDTH']}")
        if "VLM_RESIZE_HEIGHT" in os.environ:
            if "extraction" not in self._config.setdefault("vlm", {}):
                self._config["vlm"]["extraction"] = {}
            self._config["vlm"]["extraction"]["resize_height"] = _parse_int(os.environ["VLM_RESIZE_HEIGHT"], 270)
            overridden.append(f"vlm.extraction.resize_height = {os.environ['VLM_RESIZE_HEIGHT']}")
        if "VLM_JPEG_QUALITY" in os.environ:
            if "extraction" not in self._config.setdefault("vlm", {}):
                self._config["vlm"]["extraction"] = {}
            self._config["vlm"]["extraction"]["jpeg_quality"] = _parse_int(os.environ["VLM_JPEG_QUALITY"], 80)
            overridden.append(f"vlm.extraction.jpeg_quality = {os.environ['VLM_JPEG_QUALITY']}")

        # Web Server
        if "WEB_HOST" in os.environ:
            set_nested("web", "host", os.environ["WEB_HOST"])
        if "WEB_PORT" in os.environ:
            set_nested("web", "port", _parse_int(os.environ["WEB_PORT"], 5000))
        if "WEB_STREAM_JPEG_QUALITY" in os.environ:
            set_nested("web", "jpeg_quality", _parse_int(os.environ["WEB_STREAM_JPEG_QUALITY"], 80))

        # Database / MongoDB Cloud Storage
        if "MONGODB_ENABLED" in os.environ:
            set_nested("mongodb", "enabled", _parse_bool(os.environ["MONGODB_ENABLED"]))
        if "MONGODB_URI" in os.environ:
            set_nested("mongodb", "uri", os.environ["MONGODB_URI"])
        if "MONGODB_UPLOAD_DELAY_SECONDS" in os.environ:
            set_nested("mongodb", "upload_delay_seconds", _parse_float(os.environ["MONGODB_UPLOAD_DELAY_SECONDS"], 2.0))
        if "MONGODB_UPLOAD_VIDEOS" in os.environ:
            set_nested("mongodb", "upload_videos", _parse_bool(os.environ["MONGODB_UPLOAD_VIDEOS"]))
        if "MONGODB_UPLOAD_SCREENSHOTS" in os.environ:
            set_nested("mongodb", "upload_screenshots", _parse_bool(os.environ["MONGODB_UPLOAD_SCREENSHOTS"]))
        if "MONGODB_TIMEOUT_MS" in os.environ:
            set_nested("mongodb", "timeout_ms", _parse_int(os.environ["MONGODB_TIMEOUT_MS"], 5000))

        return overridden

    @property
    def config(self):
        return self._config

    def reload(self):
        """Forces reloading configuration from config.yaml and .env."""
        self._load_config()
        return self._config


# Expose a single global instance
cfg = ConfigLoader().config