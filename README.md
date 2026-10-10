# 🚨 Violence Action Detection

**Violence Action Detection** is a multi-modal, asynchronous Vision-Language surveillance pipeline. It detects violent physical altercations in real time, preserves persistent subject tracking, and automatically generates forensic evidence clips and detailed structured JSON reports.

---

## Overview & Example Output

When processing a video feed, the system detects individuals, tracks movement dynamics, and monitors for aggressive physical interactions. If an altercation occurs, it locks subject bounding boxes, buffers pre/post-event evidence clips, and triggers a Vision-Language Model (VLM) to generate a forensic incident report.

```json
{
    "threat_detected": true,
    "description": "Two individuals are engaged in a physical altercation. One person is wearing a red hoodie and camouflage pants, while the other is dressed in a white shirt and blue jeans. The person in the red hoodie appears to be pushing or shoving the other individual, who is trying to maintain balance and distance. The scene is set in an outdoor area with a house and parked cars in the background."
}
```

---

## ✨ Key Engineering Features

- **Asynchronous Pipeline**: Utilizes a Producer-Consumer threading model to decouple high-speed camera rendering from heavy AI inference, ensuring zero camera stutter or UI blocking.
- **"Infinite Latch" 3-State Memory**: A temporal memory layer (`SecurityStateManager`) tracks individuals across three states:
  - 🟢 **Green (Safe)**: Normal behavior detected.
  - 🟠 **Orange (Suspicious)**: Multi-strike escalation based on temporal action recognition. Triggers forensic clip buffering.
  - 🔴 **Red (Confirmed Threat)**: Confirmed violent altercation. Locks alert state and records pre/post-event incident evidence without prematurely resetting.
- **VLM Strategy Pattern**: The Vision-Language Model layer is decoupled via a Factory and Strategy pattern (`VisionReasonerFactory`), allowing seamless switching between:
  - Local Edge Models (`qwen2.5vl:3b` via Ollama).
  - Cloud API Models (`qwen-vl-max` via DashScope).
- **Zero-Shot Action Recognition**: Utilizes contrastive text-image pre-training (CoCa ViT-L-14) to dynamically classify behavior from natural language prompts without requiring custom action retraining.
- **Zero GPU Overhead Web Dashboard**: Browser-based command center with live streaming, dual-mode HTML5/CPU video playback, and automated background MongoDB archiving.

---

## 🧠 System Architecture (The 3 Phases)

The pipeline is split into three phases:

### 1. Phase 1: Spatial Perception (The Eyes)
* **Engines**: `YOLOv8` + `ByteTrack` (via `supervision`)
* **Role**: Runs on every frame. Detects humans (COCO Class 0), assigns persistent temporal Tracker IDs, and tracks bounding boxes.

### 2. Phase 2: Temporal Action Recognition (The Reflexes)
* **Engines**: `OpenCLIP (CoCa ViT-L-14)`
* **Role**: Extracts temporal micro-clips of tracked subjects asynchronously. Evaluates behavior against predefined text prompts (e.g., "a person punching", "people fighting", "normal behavior"). Escalates multi-strike detections to **Orange/Red** and triggers incident recording.

### 3. Phase 3: Forensic Reasoning (The Brain)
* **Engines**: `Qwen2.5-VL` (Local via Ollama) or `Qwen-VL-Max` (Cloud API)
* **Role**: Triggered after an incident recording completes. Extracts keyframes from the recorded footage and generates a structured **JSON Forensic Report**.

---

## 📁 Directory Structure

```text
 Violence-Action-Detection/
 ├── configs/
 │   └── config.yaml              # Base configurations for all phases
 ├── data/
 │   ├── inputs/                  # Source CCTV videos and uploads (.mp4)
 │   └── outputs/                 # Auto-generated incident MP4 clips, screenshots & reports
 ├── models/                      # Downloaded weights (yolov8n.pt, coca_l14.bin)
 ├── src/
 │   ├── core/
 │   │   ├── analysis/            # Phase 2 (Action Recognizer) & Phase 3 (VLM Factory)
 │   │   ├── memory/              # SecurityStateManager & Evidence buffers
 │   │   └── perception/          # Phase 1 YOLOv8 Detector & ByteTrack
 │   ├── database/                # MongoDB cloud archiver & GridFS uploader
 │   ├── pipelines/               # Rapid pipeline multi-threading logic
 │   ├── utils/                   # Config loader, logger, visualization helpers
 │   ├── web/                     # Web surveillance dashboard (Flask, MJPEG, CSS/JS)
 │   └── main.py                  # CLI pipeline entry-point
 ├── tests/                       # Unit and integration test suite
 ├── .env.example                 # Hardware and performance tuning template
 ├── thresholds.env.example       # Sensitivity and detection threshold template
 ├── requirements.txt             # Python dependencies
 ├── run_web.py                   # Web dashboard launcher
 └── README.md
```

---

## 🛠️ Step-by-Step Setup & Execution Guide

### Step 1: System Prerequisites
- **Operating System**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **Python**: **Python 3.11** (e.g., Python 3.11.9 is strongly recommended for full PyTorch CUDA 12.1 and OpenCLIP compatibility)
- **GPU**: NVIDIA GPU with CUDA drivers (recommended for real-time acceleration; safe CPU fallback is supported)
- **Ollama (Optional)**: Required only if enabling Phase 3 local VLM forensic reporting ([Download Ollama](https://ollama.com/))

---

### Step 2: Create a Python 3.11 Virtual Environment

Ensure you explicitly target **Python 3.11** when creating your virtual environment:

#### Windows (PowerShell / Command Prompt):
```powershell
# 1. Create virtual environment using the Python 3.11 launcher:
py -3.11 -m venv venv

# (Alternatively, if python3.11 is directly in your PATH:)
# python3.11 -m venv venv

# 2. Activate the virtual environment in PowerShell:
.\venv\Scripts\Activate.ps1

# Or in Command Prompt (cmd.exe):
# venv\Scripts\activate.bat
```

#### Linux / macOS:
```bash
# 1. Create virtual environment with Python 3.11:
python3.11 -m venv venv

# 2. Activate the virtual environment:
source venv/bin/activate
```

Verify your active Python version:
```bash
python --version
# Expected output: Python 3.11.x
```

---

### Step 3: Install Dependencies

1. **Install PyTorch with CUDA 12.1 acceleration** (recommended for NVIDIA GPUs):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   *(For systems without an NVIDIA GPU, run `pip install torch torchvision torchaudio` for CPU mode)*

2. **Install project dependencies** (YOLOv8, OpenCLIP, Supervision, Flask, PyAV, PyMongo):
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 4: Model Weights Setup

> [!NOTE]
> `pip install -r requirements.txt` installs the Python software libraries only; it does **not** download model binary weights.

#### Option A: Automated Download (Recommended)
Run the built-in downloader script. It automatically checks `models/` and downloads any missing weights with a progress bar:
```bash
python scripts/download_models.py
```

#### Option B: Manual Download & Placement
If downloading manually, place the following files into the `models/` directory:

1. **YOLOv8 Nano Person Detection** (`models/yolov8n.pt`, ~6.5 MB):
   - **Direct URL**: [https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)
   - PowerShell:
     ```powershell
     Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" -OutFile "models/yolov8n.pt"
     ```
   - *(Note: If missing, Ultralytics will also attempt to auto-download this on first inference).*

2. **OpenCLIP CoCa Action Recognition** (`models/coca_l14.bin`, ~2.55 GB):
   - **Direct URL**: [Hugging Face CoCa ViT-L-14 Weights](https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin)
   - PowerShell:
     ```powershell
     Invoke-WebRequest -Uri "https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin" -OutFile "models/coca_l14.bin"
     ```
   - Linux / macOS / Git Bash:
     ```bash
     curl -L -o models/coca_l14.bin "https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin"
     ```

3. **Phase 3 Local VLM (Optional)**:
   - If utilizing local forensic reasoning, pull the model via Ollama:
     ```bash
     ollama pull qwen2.5vl:3b
     ```

---

### Step 5: Configure Hardware & Detection Profiles

The system separates hardware resource settings (`.env`) from detection sensitivity thresholds (`thresholds.env`):

1. **Copy the hardware configuration template**:
   ```bash
   # Windows PowerShell:
   Copy-Item .env.example .env

   # Linux / Bash:
   cp .env.example .env
   ```
   - **Lower-VRAM GPUs (4GB VRAM or below)**: Keep detection on `cuda` and action recognition on `cpu` (`ACTION_DEVICE=cpu`, `ACTION_SAMPLE_FRAMES=1`).
   - **Mid-to-High Tier GPUs (6GB-8GB+ VRAM)**: Set `ACTION_DEVICE=cuda`, `ACTION_SAMPLE_FRAMES=3`, and `ACTION_ANALYSIS_COOLDOWN=0.4` for 30x faster inference and maximum accuracy.

2. **(Optional) Configure Detection Thresholds**:
   ```bash
   # Windows PowerShell:
   Copy-Item thresholds.env.example thresholds.env

   # Linux / Bash:
   cp thresholds.env.example thresholds.env
   ```
   Tune IoU thresholds, confidence gates, motion filters, or strike trigger counts independently.

---

### Step 6: Verify Configuration & Smoke Test

Execute the environment test script to verify that CUDA, PyTorch, models, and configuration overrides load cleanly:
```bash
python tests/test_env.py
```
*Expected: Exit code 0 with `--- Environment Check Passed ---`.*

---

### Step 7: Run the Surveillance System

#### Option A: Web Surveillance Command Center (Recommended)
Launch the browser-based control dashboard (runs on CPU threads with zero GPU overhead):
```bash
python run_web.py
```
Open **`http://localhost:5000`** in your browser:
- **Live Monitoring**: Click **"Start Live Stream"** to begin real-time monitoring of your camera feed.
- **Video Analysis**: Click **"Upload Video File"** to run analysis on pre-recorded footage.
- **Evidence Locker**: Inspect recorded altercations with high-resolution keyframe snapshots and dual-mode video playback (**Direct HTML5** or **CPU MJPEG Stream**).
- **Reset Feed & Buffers**: Click the toolbar **"Reset Feed & Buffers"** button to flush ByteTrack tracking IDs, threat alarms, and temporal memory buffers on the fly.

#### Option B: Standalone CLI Pipeline
Run directly in the terminal with OpenCV window display:
```bash
# Monitor default camera (webcam index 0):
python -m src.main

# Or analyze a video file directly:
python -m src.main --source data/inputs/V_120.mp4
```

---

### Step 8: MongoDB Cloud Archiving (Optional)
To automatically archive forensic video clips, keyframe screenshots, and incident telemetry to MongoDB:
1. Open `.env` and configure your MongoDB URI:
   ```env
   MONGODB_ENABLED=true
   MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/sentinai_surveillance?retryWrites=true&w=majority
   ```
2. Video recordings are saved to GridFS (`incident_videos`) and metadata is stored in the `incidents` collection.
3. Uploads execute asynchronously on a background CPU thread with a 2-second pacing delay to guarantee **0% impact on camera framerate and detection accuracy**.

