import os
import time
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.web.stream_manager import WebStreamManager

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Upload configuration
UPLOAD_FOLDER = os.path.join("data", "inputs", "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload limit

stream_manager = WebStreamManager.get_instance()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Main Surveillance Dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG streaming route."""
    def generate():
        while True:
            frame_bytes = stream_manager.get_latest_frame()
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.033)  # ~30 FPS delivery
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns real-time pipeline status and threat indicators."""
    status = stream_manager.get_status()
    return jsonify(status)


@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    """Starts monitoring on a camera index or video file path."""
    data = request.get_json(silent=True) or {}
    source = data.get("source", 0)

    # Convert numeric strings like "0" or "1" to integer camera index
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())

    success = stream_manager.start_stream(source)
    return jsonify({
        "success": success,
        "status": stream_manager.status,
        "message": stream_manager.status_message,
        "source": stream_manager.current_source_name
    })


@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    """Stops the active video or camera stream."""
    stream_manager.stop_stream()
    return jsonify({
        "success": True,
        "status": stream_manager.status,
        "message": stream_manager.status_message
    })


@app.route("/api/feed/reset", methods=["POST"])
def reset_feed():
    """Clears all tracking memory, threat alarms, motion buffers, and frame caches."""
    stream_manager.reset_feed()
    incidents = stream_manager.get_incidents()
    return jsonify({
        "success": True,
        "message": "Feed buffers, tracking states, and threat alarms have been cleared.",
        "incidents": incidents
    })


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Uploads a video and optionally starts analysis immediately."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False, 
            "error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    filename = secure_filename(file.filename)
    timestamp_prefix = time.strftime("%Y%m%d_%H%M%S_")
    saved_filename = timestamp_prefix + filename
    dest_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
    
    file.save(dest_path)
    logger.info(f"Video uploaded successfully to {dest_path}")

    auto_start = request.form.get("auto_start", "true").lower() == "true"
    if auto_start:
        stream_manager.start_stream(dest_path)

    return jsonify({
        "success": True,
        "filename": saved_filename,
        "filepath": dest_path,
        "started": auto_start
    })


@app.route("/api/sample_videos", methods=["GET"])
def get_sample_videos():
    """Returns available sample videos in data/inputs."""
    samples = []
    inputs_dir = cfg['paths'].get('input_source', 'data/inputs')
    if os.path.isfile(inputs_dir):
        inputs_dir = os.path.dirname(inputs_dir)
    elif not os.path.exists(inputs_dir):
        inputs_dir = "data/inputs"

    if os.path.exists(inputs_dir):
        for root, _, files in os.walk(inputs_dir):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, ".").replace("\\", "/")
                    samples.append({
                        "name": f,
                        "path": rel_p,
                        "size_mb": round(os.path.getsize(full_p) / (1024 * 1024), 2)
                    })
    return jsonify(samples)


@app.route("/api/incidents", methods=["GET"])
def list_incidents():
    """Returns list of recorded forensic incidents."""
    incidents = stream_manager.get_incidents()
    return jsonify(incidents)


@app.route("/api/incidents/screenshot/<path:filename>")
def get_screenshot(filename):
    """Serves an incident screenshot image."""
    out_dir = os.path.abspath(cfg['paths'].get('output_dir', 'data/outputs'))
    return send_from_directory(out_dir, filename, mimetype="image/jpeg")


@app.route("/api/incidents/video/<path:filename>")
def get_video(filename):
    """Serves an incident evidence MP4 clip with full HTTP Range (206) support."""
    out_dir = os.path.abspath(cfg['paths'].get('output_dir', 'data/outputs'))
    return send_from_directory(out_dir, filename, mimetype="video/mp4", conditional=True)


@app.route("/api/incidents/stream/<path:filename>")
def stream_incident_video(filename):
    """
    CPU-only fallback video stream.
    Decodes video frames on CPU and streams as MJPEG if browser lacks native MP4 codec support.
    Zero GPU pressure.
    """
    out_dir = os.path.abspath(cfg['paths'].get('output_dir', 'data/outputs'))
    video_path = os.path.join(out_dir, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    def generate():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / max(1.0, fps)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(delay)
        finally:
            cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/incidents/report/<path:filename>")
def get_report(filename):
    """Serves the forensic JSON report."""
    out_dir = os.path.abspath(cfg['paths'].get('output_dir', 'data/outputs'))
    return send_from_directory(out_dir, filename, mimetype="application/json")


@app.route("/api/incidents/<path:incident_id>", methods=["DELETE", "POST"])
def delete_incident(incident_id):
    """Deletes all evidence files associated with an incident ID."""
    success, message = stream_manager.delete_incident(incident_id)
    if success:
        return jsonify({"success": True, "message": message})
    return jsonify({"success": False, "error": message}), 404


@app.route("/api/mongodb/status", methods=["GET"])
def get_mongodb_status():
    """Returns live MongoDB cloud archiving and connection status."""
    from src.database.mongo_manager import get_mongo_uploader
    return jsonify(get_mongo_uploader().get_status())


@app.route("/api/mongodb/sync", methods=["POST"])
def sync_mongodb_archive():
    """Queues all existing local incident recordings for background MongoDB archiving."""
    from src.database.mongo_manager import get_mongo_uploader
    uploader = get_mongo_uploader()
    incidents = stream_manager.get_incidents()
    out_dir = os.path.abspath(cfg['paths'].get('output_dir', 'data/outputs'))
    queued_count = 0

    for inc in incidents:
        vid_p = os.path.join(out_dir, inc["video"]) if inc.get("video") else None
        img_p = os.path.join(out_dir, inc["screenshot"]) if inc.get("screenshot") else None
        rep_p = os.path.join(out_dir, f"{inc['id']}_report.json")
        uploader.queue_incident({
            "incident_id": inc["id"],
            "video_path": vid_p,
            "screenshot_path": img_p,
            "report_path": rep_p,
            "threat_level": inc.get("threat_level", 2),
            "threat_label": inc.get("threat_label", "VIOLENCE DETECTED"),
            "timestamp": inc.get("timestamp", "")
        })
        queued_count += 1

    return jsonify({
        "success": True,
        "message": f"Queued {queued_count} incidents for background MongoDB upload.",
        "status": uploader.get_status()
    })



def create_app():
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting SentinAI Web Surveillance Dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
