import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
import requests
import base64
import time
import os
import subprocess
import tempfile
import collections
import io
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
from datetime import datetime
from flask import Flask, request
import logging

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "minicpm-v"
WEBHOOK_PORT = 8765
SAVE_DIR = Path(os.path.expanduser("~")) / "SecurityEvents"
SAVE_DIR.mkdir(exist_ok=True)

RTSP_URL = "rtsp://192.168.0.166:7447/YD4arutidcyKjQvI"
STREAM_PREVIEW_INTERVAL = 3000  # ms between live preview refreshes

# Buffer config
BUFFER_SECONDS = 2.0          # how many seconds of frames to keep
FRAME_INTERVAL = 0.5          # grab a frame every N seconds for buffer
SNAP_BACK_SECS = 1.0          # go back this far on webhook
SNAP_FORWARD_SECS = 0.5       # then grab a frame this far forward
CROP_PADDING = 50             # px padding around bounding box

DEBUG_MODE = True             # show debug windows

DEFAULT_PROMPT = """You are a security camera analyzer. Look for people, vehicles, and animals ONLY.

For each one found, describe: count, type, appearance, and behavior.

If none are present, respond with only: CLEAR"""

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

analysis_queue = queue.Queue()

# --- Rolling frame buffer: deque of (timestamp, image_bytes) ---
frame_buffer = collections.deque()
buffer_lock = threading.Lock()

# ---------------------------------------------------------------
# FRAME BUFFER WORKER
# Continuously grabs frames via FFmpeg and stores in rolling buffer
# ---------------------------------------------------------------
def grab_raw_frame():
    """Grab one raw JPEG from RTSP. Returns bytes or None."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", RTSP_URL,
            "-frames:v", "1",
            "-q:v", "2",
            "-update", "1",
            "-y",
            tmp_path
        ]
        stderr_pipe = None if DEBUG_MODE else subprocess.DEVNULL
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=stderr_pipe, timeout=15)
        if result.returncode == 0 and Path(tmp_path).exists():
            data = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
            return data
        Path(tmp_path).unlink(missing_ok=True)
        return None
    except Exception as e:
        if DEBUG_MODE:
            print(f"[RTSP] grab_raw_frame error: {e}")
        return None

def buffer_worker():
    """Continuously grab frames and maintain rolling buffer."""
    _fail_count = 0
    while True:
        t0 = time.time()
        data = grab_raw_frame()
        if data:
            _fail_count = 0
            ts = time.time()
            with buffer_lock:
                frame_buffer.append((ts, data))
                # Prune frames older than BUFFER_SECONDS
                cutoff = ts - BUFFER_SECONDS
                while frame_buffer and frame_buffer[0][0] < cutoff:
                    frame_buffer.popleft()
            # Update live preview in UI
            b64 = base64.b64encode(data).decode()
            root.after(0, lambda b=b64: update_stream_preview(b))
        else:
            _fail_count += 1
            if DEBUG_MODE:
                print(f"[RTSP] Frame grab failed (attempt {_fail_count}) — {RTSP_URL}")
            root.after(0, lambda n=_fail_count: stream_status_var.set(
                f"🔴 Disconnected — retrying... (attempt {n})"
            ))
        elapsed = time.time() - t0
        sleep = max(0, FRAME_INTERVAL - elapsed)
        time.sleep(sleep)

def get_frame_at(target_ts):
    """Get the buffered frame closest to target_ts. Returns bytes or None."""
    with buffer_lock:
        if not frame_buffer:
            return None
        best = min(frame_buffer, key=lambda x: abs(x[0] - target_ts))
        return best[1]

# ---------------------------------------------------------------
# MOTION DETECTION & BOUNDING BOX
# ---------------------------------------------------------------
def compute_motion_crop(frame_a_bytes, frame_b_bytes):
    """
    Diff two JPEG frames. Returns:
      - cropped_bytes: JPEG of the motion crop from frame_b
      - debug_images: dict of labeled PIL images for debug view
      - bbox: (x1, y1, x2, y2) or None
    """
    # Decode to numpy
    arr_a = np.frombuffer(frame_a_bytes, dtype=np.uint8)
    arr_b = np.frombuffer(frame_b_bytes, dtype=np.uint8)
    img_a = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        return None, {}, None

    # Resize to same size if different (shouldn't happen but safety)
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    h, w = img_a.shape[:2]

    # Grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur_a = cv2.GaussianBlur(gray_a, (21, 21), 0)
    blur_b = cv2.GaussianBlur(gray_b, (21, 21), 0)

    # Absolute diff
    diff = cv2.absdiff(blur_a, blur_b)

    # Threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small noise contours (< 0.05% of frame area)
    min_area = w * h * 0.0005
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    bbox = None
    cropped_bytes = None

    if contours:
        # Bounding box enclosing all motion contours
        x1 = min(cv2.boundingRect(c)[0] for c in contours)
        y1 = min(cv2.boundingRect(c)[1] for c in contours)
        x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

        # Add padding, clamp to frame
        x1p = max(0, x1 - CROP_PADDING)
        y1p = max(0, y1 - CROP_PADDING)
        x2p = min(w, x2 + CROP_PADDING)
        y2p = min(h, y2 + CROP_PADDING)
        bbox = (x1p, y1p, x2p, y2p)

        # Crop from frame_b
        crop = img_b[y1p:y2p, x1p:x2p]
        _, crop_enc = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cropped_bytes = crop_enc.tobytes()

    # --- Debug images ---
    debug = {}

    def cv2_to_pil(img, gray=False):
        if gray:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    debug["Frame A\n(t-1s)"] = cv2_to_pil(img_a)
    debug["Frame B\n(t+0.5s)"] = cv2_to_pil(img_b)
    debug["Diff"] = cv2_to_pil(diff, gray=True)
    debug["Threshold"] = cv2_to_pil(thresh, gray=True)
    debug["Dilated\nmask"] = cv2_to_pil(dilated, gray=True)

    # Draw bounding box on frame B copy
    img_b_annot = img_b.copy()
    if bbox:
        x1p, y1p, x2p, y2p = bbox
        cv2.rectangle(img_b_annot, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)
        # Draw all contours in red
        cv2.drawContours(img_b_annot, contours, -1, (0, 0, 255), 2)
    debug["Bounding\nbox"] = cv2_to_pil(img_b_annot)

    if cropped_bytes:
        crop_pil = Image.open(io.BytesIO(cropped_bytes))
        debug["AI\nCrop"] = crop_pil

    return cropped_bytes, debug, bbox

# ---------------------------------------------------------------
# DEBUG WINDOW
# ---------------------------------------------------------------
debug_window = None
debug_labels = {}

def show_debug_window(debug_images):
    """Show/update a Toplevel window with all debug frames."""
    global debug_window, debug_labels

    if not DEBUG_MODE:
        return

    if debug_window is None or not debug_window.winfo_exists():
        debug_window = tk.Toplevel(root)
        debug_window.title("Debug — Motion Pipeline")
        debug_window.configure(bg="#0a0a0a")
        debug_labels = {}

    # Clear old widgets
    for w in debug_window.winfo_children():
        w.destroy()
    debug_labels = {}

    cols = 4
    thumb_w, thumb_h = 280, 160

    for i, (label, pil_img) in enumerate(debug_images.items()):
        row, col = divmod(i, cols)
        frame = tk.Frame(debug_window, bg="#0a0a0a")
        frame.grid(row=row, column=col, padx=6, pady=6)

        tk.Label(frame, text=label, bg="#0a0a0a", fg="#555555",
                 font=("Courier New", 8, "bold")).pack()

        img_copy = pil_img.copy()
        img_copy.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        # Convert to RGB if grayscale
        if img_copy.mode != "RGB":
            img_copy = img_copy.convert("RGB")
        photo = ImageTk.PhotoImage(img_copy)
        lbl = tk.Label(frame, image=photo, bg="#111111")
        lbl.image = photo
        lbl.pack()

    debug_window.lift()

# ---------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------
def save_event_background(frame_a, frame_b, cropped, description, bbox, ts):
    def _save():
        try:
            safe_ts = ts.replace(":", "-").replace(" ", "_")
            folder = SAVE_DIR / safe_ts
            folder.mkdir(exist_ok=True)
            if frame_a:
                (folder / "frame_a.jpg").write_bytes(frame_a)
            if frame_b:
                (folder / "frame_b.jpg").write_bytes(frame_b)
            if cropped:
                (folder / "crop.jpg").write_bytes(cropped)
            (folder / "description.txt").write_text(
                f"Time: {ts}\nBBox: {bbox}\n\n{description}"
            )
        except Exception as e:
            print(f"[Save error] {e}")
    threading.Thread(target=_save, daemon=True).start()

# ---------------------------------------------------------------
# OLLAMA
# ---------------------------------------------------------------
def fetch_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            model_dropdown["values"] = models
            if "minicpm-v" in models:
                model_var.set("minicpm-v")
            else:
                model_var.set(models[0])
    except Exception as e:
        status_var.set(f"⚠️ Could not fetch models: {e}")

def warmup_model():
    status_var.set("⏳ Loading models...")
    fetch_models()
    try:
        requests.post(OLLAMA_URL, json={
            "model": model_var.get(),
            "prompt": "ready",
            "stream": False,
            "keep_alive": -1
        }, timeout=60)
        update_queue_status()
    except Exception as e:
        status_var.set(f"⚠️ Warmup failed: {e}")

def analyze_image_bytes(image_bytes, prompt, model):
    image_b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": -1
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "No response received.")

def update_queue_status():
    qsize = analysis_queue.qsize()
    buf_size = len(frame_buffer)
    if qsize > 0:
        status_var.set(f"✅ Ready — webhook :{WEBHOOK_PORT} — 📋 {qsize} queued — 🎞 {buf_size} frames buffered")
    else:
        status_var.set(f"✅ Ready — webhook :{WEBHOOK_PORT} — 🎞 {buf_size} frames buffered")

# ---------------------------------------------------------------
# QUEUE WORKER
# ---------------------------------------------------------------
def queue_worker():
    while True:
        item = analysis_queue.get()
        try:
            trigger_ts = item["trigger_ts"]   # float epoch time of webhook
            ts_str = item["ts_str"]
            model = item["model"]
            prompt = item["prompt"]
            source = item.get("source", "webhook")

            root.after(0, lambda t=ts_str: status_var.set(f"📡 Processing event [{t}]"))

            if source == "manual":
                # Manual browse — skip motion detection, send image directly
                image_bytes = base64.b64decode(item["image_b64"])
                t0 = time.time()
                result = analyze_image_bytes(image_bytes, prompt, model)
                elapsed = time.time() - t0
                _b = item["image_b64"]
                root.after(0, lambda b=_b, r=result, t=ts_str, e=elapsed:
                           finish_analysis(b, r, t, e))
                save_event_background(None, None, image_bytes, result, None, ts_str)
                continue

            # --- Get frame A: 1s BEFORE webhook ---
            ts_a = trigger_ts - SNAP_BACK_SECS
            frame_a = get_frame_at(ts_a)

            # --- Get frame B: 0.5s AFTER webhook ---
            # Wait briefly to let buffer catch up
            time.sleep(SNAP_FORWARD_SECS + 0.2)
            ts_b = trigger_ts + SNAP_FORWARD_SECS
            frame_b = get_frame_at(ts_b)

            if frame_a is None or frame_b is None:
                root.after(0, lambda t=ts_str: status_var.set(
                    f"⚠️ [{t}] Buffer miss — not enough frames yet"))
                continue

            root.after(0, lambda t=ts_str: status_var.set(f"🔬 Computing motion diff [{t}]"))

            # --- Motion diff + crop ---
            cropped_bytes, debug_imgs, bbox = compute_motion_crop(frame_a, frame_b)

            # Show debug window
            if debug_imgs:
                root.after(0, lambda d=debug_imgs: show_debug_window(d))

            if cropped_bytes is None:
                # No motion detected — use full frame B
                root.after(0, lambda t=ts_str: status_var.set(
                    f"⚠️ [{t}] No motion detected — using full frame"))
                cropped_bytes = frame_b

            # Show crop in detected panel
            crop_b64 = base64.b64encode(cropped_bytes).decode()
            root.after(0, lambda b=crop_b64: show_detected_image(b))

            # --- Analyze ---
            qsize = analysis_queue.qsize()
            root.after(0, lambda q=qsize, t=ts_str: status_var.set(
                f"🔍 Analyzing [{t}]" + (f" — {q} more queued" if q > 0 else "")
            ))

            t0 = time.time()
            result = analyze_image_bytes(cropped_bytes, prompt, model)
            elapsed = time.time() - t0

            _b64 = crop_b64
            root.after(0, lambda b=_b64, r=result, t=ts_str, e=elapsed:
                       finish_analysis(b, r, t, e))

            save_event_background(frame_a, frame_b, cropped_bytes, result, bbox, ts_str)

        except Exception as e:
            import traceback
            traceback.print_exc()
            root.after(0, lambda err=str(e): status_var.set(f"⚠️ Error: {err}"))
        finally:
            analysis_queue.task_done()
            root.after(0, update_queue_status)

def finish_analysis(image_b64, result, ts, elapsed):
    timer_var.set(f"⏱  {elapsed:.2f}s")
    show_detected_image(image_b64)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result)
    output_text.config(state=tk.DISABLED)
    update_queue_status()

def enqueue_event(ts_float, ts_str, source="webhook", image_b64=None):
    model = model_var.get()
    prompt = prompt_text.get("1.0", tk.END).strip()
    item = {
        "trigger_ts": ts_float,
        "ts_str": ts_str,
        "model": model,
        "prompt": prompt,
        "source": source,
    }
    if image_b64:
        item["image_b64"] = image_b64
    analysis_queue.put(item)
    qsize = analysis_queue.qsize()
    root.after(0, lambda q=qsize: status_var.set(f"📋 Event queued — {q} in queue"))

def show_detected_image(image_b64):
    try:
        img_data = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_data))
        img.thumbnail((320, 180), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        detected_label.config(image=photo, text="")
        detected_label.image = photo
    except Exception:
        pass

def update_stream_preview(image_b64):
    try:
        img_data = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_data))
        img.thumbnail((320, 180), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        stream_label.config(image=photo, text="")
        stream_label.image = photo
        stream_status_var.set(f"🟢 Live  {datetime.now().strftime('%H:%M:%S')}")
    except Exception:
        stream_status_var.set("⚠️ Preview error")

# ---------------------------------------------------------------
# FLASK WEBHOOK
# ---------------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/event", methods=["POST"])
def event():
    try:
        data = request.get_json(force=True)
        alarm = data.get("alarm", {})
        triggers = alarm.get("triggers", [])
        trigger_key = triggers[0].get("key", "unknown") if triggers else "unknown"
        if trigger_key != "line_crossed":
            return "SKIP", 200
        ts_float = time.time()
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enqueue_event(ts_float, ts_str, source="webhook")
        return "OK", 200
    except Exception as e:
        print(f"[Webhook error] {e}")
        return "ERROR", 500

def run_flask():
    flask_app.run(host="0.0.0.0", port=WEBHOOK_PORT, debug=False, use_reloader=False)

# ---------------------------------------------------------------
# MANUAL BROWSE
# ---------------------------------------------------------------
def browse_and_analyze():
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
    )
    if not path:
        return
    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enqueue_event(time.time(), ts_str, source="manual", image_b64=image_b64)

def snap_from_stream():
    """Simulate a webhook trigger for testing."""
    ts_float = time.time()
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enqueue_event(ts_float, ts_str, source="webhook")

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
root = tk.Tk()
root.title("Security Vision — Preston" + (" [DEBUG]" if DEBUG_MODE else ""))
root.geometry("860x760")
root.configure(bg="#0a0a0a")
root.resizable(True, True)

# Status bar
status_var = tk.StringVar(value="Starting...")
tk.Label(root, textvariable=status_var, bg="#0a0a0a", fg="#444444",
         font=("Courier New", 9), anchor="w", padx=12, pady=4).pack(fill=tk.X, side=tk.BOTTOM)

# Debug indicator
if DEBUG_MODE:
    tk.Label(root, text="● DEBUG MODE ON", bg="#0a0a0a", fg="#ff6600",
             font=("Courier New", 9, "bold"), anchor="e", padx=12).pack(fill=tk.X, side=tk.BOTTOM)

# Top: model
top_bar = tk.Frame(root, bg="#0a0a0a")
top_bar.pack(fill=tk.X, padx=14, pady=(14, 4))

tk.Label(top_bar, text="MODEL", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))

model_var = tk.StringVar(value=DEFAULT_MODEL)
style = ttk.Style()
style.theme_use("clam")
style.configure("Dark.TCombobox",
    fieldbackground="#111111", background="#111111",
    foreground="#00ff88", arrowcolor="#00ff88",
    selectbackground="#003322", selectforeground="#00ff88",
    bordercolor="#222222", lightcolor="#111111", darkcolor="#111111"
)
model_dropdown = ttk.Combobox(top_bar, textvariable=model_var,
                               font=("Courier New", 10), style="Dark.TCombobox",
                               state="readonly", width=28)
model_dropdown.pack(side=tk.LEFT)

tk.Label(top_bar, text=f"💾 {SAVE_DIR}", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8), padx=12).pack(side=tk.LEFT)

# Two-panel: live stream | last detected
panels = tk.Frame(root, bg="#0a0a0a")
panels.pack(fill=tk.X, padx=14, pady=(8, 4))

# Stream panel
stream_panel = tk.Frame(panels, bg="#0a0a0a")
stream_panel.pack(side=tk.LEFT, padx=(0, 8))

stream_header = tk.Frame(stream_panel, bg="#0a0a0a")
stream_header.pack(fill=tk.X)
tk.Label(stream_header, text="LIVE STREAM", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8, "bold")).pack(side=tk.LEFT)
stream_status_var = tk.StringVar(value="⏳ Connecting...")
tk.Label(stream_header, textvariable=stream_status_var, bg="#0a0a0a", fg="#2a6a3a",
         font=("Courier New", 8)).pack(side=tk.LEFT, padx=8)

stream_label = tk.Label(stream_panel, bg="#0d1a0d", text="Connecting to stream...",
                         fg="#2a5a2a", font=("Courier New", 9),
                         width=40, height=10, anchor="center", relief=tk.FLAT)
stream_label.pack()

# Detected panel
detected_panel = tk.Frame(panels, bg="#0a0a0a")
detected_panel.pack(side=tk.LEFT)

tk.Label(detected_panel, text="LAST DETECTED (AI CROP)", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8, "bold")).pack(anchor="w")

detected_label = tk.Label(detected_panel, bg="#111111", text="Waiting for event...",
                            fg="#333333", font=("Courier New", 9),
                            width=40, height=10, anchor="center", relief=tk.FLAT)
detected_label.pack()

# Output
tk.Label(root, text="OUTPUT", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8, "bold"), anchor="w", padx=14).pack(fill=tk.X, pady=(6,0))

out_frame = tk.Frame(root, bg="#0a0a0a")
out_frame.pack(fill=tk.BOTH, expand=True, padx=14)

output_text = tk.Text(out_frame, bg="#111111", fg="#e0e0e0",
    font=("Courier New", 11), relief=tk.FLAT,
    padx=10, pady=8, wrap=tk.WORD, height=6,
    state=tk.DISABLED, insertbackground="#00ff88",
    selectbackground="#003322")
out_scroll = tk.Scrollbar(out_frame, command=output_text.yview, bg="#111111")
output_text.configure(yscrollcommand=out_scroll.set)
out_scroll.pack(side=tk.RIGHT, fill=tk.Y)
output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Action row
action_row = tk.Frame(root, bg="#0a0a0a")
action_row.pack(fill=tk.X, padx=14, pady=(6, 4))

browse_btn = tk.Button(
    action_row, text="📁  BROWSE", command=browse_and_analyze,
    bg="#111111", fg="#00ff88", font=("Courier New", 10, "bold"),
    relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
    activebackground="#003322", activeforeground="#00ff88", bd=0
)
browse_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

snap_btn = tk.Button(
    action_row, text="📡  TRIGGER TEST", command=snap_from_stream,
    bg="#111111", fg="#00aaff", font=("Courier New", 10, "bold"),
    relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
    activebackground="#001a33", activeforeground="#00aaff", bd=0
)
snap_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

timer_var = tk.StringVar(value="⏱  —")
tk.Label(action_row, textvariable=timer_var, bg="#0a0a0a", fg="#00aaff",
         font=("Courier New", 12, "bold"), padx=14).pack(side=tk.RIGHT)

# Prompt
tk.Label(root, text="PROMPT", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8, "bold"), anchor="w", padx=14).pack(fill=tk.X)

prompt_frame = tk.Frame(root, bg="#0a0a0a")
prompt_frame.pack(fill=tk.X, padx=14, pady=(2, 12))

prompt_text = tk.Text(prompt_frame, bg="#111111", fg="#999999",
    font=("Courier New", 9), relief=tk.FLAT, padx=10, pady=6,
    wrap=tk.WORD, height=4, insertbackground="#00ff88",
    selectbackground="#003322")
ps = tk.Scrollbar(prompt_frame, command=prompt_text.yview, bg="#111111")
prompt_text.configure(yscrollcommand=ps.set)
ps.pack(side=tk.RIGHT, fill=tk.Y)
prompt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
prompt_text.insert(tk.END, DEFAULT_PROMPT)

# ---------------------------------------------------------------
# START SERVICES
# ---------------------------------------------------------------
threading.Thread(target=run_flask, daemon=True).start()
threading.Thread(target=queue_worker, daemon=True).start()
threading.Thread(target=buffer_worker, daemon=True).start()
threading.Thread(target=warmup_model, daemon=True).start()

root.mainloop()
