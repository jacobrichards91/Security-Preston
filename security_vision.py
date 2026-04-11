import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
import requests
import base64
import time
import os
import collections
import io
import json
import websocket
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, request
import logging

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_VISION_MODEL = "minicpm-v:latest"
DEFAULT_TEXT_MODEL   = "minicpm-v:latest"

# On Windows, zoneinfo needs the tzdata package: pip install tzdata
try:
    CHICAGO_TZ = ZoneInfo("America/Chicago")
except Exception:
    print("[Warning] tzdata not installed — run: pip install tzdata")
    print("[Warning] Falling back to UTC-5 (CDT). Install tzdata for correct DST handling.")
    CHICAGO_TZ = timezone(timedelta(hours=-5))
WEBHOOK_PORT = 8765
SAVE_DIR = Path(os.path.expanduser("~")) / "SecurityEvents"
SAVE_DIR.mkdir(exist_ok=True)
CONFIG_PATH = Path.home() / "Documents" / "GitHub" / "security_preston_config.json"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

RTSP_URL = "rtsp://192.168.0.166:7447/YD4arutidcyKjQvI"
STREAM_PREVIEW_INTERVAL = 3000  # ms between live preview refreshes

# Buffer config — defaults (live values come from UI vars after root is created)
BUFFER_SECONDS = 3.0          # how many seconds of frames to keep
FRAME_INTERVAL = 0.2          # grab a frame every N seconds for buffer (not tunable)
SNAP_BEFORE_SECS = 2.5        # frame A: this many seconds before the trigger
SNAP_AFTER_SECS  = 2.0        # frame B: this many seconds before the trigger (closer)
MIN_BOX_PCT = 0.05            # minimum contour size as % of frame area

system_active = threading.Event()
system_active.set()           # ON by default
CROP_PADDING = 50             # px padding around bounding box (default, overridden by UI var)

DEBUG_MODE = True             # show debug windows

DEFAULT_VISION_PROMPT = """You are a security camera analyzer. Look for people, vehicles, and animals ONLY.

For each one found, describe: count, type, appearance, and behavior.

If none are present, respond with only: CLEAR"""

DEFAULT_TEXT_PROMPT = """You are a home security analyst with full situational awareness of the house.

You will receive:
1. The current date and time (Chicago)
2. The live state of all sensors, doors, locks, and occupancy in the home
3. A visual description from a security camera that just detected motion

Based on ALL of this context, provide a concise security assessment:
- What was detected and is it expected given the time and home state?
- Is this benign (resident, pet, expected visitor) or suspicious?
- Any recommended action?

Be brief and direct. If clearly benign, say so."""

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

analysis_queue = queue.Queue()

# --- Rolling frame buffer: deque of (timestamp, image_bytes) ---
frame_buffer = collections.deque()
buffer_lock  = threading.Lock()

# --- Area exclusion mask: list of (x1, y1, x2, y2) in native frame pixels ---
mask_rects = []

# ---------------------------------------------------------------
# HOME ASSISTANT CONFIG
# ---------------------------------------------------------------
HA_HOST  = "192.168.0.209"
HA_TOKEN = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJpc3MiOiJmMjNkMzE4Nzc1OWY0ZTQ4YTZmOWZhNTYyNjc2ZTE1ZCIsImlhdCI6MTc3NT"
            "kyMTcxMiwiZXhwIjoyMDkxMjgxNzEyfQ"
            ".rfCWpPO1-eaeHHlblGrhO4XzzDZtPT1BIMeV5piaEaQ")

WATCHED_ENTITIES = {
    # Occupancy
    "binary_sensor.house_occupied_3",
    "input_boolean.jacob_is_home",
    "input_boolean.lauren_is_home",
    # Doors
    "binary_sensor.front_door_door",
    "binary_sensor.aqara_door_and_window_sensor_p2_door",
    "binary_sensor.garage_door_door",
    # Locks
    "lock.aqara_smart_lock_u100",
    "lock.aqara_smart_lock_u100_2",
    # Garage
    "cover.smart_garage_door_opener_msg100_main_channel",
    # Person detected
    "binary_sensor.side_henrys_room_person_detected",
    "binary_sensor.front_person_detected",
    "binary_sensor.front_door_person_detected",
    "binary_sensor.side_yard_cul_de_sac_person_detected",
    "binary_sensor.side_yard_street_person_detected",
    "binary_sensor.patio2_person_detected",
    "binary_sensor.backyard_person_detected",
    # Animal detected
    "binary_sensor.front_animal_detected",
    "binary_sensor.side_henrys_room_animal_detected",
    "binary_sensor.front_door_animal_detected",
    "binary_sensor.side_yard_cul_de_sac_animal_detected",
    "binary_sensor.side_yard_street_animal_detected",
    "binary_sensor.patio2_animal_detected",
    "binary_sensor.backyard_animal_detected",
}

# Short display names for each entity
HA_NAMES = {
    "binary_sensor.house_occupied_3":                         "house occupied",
    "input_boolean.jacob_is_home":                            "jacob home",
    "input_boolean.lauren_is_home":                           "lauren home",
    "binary_sensor.front_door_door":                          "front door",
    "binary_sensor.aqara_door_and_window_sensor_p2_door":     "back door",
    "binary_sensor.garage_door_door":                         "garage door",
    "lock.aqara_smart_lock_u100":                             "lock u100",
    "lock.aqara_smart_lock_u100_2":                           "lock u100 2",
    "cover.smart_garage_door_opener_msg100_main_channel":     "garage cover",
    "binary_sensor.side_henrys_room_person_detected":         "henry's room",
    "binary_sensor.front_person_detected":                    "front",
    "binary_sensor.front_door_person_detected":               "front door",
    "binary_sensor.side_yard_cul_de_sac_person_detected":     "cul-de-sac",
    "binary_sensor.side_yard_street_person_detected":         "street",
    "binary_sensor.patio2_person_detected":                   "patio",
    "binary_sensor.backyard_person_detected":                 "backyard",
    "binary_sensor.front_animal_detected":                    "front",
    "binary_sensor.side_henrys_room_animal_detected":         "henry's room",
    "binary_sensor.front_door_animal_detected":               "front door",
    "binary_sensor.side_yard_cul_de_sac_animal_detected":     "cul-de-sac",
    "binary_sensor.side_yard_street_animal_detected":         "street",
    "binary_sensor.patio2_animal_detected":                   "patio",
    "binary_sensor.backyard_animal_detected":                 "backyard",
}

# Groups for display order
HA_GROUPS = [
    ("OCCUPANCY",        ["binary_sensor.house_occupied_3",
                          "input_boolean.jacob_is_home",
                          "input_boolean.lauren_is_home"]),
    ("DOORS",            ["binary_sensor.front_door_door",
                          "binary_sensor.aqara_door_and_window_sensor_p2_door",
                          "binary_sensor.garage_door_door"]),
    ("LOCKS",            ["lock.aqara_smart_lock_u100",
                          "lock.aqara_smart_lock_u100_2"]),
    ("GARAGE",           ["cover.smart_garage_door_opener_msg100_main_channel"]),
    ("PERSON DETECTED",  ["binary_sensor.front_person_detected",
                          "binary_sensor.front_door_person_detected",
                          "binary_sensor.side_henrys_room_person_detected",
                          "binary_sensor.side_yard_cul_de_sac_person_detected",
                          "binary_sensor.side_yard_street_person_detected",
                          "binary_sensor.patio2_person_detected",
                          "binary_sensor.backyard_person_detected"]),
    ("ANIMAL DETECTED",  ["binary_sensor.front_animal_detected",
                          "binary_sensor.front_door_animal_detected",
                          "binary_sensor.side_henrys_room_animal_detected",
                          "binary_sensor.side_yard_cul_de_sac_animal_detected",
                          "binary_sensor.side_yard_street_animal_detected",
                          "binary_sensor.patio2_animal_detected",
                          "binary_sensor.backyard_animal_detected"]),
]

ha_state      = {}          # entity_id -> state string
ha_row_labels = {}          # entity_id -> {"dot": Label, "val": Label}

# ---------------------------------------------------------------
# FRAME BUFFER WORKER
# Continuously grabs frames via FFmpeg and stores in rolling buffer
# ---------------------------------------------------------------
def buffer_worker():
    """Open RTSP stream once with OpenCV and maintain a rolling frame buffer."""
    cap = None
    _fail_count = 0
    _last_frame_ts = 0.0

    while True:
        # (Re)open the capture if needed
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            root.after(0, lambda: stream_status_var.set("⏳ Connecting to stream..."))
            if DEBUG_MODE:
                print(f"[RTSP] Opening stream (attempt {_fail_count + 1}): {RTSP_URL}")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
            if not cap.isOpened():
                _fail_count += 1
                if DEBUG_MODE:
                    print(f"[RTSP] Could not open stream (attempt {_fail_count})")
                root.after(0, lambda n=_fail_count: stream_status_var.set(
                    f"🔴 Disconnected — retrying... (attempt {n})"
                ))
                time.sleep(2)
                continue
            _fail_count = 0

        ret, frame = cap.read()
        if not ret:
            _fail_count += 1
            if DEBUG_MODE:
                print(f"[RTSP] Read failed — reconnecting (attempt {_fail_count})")
            root.after(0, lambda n=_fail_count: stream_status_var.set(
                f"🔴 Stream lost — reconnecting... (attempt {n})"
            ))
            cap.release()
            cap = None
            time.sleep(1)
            continue

        _fail_count = 0
        now = time.time()

        # Throttle: only store a frame every FRAME_INTERVAL seconds
        if now - _last_frame_ts < FRAME_INTERVAL:
            continue
        _last_frame_ts = now

        # Encode to JPEG bytes and store in buffer
        _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = enc.tobytes()

        with buffer_lock:
            frame_buffer.append((now, data))
            cutoff = now - buffer_secs_var.get()
            while frame_buffer and frame_buffer[0][0] < cutoff:
                frame_buffer.popleft()

        b64 = base64.b64encode(data).decode()
        root.after(0, lambda b=b64: update_stream_preview(b))

def get_frame_at(target_ts):
    """Get the buffered frame closest to target_ts. Returns bytes or None."""
    with buffer_lock:
        if not frame_buffer:
            return None
        best = min(frame_buffer, key=lambda x: abs(x[0] - target_ts))
        return best[1]

def apply_mask(img_bgr):
    """Paint black over every rect in mask_rects. Operates in-place on a copy."""
    if not mask_rects:
        return img_bgr
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in mask_rects:
        out[y1:y2, x1:x2] = 0
    return out

def jpeg_apply_mask(jpeg_bytes):
    """Decode JPEG bytes → apply mask → re-encode. Returns bytes."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    img = apply_mask(img)
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return enc.tobytes()

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
    min_area = w * h * (min_box_pct_var.get() / 100.0)
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
        pad = int(crop_padding_var.get())
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)
        y2p = min(h, y2 + pad)
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
            vision_model_dropdown["values"] = models
            text_model_dropdown["values"]   = models
            # Default vision model to minicpm-v
            if "minicpm-v:latest" in models:
                vision_model_var.set("minicpm-v:latest")
            elif any("minicpm-v" in m for m in models):
                vision_model_var.set(next(m for m in models if "minicpm-v" in m))
            else:
                vision_model_var.set(models[0])
            # Default text model — keep whatever is loaded, fall back to first
            if text_model_var.get() not in models:
                text_model_var.set(models[0])
    except Exception as e:
        status_var.set(f"⚠️ Could not fetch models: {e}")

def warmup_model():
    status_var.set("⏳ Loading models...")
    fetch_models()
    try:
        for m in {vision_model_var.get(), text_model_var.get()}:
            requests.post(OLLAMA_URL, json={
                "model": m, "prompt": "ready", "stream": False, "keep_alive": -1
            }, timeout=60)
        update_queue_status()
    except Exception as e:
        status_var.set(f"⚠️ Warmup failed: {e}")

def analyze_image_bytes(image_bytes, prompt, model):
    """Run a vision model with an image attached."""
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

def analyze_text(prompt, model):
    """Run a text-only model (no image)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "No response received.")

def build_ha_context():
    """Format current HA state into a readable string for the text model."""
    lines = []
    for group_name, entities in HA_GROUPS:
        parts = [f"{HA_NAMES.get(e, e)}={ha_state.get(e, 'unknown')}" for e in entities]
        lines.append(f"{group_name}: {', '.join(parts)}")
    return "\n".join(lines)

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
            trigger_ts    = item["trigger_ts"]
            ts_str        = item["ts_str"]
            vision_model  = item["vision_model"]
            text_model    = item["text_model"]
            vision_prompt = item["vision_prompt"]
            text_prompt   = item["text_prompt"]
            source        = item.get("source", "webhook")

            root.after(0, lambda t=ts_str: status_var.set(f"📡 Processing event [{t}]"))

            if source == "manual":
                image_bytes = base64.b64decode(item["image_b64"])
                root.after(0, lambda t=ts_str: status_var.set(f"👁 Vision [{t}]"))
                t0 = time.time()
                vision_result = analyze_image_bytes(image_bytes, vision_prompt, vision_model)

                root.after(0, lambda t=ts_str: status_var.set(f"🧠 Judgment [{t}]"))
                chicago_now = datetime.now(CHICAGO_TZ).strftime("%A %B %d %Y  %I:%M:%S %p %Z")
                full_text_prompt = (
                    f"{text_prompt}\n\n"
                    f"Time: {chicago_now}\n\n"
                    f"Home state:\n{build_ha_context()}\n\n"
                    f"Visual observation:\n{vision_result}"
                )
                text_result = analyze_text(full_text_prompt, text_model)
                elapsed = time.time() - t0

                _b = item["image_b64"]
                root.after(0, lambda b=_b, vr=vision_result, tr=text_result, t=ts_str, e=elapsed:
                           finish_analysis(b, vr, tr, t, e))
                save_event_background(None, None, image_bytes, text_result, None, ts_str)
                continue

            # --- Grab frames (both in the past) ---
            ts_a = trigger_ts - snap_before_var.get()
            ts_b = trigger_ts - snap_after_var.get()
            frame_a = get_frame_at(ts_a)
            frame_b = get_frame_at(ts_b)

            if frame_a is None or frame_b is None:
                root.after(0, lambda t=ts_str: status_var.set(
                    f"⚠️ [{t}] Buffer miss — not enough frames"))
                continue

            if mask_rects:
                frame_a = jpeg_apply_mask(frame_a)
                frame_b = jpeg_apply_mask(frame_b)

            root.after(0, lambda t=ts_str: status_var.set(f"🔬 Motion diff [{t}]"))
            cropped_bytes, debug_imgs, bbox = compute_motion_crop(frame_a, frame_b)

            if debug_imgs:
                root.after(0, lambda d=debug_imgs: show_debug_window(d))
            if cropped_bytes is None:
                root.after(0, lambda t=ts_str: status_var.set(
                    f"⚠️ [{t}] No motion — using full frame"))
                cropped_bytes = frame_b

            crop_b64 = base64.b64encode(cropped_bytes).decode()
            root.after(0, lambda b=crop_b64: show_detected_image(b))

            # --- Stage 1: Vision model ---
            qsize = analysis_queue.qsize()
            root.after(0, lambda q=qsize, t=ts_str: status_var.set(
                f"👁 Vision [{t}]" + (f" — {q} queued" if q else "")))
            t0 = time.time()
            vision_result = analyze_image_bytes(cropped_bytes, vision_prompt, vision_model)

            # --- Stage 2: Text model with full context ---
            root.after(0, lambda t=ts_str: status_var.set(f"🧠 Judgment [{t}]"))
            chicago_now = datetime.now(CHICAGO_TZ).strftime("%A %B %d %Y  %I:%M:%S %p %Z")
            full_text_prompt = (
                f"{text_prompt}\n\n"
                f"Time: {chicago_now}\n\n"
                f"Home state:\n{build_ha_context()}\n\n"
                f"Visual observation:\n{vision_result}"
            )
            text_result = analyze_text(full_text_prompt, text_model)
            elapsed = time.time() - t0

            _b64 = crop_b64
            root.after(0, lambda b=_b64, vr=vision_result, tr=text_result, t=ts_str, e=elapsed:
                       finish_analysis(b, vr, tr, t, e))

            save_event_background(frame_a, frame_b, cropped_bytes,
                                  f"VISION:\n{vision_result}\n\nJUDGMENT:\n{text_result}",
                                  bbox, ts_str)

        except Exception as e:
            import traceback
            traceback.print_exc()
            root.after(0, lambda err=str(e): status_var.set(f"⚠️ Error: {err}"))
        finally:
            analysis_queue.task_done()
            root.after(0, update_queue_status)

def finish_analysis(image_b64, vision_result, text_result, ts, elapsed):
    timer_var.set(f"⏱  {elapsed:.2f}s")
    show_detected_image(image_b64)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.tag_configure("dim",   foreground="#555555", font=("Courier New", 9))
    output_text.tag_configure("label", foreground="#444444", font=("Courier New", 8, "bold"))
    output_text.tag_configure("main",  foreground="#e0e0e0", font=("Courier New", 11))
    output_text.insert(tk.END, "👁  VISION\n", "label")
    output_text.insert(tk.END, vision_result + "\n\n", "dim")
    output_text.insert(tk.END, "🧠  JUDGMENT\n", "label")
    output_text.insert(tk.END, text_result, "main")
    output_text.config(state=tk.DISABLED)
    update_queue_status()

def enqueue_event(ts_float, ts_str, source="webhook", image_b64=None):
    item = {
        "trigger_ts":    ts_float,
        "ts_str":        ts_str,
        "vision_model":  vision_model_var.get(),
        "text_model":    text_model_var.get(),
        "vision_prompt": vision_prompt_text.get("1.0", tk.END).strip(),
        "text_prompt":   text_prompt_text.get("1.0", tk.END).strip(),
        "source":        source,
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
        img.thumbnail((640, 360), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        stream_label.config(image=photo, text="")
        stream_label.image = photo
        stream_status_var.set(f"🟢 Live  {datetime.now().strftime('%H:%M:%S')}")
    except Exception:
        stream_status_var.set("⚠️ Preview error")

# ---------------------------------------------------------------
# HOME ASSISTANT WEBSOCKET WORKER
# ---------------------------------------------------------------
def _ha_is_active(state):
    return state in ("on", "open", "unlocked", "detected", "playing", "home", "true")

def _ha_apply_entity(entity_id, state):
    """Update dict + UI labels for one entity (call via root.after on main thread)."""
    ha_state[entity_id] = state
    if entity_id not in ha_row_labels:
        return
    row = ha_row_labels[entity_id]
    active = _ha_is_active(state)
    row["dot"].config(text="●" if active else "○",
                      fg="#00ff88" if active else "#444444")
    row["val"].config(text=state,
                      fg="#00ff88" if active else "#666666")

def ha_worker():
    while True:
        try:
            ws = websocket.create_connection(
                f"ws://{HA_HOST}:8123/api/websocket",
                timeout=15
            )
            root.after(0, lambda: ha_status_var.set("⟳  Authenticating..."))

            # 1. auth_required → send token
            msg = json.loads(ws.recv())
            assert msg.get("type") == "auth_required", f"Expected auth_required, got {msg}"
            ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))

            # 2. auth_ok
            msg = json.loads(ws.recv())
            assert msg.get("type") == "auth_ok", f"Auth failed: {msg}"
            root.after(0, lambda: ha_status_var.set("⟳  Fetching states..."))

            # 3. get_states
            ws.send(json.dumps({"id": 1, "type": "get_states"}))

            # 4. subscribe to state_changed
            ws.send(json.dumps({"id": 2, "type": "subscribe_events",
                                "event_type": "state_changed"}))

            # 5. process messages indefinitely
            while True:
                raw = ws.recv()
                msg = json.loads(raw)

                if msg.get("id") == 1 and msg.get("type") == "result":
                    # Initial state dump
                    for s in msg.get("result", []):
                        eid = s.get("entity_id", "")
                        if eid in WATCHED_ENTITIES:
                            state = s.get("state", "unknown")
                            root.after(0, lambda e=eid, st=state: _ha_apply_entity(e, st))
                    root.after(0, lambda: ha_status_var.set("● Connected"))
                    print("[HA] Initial states loaded")

                elif msg.get("type") == "event":
                    edata = msg.get("event", {}).get("data", {})
                    eid   = edata.get("entity_id", "")
                    if eid in WATCHED_ENTITIES:
                        state = (edata.get("new_state") or {}).get("state", "unknown")
                        root.after(0, lambda e=eid, st=state: _ha_apply_entity(e, st))
                        print(f"[HA] {eid} → {state}")

        except Exception as e:
            print(f"[HA] Disconnected: {e} — retrying in 10s")
            root.after(0, lambda: ha_status_var.set("○ Disconnected — retrying..."))
            try:
                ws.close()
            except Exception:
                pass
            time.sleep(10)

# ---------------------------------------------------------------
# FLASK WEBHOOK
# ---------------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/event", methods=["POST"])
def event():
    if not system_active.is_set():
        print("[Webhook] System is OFF — ignoring event")
        return "INACTIVE", 200
    try:
        raw = request.get_data(as_text=True)
        data = request.get_json(force=True) or {}

        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Webhook] {ts_str} — incoming POST to /event")
        print(f"[Webhook] payload: {raw[:500]}")  # first 500 chars to avoid flood

        alarm = data.get("alarm", {})
        triggers = alarm.get("triggers", [])
        trigger_key = triggers[0].get("key", "unknown") if triggers else "unknown"
        print(f"[Webhook] trigger_key={trigger_key!r}")

        if trigger_key != "line_crossed":
            print(f"[Webhook] SKIPPED — expected 'line_crossed', got {trigger_key!r}")
            root.after(0, lambda k=trigger_key: status_var.set(
                f"⚠️ Webhook received but skipped (trigger={k!r}) — check console"
            ))
            return "SKIP", 200

        print(f"[Webhook] ACCEPTED — queuing analysis")
        ts_float = time.time()
        enqueue_event(ts_float, ts_str, source="webhook")
        return "OK", 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Webhook error] {e}")
        return "ERROR", 500

def run_flask():
    flask_app.run(host="0.0.0.0", port=WEBHOOK_PORT, debug=False, use_reloader=False)

# ---------------------------------------------------------------
# AREA RESTRICTOR WIZARD
# ---------------------------------------------------------------
def open_mask_wizard():
    global mask_rects

    # Grab the most recent frame from the buffer
    with buffer_lock:
        snap = frame_buffer[-1][1] if frame_buffer else None
    if snap is None:
        status_var.set("⚠️ No frame in buffer yet — wait for stream to connect")
        return

    arr = np.frombuffer(snap, dtype=np.uint8)
    native = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if native is None:
        return
    native_h, native_w = native.shape[:2]

    # Display size
    disp_w, disp_h = 960, 540
    scale_x = native_w / disp_w
    scale_y = native_h / disp_h

    win = tk.Toplevel(root)
    win.title("Area Restrictor — draw boxes to exclude, right-click to delete")
    win.configure(bg="#0a0a0a")
    win.resizable(False, False)

    # Convert frame to PIL for display
    rgb = cv2.cvtColor(native, cv2.COLOR_BGR2RGB)
    pil_bg = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)

    # We'll redraw the canvas whenever rects change
    canvas = tk.Canvas(win, width=disp_w, height=disp_h,
                       bg="#111111", cursor="crosshair",
                       highlightthickness=0)
    canvas.pack(padx=10, pady=(10, 4))

    # Header info
    info_var = tk.StringVar(value=f"{len(mask_rects)} zone(s) active — drag to add, right-click to delete")
    tk.Label(win, textvariable=info_var, bg="#0a0a0a", fg="#555555",
             font=("Courier New", 8)).pack()

    # Button row
    btn_row = tk.Frame(win, bg="#0a0a0a")
    btn_row.pack(fill=tk.X, padx=10, pady=(4, 10))

    def redraw():
        canvas.delete("all")
        # Draw background frame
        tk_img = ImageTk.PhotoImage(pil_bg)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas._bg_ref = tk_img  # keep reference

        # Draw saved rects
        for i, (x1, y1, x2, y2) in enumerate(mask_rects):
            dx1 = int(x1 / scale_x)
            dy1 = int(y1 / scale_y)
            dx2 = int(x2 / scale_x)
            dy2 = int(y2 / scale_y)
            canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                    fill="black", outline="#ff4444", width=2,
                                    tags=f"rect{i}")
            # Label in corner
            canvas.create_text(dx1 + 4, dy1 + 4, anchor="nw",
                                text=str(i + 1), fill="#ff4444",
                                font=("Courier New", 9, "bold"))

        info_var.set(f"{len(mask_rects)} zone(s) active — drag to add, right-click to delete")

    # Drawing state
    _draw = {"start": None, "live_rect": None}

    def on_press(e):
        _draw["start"] = (e.x, e.y)
        if _draw["live_rect"]:
            canvas.delete(_draw["live_rect"])
            _draw["live_rect"] = None

    def on_drag(e):
        if _draw["start"] is None:
            return
        x0, y0 = _draw["start"]
        if _draw["live_rect"]:
            canvas.delete(_draw["live_rect"])
        _draw["live_rect"] = canvas.create_rectangle(
            x0, y0, e.x, e.y,
            fill="black", outline="#ffaa00", width=2, stipple="gray50"
        )

    def on_release(e):
        if _draw["start"] is None:
            return
        x0, y0 = _draw["start"]
        x1_d, y1_d = min(x0, e.x), min(y0, e.y)
        x2_d, y2_d = max(x0, e.x), max(y0, e.y)
        _draw["start"] = None
        if _draw["live_rect"]:
            canvas.delete(_draw["live_rect"])
            _draw["live_rect"] = None
        if abs(x2_d - x1_d) < 5 or abs(y2_d - y1_d) < 5:
            return  # too small, ignore
        # Scale back to native resolution
        nx1 = max(0, int(x1_d * scale_x))
        ny1 = max(0, int(y1_d * scale_y))
        nx2 = min(native_w, int(x2_d * scale_x))
        ny2 = min(native_h, int(y2_d * scale_y))
        mask_rects.append((nx1, ny1, nx2, ny2))
        redraw()
        schedule_save()

    def on_right_click(e):
        # Find and delete the rect clicked on
        for i, (x1, y1, x2, y2) in enumerate(mask_rects):
            dx1 = int(x1 / scale_x)
            dy1 = int(y1 / scale_y)
            dx2 = int(x2 / scale_x)
            dy2 = int(y2 / scale_y)
            if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                mask_rects.pop(i)
                redraw()
                schedule_save()
                return

    def clear_all():
        mask_rects.clear()
        redraw()
        schedule_save()

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>",        on_right_click)

    tk.Button(btn_row, text="🗑  CLEAR ALL", command=clear_all,
              bg="#111111", fg="#ff4444", font=("Courier New", 9, "bold"),
              relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
              activebackground="#2a0000", activeforeground="#ff4444", bd=0
              ).pack(side=tk.LEFT)

    tk.Button(btn_row, text="✓  DONE", command=win.destroy,
              bg="#111111", fg="#00ff88", font=("Courier New", 9, "bold"),
              relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
              activebackground="#003322", activeforeground="#00ff88", bd=0
              ).pack(side=tk.RIGHT)

    redraw()
    win.grab_set()

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
# CONFIG PERSISTENCE
# ---------------------------------------------------------------
_save_job = None

def save_config(*_):
    """Write all tunable settings to config.json."""
    try:
        data = {
            "vision_model":    vision_model_var.get(),
            "text_model":      text_model_var.get(),
            "vision_prompt":   vision_prompt_text.get("1.0", tk.END).rstrip("\n"),
            "text_prompt":     text_prompt_text.get("1.0", tk.END).rstrip("\n"),
            "snap_before":     snap_before_var.get(),
            "snap_after":      snap_after_var.get(),
            "buffer_secs":     buffer_secs_var.get(),
            "crop_padding":    crop_padding_var.get(),
            "min_box_pct":     min_box_pct_var.get(),
            "cam_name":        cam_name_var.get(),
            "system_active":   system_active_var.get(),
            "mask_rects":      [list(r) for r in mask_rects],
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[Config] Save error: {e}")

def schedule_save(*_):
    """Debounce saves — write 400 ms after the last change."""
    global _save_job
    if _save_job:
        root.after_cancel(_save_job)
    _save_job = root.after(400, save_config)

def load_config():
    """Read config.json and apply all stored values to the UI."""
    if not CONFIG_PATH.exists():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text())
        if "vision_model" in data:
            vision_model_var.set(data["vision_model"])
        if "text_model" in data:
            text_model_var.set(data["text_model"])
        if "vision_prompt" in data:
            vision_prompt_text.delete("1.0", tk.END)
            vision_prompt_text.insert("1.0", data["vision_prompt"])
        if "text_prompt" in data:
            text_prompt_text.delete("1.0", tk.END)
            text_prompt_text.insert("1.0", data["text_prompt"])
        if "snap_before" in data:
            snap_before_var.set(data["snap_before"])
        if "snap_after" in data:
            snap_after_var.set(data["snap_after"])
        if "buffer_secs" in data:
            buffer_secs_var.set(data["buffer_secs"])
        if "crop_padding" in data:
            crop_padding_var.set(data["crop_padding"])
        if "min_box_pct" in data:
            min_box_pct_var.set(data["min_box_pct"])
        if "cam_name" in data:
            cam_name_var.set(data["cam_name"])
        if "system_active" in data:
            system_active_var.set(data["system_active"])
        if "mask_rects" in data:
            mask_rects.clear()
            mask_rects.extend(tuple(r) for r in data["mask_rects"])
        print(f"[Config] Loaded from {CONFIG_PATH}")
    except Exception as e:
        print(f"[Config] Load error: {e}")

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
root = tk.Tk()
root.title("Security Vision — Preston" + (" [DEBUG]" if DEBUG_MODE else ""))
root.geometry("1100x860")
root.configure(bg="#0a0a0a")
root.resizable(True, True)

# --- All tunable vars ---
snap_before_var   = tk.DoubleVar(value=SNAP_BEFORE_SECS)
snap_after_var    = tk.DoubleVar(value=SNAP_AFTER_SECS)
buffer_secs_var   = tk.DoubleVar(value=BUFFER_SECONDS)
crop_padding_var  = tk.IntVar(value=CROP_PADDING)
min_box_pct_var   = tk.DoubleVar(value=MIN_BOX_PCT)
vision_model_var  = tk.StringVar(value=DEFAULT_VISION_MODEL)
text_model_var    = tk.StringVar(value=DEFAULT_TEXT_MODEL)
cam_name_var      = tk.StringVar(value="Front Door")
system_active_var = tk.BooleanVar(value=True)

for _v in (snap_before_var, snap_after_var, buffer_secs_var,
           crop_padding_var, min_box_pct_var, vision_model_var, text_model_var):
    _v.trace_add("write", schedule_save)

# --- Status bar and debug (outside tabs, always visible) ---
status_var = tk.StringVar(value="Starting...")
tk.Label(root, textvariable=status_var, bg="#0a0a0a", fg="#444444",
         font=("Courier New", 9), anchor="w", padx=12, pady=4).pack(fill=tk.X, side=tk.BOTTOM)
if DEBUG_MODE:
    tk.Label(root, text="● DEBUG MODE ON", bg="#0a0a0a", fg="#ff6600",
             font=("Courier New", 9, "bold"), anchor="e", padx=12).pack(fill=tk.X, side=tk.BOTTOM)

# --- Notebook (tabs) ---
style = ttk.Style()
style.theme_use("clam")
style.configure("Dark.TNotebook",
    background="#0a0a0a", borderwidth=0, tabmargins=[0, 0, 0, 0])
style.configure("Dark.TNotebook.Tab",
    background="#111111", foreground="#555555",
    padding=[16, 6], font=("Courier New", 10, "bold"),
    borderwidth=0)
style.map("Dark.TNotebook.Tab",
    background=[("selected", "#0a0a0a")],
    foreground=[("selected", "#00ff88")])
style.configure("Dark.TCombobox",
    fieldbackground="#111111", background="#111111",
    foreground="#00ff88", arrowcolor="#00ff88",
    selectbackground="#003322", selectforeground="#00ff88",
    bordercolor="#222222", lightcolor="#111111", darkcolor="#111111")

notebook = ttk.Notebook(root, style="Dark.TNotebook")
notebook.pack(fill=tk.BOTH, expand=True)

tab_camera = tk.Frame(notebook, bg="#0a0a0a")
tab_master  = tk.Frame(notebook, bg="#0a0a0a")
notebook.add(tab_camera, text="Front Door")
notebook.add(tab_master,  text="Master")

# ── update tab label when cam_name_var changes ──
def _on_cam_name(*_):
    notebook.tab(tab_camera, text=cam_name_var.get() or "Camera")
    schedule_save()
cam_name_var.trace_add("write", _on_cam_name)

# ═══════════════════════════════════════════════
# TAB 1 — CAMERA (Front Door)
# ═══════════════════════════════════════════════

# Live stream + detected panels
panels = tk.Frame(tab_camera, bg="#0a0a0a")
panels.pack(fill=tk.X, padx=14, pady=(12, 4))

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
                         anchor="center", relief=tk.FLAT)
stream_label.pack()

detected_panel = tk.Frame(panels, bg="#0a0a0a")
detected_panel.pack(side=tk.LEFT)
tk.Label(detected_panel, text="LAST DETECTED (AI CROP)", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8, "bold")).pack(anchor="w")
detected_label = tk.Label(detected_panel, bg="#111111", text="Waiting for event...",
                            fg="#333333", font=("Courier New", 9),
                            anchor="center", relief=tk.FLAT)
detected_label.pack()

# Output
tk.Label(tab_camera, text="OUTPUT", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8, "bold"), anchor="w", padx=14).pack(fill=tk.X, pady=(6, 0))
out_frame = tk.Frame(tab_camera, bg="#0a0a0a")
out_frame.pack(fill=tk.BOTH, expand=True, padx=14)
output_text = tk.Text(out_frame, bg="#111111", fg="#e0e0e0",
    font=("Courier New", 11), relief=tk.FLAT,
    padx=10, pady=8, wrap=tk.WORD, height=6,
    state=tk.DISABLED, insertbackground="#00ff88", selectbackground="#003322")
out_scroll = tk.Scrollbar(out_frame, command=output_text.yview, bg="#111111")
output_text.configure(yscrollcommand=out_scroll.set)
out_scroll.pack(side=tk.RIGHT, fill=tk.Y)
output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Action row
action_row = tk.Frame(tab_camera, bg="#0a0a0a")
action_row.pack(fill=tk.X, padx=14, pady=(6, 4))
tk.Button(action_row, text="📁  BROWSE", command=browse_and_analyze,
          bg="#111111", fg="#00ff88", font=("Courier New", 10, "bold"),
          relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
          activebackground="#003322", activeforeground="#00ff88", bd=0
          ).pack(side=tk.LEFT, fill=tk.X, expand=True)
tk.Button(action_row, text="📡  TRIGGER TEST", command=snap_from_stream,
          bg="#111111", fg="#00aaff", font=("Courier New", 10, "bold"),
          relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
          activebackground="#001a33", activeforeground="#00aaff", bd=0
          ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
tk.Button(action_row, text="⬛  MASK ZONES", command=open_mask_wizard,
          bg="#111111", fg="#ff8800", font=("Courier New", 10, "bold"),
          relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
          activebackground="#2a1800", activeforeground="#ff8800", bd=0
          ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
timer_var = tk.StringVar(value="⏱  —")
tk.Label(action_row, textvariable=timer_var, bg="#0a0a0a", fg="#00aaff",
         font=("Courier New", 12, "bold"), padx=14).pack(side=tk.RIGHT)

# Timing / settings row
def _timing_spin(parent, label, var, from_, to, increment, unit="s"):
    f = tk.Frame(parent, bg="#0a0a0a")
    f.pack(side=tk.LEFT, padx=(0, 14))
    tk.Label(f, text=label, bg="#0a0a0a", fg="#555555",
             font=("Courier New", 8)).pack(side=tk.LEFT)
    tk.Spinbox(f, textvariable=var, from_=from_, to=to, increment=increment,
               format="%.2f", width=6,
               bg="#111111", fg="#00ff88", buttonbackground="#1a1a1a",
               relief=tk.FLAT, font=("Courier New", 9),
               insertbackground="#00ff88", highlightthickness=0
               ).pack(side=tk.LEFT, padx=(4, 0))
    tk.Label(f, text=unit, bg="#0a0a0a", fg="#444444",
             font=("Courier New", 8)).pack(side=tk.LEFT, padx=(2, 0))

timing_frame = tk.Frame(tab_camera, bg="#0a0a0a")
timing_frame.pack(fill=tk.X, padx=14, pady=(4, 10))
tk.Label(timing_frame, text="SETTINGS", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(side=tk.LEFT, padx=(0, 14))
_timing_spin(timing_frame, "buffer",   buffer_secs_var,  0.5, 30.0, 0.5)
_timing_spin(timing_frame, "before",   snap_before_var,  0.1, 29.0, 0.1)
_timing_spin(timing_frame, "after",    snap_after_var,   0.0, 29.0, 0.1)
_timing_spin(timing_frame, "padding",  crop_padding_var, 0,   500,  10,  unit="px")
_timing_spin(timing_frame, "min box",  min_box_pct_var,  0.0, 10.0, 0.01, unit="%")

# ═══════════════════════════════════════════════
# TAB 2 — MASTER  (left controls | right HA panel)
# ═══════════════════════════════════════════════

master_cols = tk.Frame(tab_master, bg="#0a0a0a")
master_cols.pack(fill=tk.BOTH, expand=True)

# ── LEFT COLUMN ──────────────────────────────
master_left = tk.Frame(master_cols, bg="#0a0a0a")
master_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 10), pady=16)

# System on/off toggle
def _update_toggle(*_):
    active = system_active_var.get()
    if active:
        system_active.set()
        toggle_btn.config(text="● SYSTEM  ON", fg="#00ff88",
                          activeforeground="#00ff88", activebackground="#003322")
    else:
        system_active.clear()
        toggle_btn.config(text="○ SYSTEM  OFF", fg="#ff4444",
                          activeforeground="#ff4444", activebackground="#2a0000")
    schedule_save()

system_active_var.trace_add("write", _update_toggle)

tk.Label(master_left, text="SYSTEM", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
toggle_btn = tk.Button(
    master_left, text="● SYSTEM  ON",
    command=lambda: system_active_var.set(not system_active_var.get()),
    bg="#111111", fg="#00ff88", font=("Courier New", 14, "bold"),
    relief=tk.FLAT, padx=20, pady=14, cursor="hand2",
    activebackground="#003322", activeforeground="#00ff88", bd=0, width=20)
toggle_btn.pack(anchor="w", pady=(0, 20))

# Camera name
tk.Label(master_left, text="CAMERA TAB NAME", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
tk.Entry(master_left, textvariable=cam_name_var,
         bg="#111111", fg="#00ff88", font=("Courier New", 11),
         relief=tk.FLAT, insertbackground="#00ff88",
         selectbackground="#003322", width=30
         ).pack(anchor="w", pady=(0, 20))

# Models — two side-by-side
model_row = tk.Frame(master_left, bg="#0a0a0a")
model_row.pack(fill=tk.X, pady=(0, 4))

vision_col = tk.Frame(model_row, bg="#0a0a0a")
vision_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
tk.Label(vision_col, text="VISION MODEL", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
vision_model_dropdown = ttk.Combobox(vision_col, textvariable=vision_model_var,
                                      font=("Courier New", 10), style="Dark.TCombobox",
                                      state="readonly", width=20)
vision_model_dropdown.pack(anchor="w", pady=(0, 4))

text_col = tk.Frame(model_row, bg="#0a0a0a")
text_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
tk.Label(text_col, text="TEXT MODEL", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
text_model_dropdown = ttk.Combobox(text_col, textvariable=text_model_var,
                                    font=("Courier New", 10), style="Dark.TCombobox",
                                    state="readonly", width=20)
text_model_dropdown.pack(anchor="w", pady=(0, 4))

tk.Label(master_left, text=f"💾  {SAVE_DIR}", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 8)).pack(anchor="w", pady=(0, 12))

# Vision Prompt
tk.Label(master_left, text="VISION PROMPT", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
vision_prompt_frame = tk.Frame(master_left, bg="#0a0a0a")
vision_prompt_frame.pack(fill=tk.BOTH, expand=True)
vision_prompt_text = tk.Text(vision_prompt_frame, bg="#111111", fg="#999999",
    font=("Courier New", 9), relief=tk.FLAT, padx=10, pady=8,
    wrap=tk.WORD, insertbackground="#00ff88", selectbackground="#003322", height=7)
vps = tk.Scrollbar(vision_prompt_frame, command=vision_prompt_text.yview, bg="#111111")
vision_prompt_text.configure(yscrollcommand=vps.set)
vps.pack(side=tk.RIGHT, fill=tk.Y)
vision_prompt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
vision_prompt_text.insert(tk.END, DEFAULT_VISION_PROMPT)
vision_prompt_text.bind("<KeyRelease>", schedule_save)
vision_prompt_text.bind("<<Paste>>",    schedule_save)

# Text Prompt
tk.Label(master_left, text="TEXT PROMPT", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(10, 4))
text_prompt_frame = tk.Frame(master_left, bg="#0a0a0a")
text_prompt_frame.pack(fill=tk.BOTH, expand=True)
text_prompt_text = tk.Text(text_prompt_frame, bg="#111111", fg="#999999",
    font=("Courier New", 9), relief=tk.FLAT, padx=10, pady=8,
    wrap=tk.WORD, insertbackground="#00ff88", selectbackground="#003322", height=7)
tps = tk.Scrollbar(text_prompt_frame, command=text_prompt_text.yview, bg="#111111")
text_prompt_text.configure(yscrollcommand=tps.set)
tps.pack(side=tk.RIGHT, fill=tk.Y)
text_prompt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
text_prompt_text.insert(tk.END, DEFAULT_TEXT_PROMPT)
text_prompt_text.bind("<KeyRelease>", schedule_save)
text_prompt_text.bind("<<Paste>>",    schedule_save)

# ── RIGHT COLUMN — HOME ASSISTANT ────────────
master_right = tk.Frame(master_cols, bg="#0d0d0d", width=320)
master_right.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 0), pady=0)
master_right.pack_propagate(False)

# Header
ha_header = tk.Frame(master_right, bg="#0d0d0d")
ha_header.pack(fill=tk.X, padx=12, pady=(14, 6))
tk.Label(ha_header, text="HOME ASSISTANT", bg="#0d0d0d", fg="#444444",
         font=("Courier New", 8, "bold")).pack(side=tk.LEFT)
ha_status_var = tk.StringVar(value="○ Disconnected")
tk.Label(ha_header, textvariable=ha_status_var, bg="#0d0d0d", fg="#336633",
         font=("Courier New", 8)).pack(side=tk.LEFT, padx=(8, 0))

# Chicago time clock — updates every second
ha_clock_var = tk.StringVar(value="")
tk.Label(ha_header, textvariable=ha_clock_var, bg="#0d0d0d", fg="#3a6a4a",
         font=("Courier New", 8)).pack(side=tk.RIGHT)

def _update_ha_clock():
    ha_clock_var.set(datetime.now(CHICAGO_TZ).strftime("%I:%M:%S %p CDT"))
    root.after(1000, _update_ha_clock)
_update_ha_clock()

# Scrollable entity list
ha_canvas = tk.Canvas(master_right, bg="#0d0d0d", highlightthickness=0)
ha_scroll = tk.Scrollbar(master_right, orient="vertical",
                          command=ha_canvas.yview, bg="#111111")
ha_canvas.configure(yscrollcommand=ha_scroll.set)
ha_scroll.pack(side=tk.RIGHT, fill=tk.Y)
ha_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

ha_list = tk.Frame(ha_canvas, bg="#0d0d0d")
ha_canvas_window = ha_canvas.create_window((0, 0), window=ha_list, anchor="nw")

def _ha_canvas_resize(e):
    ha_canvas.configure(scrollregion=ha_canvas.bbox("all"))
    ha_canvas.itemconfig(ha_canvas_window, width=e.width)
ha_canvas.bind("<Configure>", _ha_canvas_resize)

# Build entity rows grouped
for group_name, entities in HA_GROUPS:
    tk.Label(ha_list, text=group_name, bg="#0d0d0d", fg="#333333",
             font=("Courier New", 7, "bold"), padx=12,
             anchor="w").pack(fill=tk.X, pady=(8, 2))
    for eid in entities:
        row = tk.Frame(ha_list, bg="#0d0d0d")
        row.pack(fill=tk.X, padx=12, pady=1)
        dot_lbl = tk.Label(row, text="○", bg="#0d0d0d", fg="#333333",
                           font=("Courier New", 10, "bold"), width=2)
        dot_lbl.pack(side=tk.LEFT)
        tk.Label(row, text=HA_NAMES.get(eid, eid), bg="#0d0d0d", fg="#555555",
                 font=("Courier New", 8), width=16, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text="—", bg="#0d0d0d", fg="#444444",
                           font=("Courier New", 8), anchor="w")
        val_lbl.pack(side=tk.LEFT)
        ha_row_labels[eid] = {"dot": dot_lbl, "val": val_lbl}

ha_list.update_idletasks()
ha_canvas.configure(scrollregion=ha_canvas.bbox("all"))

# ---------------------------------------------------------------
# START SERVICES
# ---------------------------------------------------------------
load_config()   # apply saved settings before threads start

threading.Thread(target=run_flask,    daemon=True).start()
threading.Thread(target=queue_worker, daemon=True).start()
threading.Thread(target=buffer_worker, daemon=True).start()
threading.Thread(target=warmup_model, daemon=True).start()
threading.Thread(target=ha_worker,    daemon=True).start()

root.mainloop()
