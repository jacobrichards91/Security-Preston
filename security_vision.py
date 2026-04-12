import tkinter as tk
from tkinter import ttk
import threading
import base64
import io
import os
import time
import json
import websocket
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime, date
from flask import Flask, request
import logging

from constants import (
    DEFAULT_VISION_MODEL, DEFAULT_TEXT_MODEL,
    CHICAGO_TZ, WEBHOOK_PORT, SAVE_DIR, CONFIG_PATH,
    DEBUG_MODE, DEFAULT_VISION_PROMPT, DEFAULT_TEXT_PROMPT,
    HA_HOST, HA_TOKEN, WATCHED_ENTITIES, HA_NAMES, HA_GROUPS,
)
from priority_queue import NewestFirstQueue
from motion import compute_motion_crop, compute_distance
from ollama_api import fmt_size, list_models, warmup, analyze_image_bytes, analyze_text
from synthetic_sensors import SyntheticSensors
from camera_tab import CameraTab

system_active = threading.Event()
system_active.set()           # ON by default

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Priority queue for detection events (imported from priority_queue.py) ---
analysis_queue = NewestFirstQueue()

# State exposed to the queue-detail popup
_queue_current = None   # item dict currently running, or None
_queue_stage   = ""     # "diff" | "vision" | "judgment" | ""

# --- Camera tabs — list of CameraTab instances (populated as tabs are added) ---
cameras = []

# In-memory detection history — appended by queue_worker after each completed analysis.
# Each entry: {ts, cam_name, vision_result, text_result, image_b64, elapsed}
detection_history = []

# Persist history to disk: one subfolder per day, one JSONL file per day.
HISTORY_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "detection_history"
HISTORY_DIR.mkdir(exist_ok=True)


def _save_history_entry(entry):
    """Append one detection entry to today's JSONL file on disk."""
    try:
        day_str = entry["ts"][:10]   # "YYYY-MM-DD"
        day_dir = HISTORY_DIR / day_str
        day_dir.mkdir(exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with open(day_dir / "detections.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[History] Save error: {e}")


def _load_history():
    """Load all past detection history from disk into detection_history list."""
    try:
        for day_dir in sorted(HISTORY_DIR.iterdir()):
            jl = day_dir / "detections.jsonl"
            if not jl.exists():
                continue
            with open(jl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        detection_history.append(json.loads(line))
        print(f"[History] Loaded {len(detection_history)} entries from disk")
    except Exception as e:
        print(f"[History] Load error: {e}")


# Placeholder refs for the distance sensor labels in the HA panel (set during UI build)
distance_dot_lbl = None
distance_val_lbl = None

# Model metadata fetched from Ollama at startup: {name: {param_size, size_gb, quantization}}
_model_info = {}
# UI label refs for per-model info display (set during UI build)
vision_model_info_lbl = None
text_model_info_lbl   = None

# --- Home Assistant live state (populated by ha_worker) ---
ha_state      = {}          # entity_id -> state string
ha_row_labels = {}          # entity_id -> {"dot": Label, "val": Label}

# Synthetic sensors — instance created after root exists (see below)
synth = None


def _update_distance_display(distance):
    """Update the distance sensor row in the HA panel (must run on main thread)."""
    if distance_dot_lbl is None:
        return
    is_close = "closer" in distance
    distance_dot_lbl.config(
        text="●" if is_close else "○",
        fg="#ff8800" if is_close else "#00ff88"
    )
    distance_val_lbl.config(
        text="close  <15ft" if is_close else "far  >15ft",
        fg="#ff8800" if is_close else "#00ff88"
    )

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
def _update_model_info_labels(*_):
    """Refresh the per-model info labels in the Master tab after a selection change."""
    for var, lbl in [(vision_model_var, vision_model_info_lbl),
                     (text_model_var,   text_model_info_lbl)]:
        if lbl is None:
            continue
        info = _model_info.get(var.get(), {})
        if info:
            parts = []
            if info.get("param_size"):
                parts.append(info["param_size"])
            if info.get("size_gb", 0) > 0:
                parts.append(fmt_size(info["size_gb"]))
            if info.get("quantization"):
                parts.append(info["quantization"])
            if info.get("family"):
                parts.append(info["family"])
            lbl.config(text="  ·  ".join(parts) if parts else "—")
        else:
            lbl.config(text="—")

def fetch_models():
    global _model_info
    try:
        _model_info.clear()
        _model_info.update(list_models())

        model_names = list(_model_info.keys())
        print(f"[Models] {len(model_names)} available:")
        for n in model_names:
            i = _model_info[n]
            print(f"  {n:<40}  {i['param_size']:<6}  {fmt_size(i['size_gb']):<10}  {i['quantization']}")

        if model_names:
            vision_model_dropdown["values"] = model_names
            text_model_dropdown["values"]   = model_names
            # Default vision model to minicpm-v if present, else keep current or use first
            if vision_model_var.get() not in model_names:
                if "minicpm-v:latest" in model_names:
                    vision_model_var.set("minicpm-v:latest")
                elif any("minicpm-v" in m for m in model_names):
                    vision_model_var.set(next(m for m in model_names if "minicpm-v" in m))
                else:
                    vision_model_var.set(model_names[0])
            if text_model_var.get() not in model_names:
                text_model_var.set(model_names[0])

            root.after(0, _update_model_info_labels)

    except Exception as e:
        root.after(0, lambda: status_var.set(f"⚠️ Could not fetch models: {e}"))

def warmup_model():
    status_var.set("⏳ Loading models...")
    fetch_models()
    try:
        for m in {vision_model_var.get(), text_model_var.get()}:
            warmup(m)
        update_queue_status()
    except Exception as e:
        status_var.set(f"⚠️ Warmup failed: {e}")

def build_ha_context():
    """Format current HA state into a readable string for the text model."""
    lines = []
    # Synthetic sensors first
    child, jacob, lauren, emerg = synth.get_state()
    lines.append(
        f"SYNTHETIC SENSORS: "
        f"child_detected={'on' if child else 'off'}, "
        f"jacob_detected={'on' if jacob else 'off'}, "
        f"lauren_detected={'on' if lauren else 'off'}, "
        f"emergency_child_alone={'on' if emerg else 'off'}"
    )
    for group_name, entities in HA_GROUPS:
        parts = [f"{HA_NAMES.get(e, e)}={ha_state.get(e, 'unknown')}" for e in entities]
        lines.append(f"{group_name}: {', '.join(parts)}")
    return "\n".join(lines)

def update_queue_status():
    qsize    = analysis_queue.qsize()
    buf_size = sum(len(c.frame_buffer) for c in cameras)
    processing = _queue_current is not None
    if qsize > 0 or processing:
        status_var.set(f"✅ Ready — webhook :{WEBHOOK_PORT} — 📋 {qsize} waiting — 🎞 {buf_size} frames buffered")
    else:
        status_var.set(f"✅ Ready — webhook :{WEBHOOK_PORT} — 🎞 {buf_size} frames buffered")

# UI ref for the queue badge button in Master tab (set during UI build)
_queue_badge_var = None

def _refresh_queue_badge(*_):
    """Update the clickable queue badge at the top of the Master tab."""
    if _queue_badge_var is None:
        return
    qsize   = analysis_queue.qsize()
    current = _queue_current
    stage   = _queue_stage
    stage_label = {"diff": "diffing", "vision": "vision model",
                   "judgment": "judgment model"}.get(stage, "processing")

    if current is None and qsize == 0:
        _queue_badge_var.set("  ● IDLE  —  click for queue detail")
        try: queue_badge.config(fg="#333333")
        except Exception: pass
    elif current is not None and qsize == 0:
        _queue_badge_var.set(f"  ⏳ {stage_label}  —  nothing waiting")
        try: queue_badge.config(fg="#00aaff")
        except Exception: pass
    else:
        _queue_badge_var.set(f"  ⏳ {stage_label}  —  📋 {qsize} waiting  (click for detail)")
        try: queue_badge.config(fg="#ffaa00")
        except Exception: pass

_queue_win = None

def open_queue_window():
    global _queue_win
    if _queue_win and _queue_win.winfo_exists():
        _queue_win.lift()
        return

    _queue_win = tk.Toplevel(root)
    _queue_win.title("Detection Queue")
    _queue_win.configure(bg="#0a0a0a")
    _queue_win.geometry("520x420")
    _queue_win.resizable(True, True)

    # --- CURRENTLY PROCESSING ---
    tk.Label(_queue_win, text="PROCESSING", bg="#0a0a0a", fg="#444444",
             font=("Courier New", 8, "bold"), anchor="w", padx=14).pack(fill=tk.X, pady=(14, 4))

    curr_frame = tk.Frame(_queue_win, bg="#111111")
    curr_frame.pack(fill=tk.X, padx=14, pady=(0, 8))
    curr_dot  = tk.Label(curr_frame, text="○", bg="#111111", fg="#333333",
                          font=("Courier New", 11, "bold"), padx=8)
    curr_dot.pack(side=tk.LEFT)
    curr_lbl  = tk.Label(curr_frame, text="idle", bg="#111111", fg="#555555",
                          font=("Courier New", 10), anchor="w", pady=6)
    curr_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    curr_stage = tk.Label(curr_frame, text="", bg="#111111", fg="#00aaff",
                           font=("Courier New", 9, "bold"), padx=8)
    curr_stage.pack(side=tk.RIGHT)

    # --- WAITING ---
    tk.Label(_queue_win, text="WAITING  (newest first)", bg="#0a0a0a", fg="#444444",
             font=("Courier New", 8, "bold"), anchor="w", padx=14).pack(fill=tk.X, pady=(4, 4))

    list_frame = tk.Frame(_queue_win, bg="#0a0a0a")
    list_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
    list_text = tk.Text(list_frame, bg="#111111", fg="#555555",
                        font=("Courier New", 9), relief=tk.FLAT,
                        padx=10, pady=8, wrap=tk.WORD, state=tk.DISABLED,
                        selectbackground="#003322")
    list_scroll = tk.Scrollbar(list_frame, command=list_text.yview, bg="#111111")
    list_text.configure(yscrollcommand=list_scroll.set)
    list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    list_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    stage_names = {"diff": "🔬 motion diff", "vision": "👁 vision model",
                   "judgment": "🧠 judgment model"}

    def _refresh_win():
        if not _queue_win.winfo_exists():
            return
        # Currently processing
        cur = _queue_current
        if cur:
            ts   = cur.get("ts_str", "—")
            src  = cur.get("source", "webhook")
            curr_dot.config(text="●", fg="#00ff88")
            curr_lbl.config(text=f"{ts}  [{src}]", fg="#e0e0e0")
            curr_stage.config(text=stage_names.get(_queue_stage, _queue_stage))
        else:
            curr_dot.config(text="○", fg="#333333")
            curr_lbl.config(text="idle", fg="#555555")
            curr_stage.config(text="")
        # Waiting items
        waiting = analysis_queue.peek_all()
        list_text.config(state=tk.NORMAL)
        list_text.delete("1.0", tk.END)
        if not waiting:
            list_text.insert(tk.END, "— queue empty —")
        else:
            for i, it in enumerate(waiting, 1):
                ts  = it.get("ts_str", "—")
                src = it.get("source", "webhook")
                list_text.insert(tk.END, f"  {i}.  {ts}  [{src}]\n")
        list_text.config(state=tk.DISABLED)
        _queue_win.after(400, _refresh_win)

    _refresh_win()

# ---------------------------------------------------------------
# HISTORY WINDOW
# ---------------------------------------------------------------
_history_win = None

def open_history_window():
    global _history_win
    if _history_win and _history_win.winfo_exists():
        _history_win.lift()
        return

    _history_win = tk.Toplevel(root)
    _history_win.title("Detection History")
    _history_win.configure(bg="#0a0a0a")
    _history_win.geometry("960x580")
    _history_win.resizable(True, True)

    pane = tk.Frame(_history_win, bg="#0a0a0a")
    pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ── LEFT: detection list ─────────────────────────────────────────
    left = tk.Frame(pane, bg="#111111", width=290)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
    left.pack_propagate(False)

    count = len(detection_history)
    tk.Label(left, text=f"DETECTIONS  ({count})", bg="#111111", fg="#444444",
             font=("Courier New", 8, "bold"), padx=8, pady=6, anchor="w").pack(fill=tk.X)

    list_frame = tk.Frame(left, bg="#111111")
    list_frame.pack(fill=tk.BOTH, expand=True)
    listbox = tk.Listbox(list_frame, bg="#111111", fg="#888888",
                         font=("Courier New", 9), relief=tk.FLAT,
                         selectbackground="#003322", selectforeground="#00ff88",
                         activestyle="none", borderwidth=0, highlightthickness=0)
    list_scr = tk.Scrollbar(list_frame, command=listbox.yview, bg="#111111")
    listbox.configure(yscrollcommand=list_scr.set)
    list_scr.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    for entry in reversed(detection_history):   # newest first
        secs = entry.get("elapsed", 0)
        listbox.insert(tk.END, f"  {entry['ts']}  [{entry['cam_name']}]  {secs:.1f}s")

    if not detection_history:
        listbox.insert(tk.END, "  — no detections yet —")

    # ── RIGHT: detail panel ──────────────────────────────────────────
    right = tk.Frame(pane, bg="#0a0a0a")
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    detail_img_lbl = tk.Label(right, bg="#111111", text="—", fg="#333333",
                               font=("Courier New", 8), width=28, height=8, anchor="center")
    detail_img_lbl.pack(anchor="nw", pady=(0, 6))

    def _make_txt(parent, title, font_size=9, fg="#888888", height=5):
        tk.Label(parent, text=title, bg="#0a0a0a", fg="#444444",
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        frm = tk.Frame(parent, bg="#0a0a0a")
        frm.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        txt = tk.Text(frm, bg="#111111", fg=fg,
                      font=("Courier New", font_size), relief=tk.FLAT,
                      padx=6, pady=6, wrap=tk.WORD,
                      state=tk.DISABLED, selectbackground="#003322", height=height)
        scr = tk.Scrollbar(frm, command=txt.yview, bg="#111111")
        txt.configure(yscrollcommand=scr.set)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return txt

    vision_txt = _make_txt(right, "VISION",   font_size=8, fg="#666666", height=5)
    judg_txt   = _make_txt(right, "JUDGMENT", font_size=9, fg="#e0e0e0", height=6)

    def _on_select(evt):
        sel = listbox.curselection()
        if not sel or not detection_history:
            return
        idx = len(detection_history) - 1 - sel[0]   # newest-first mapping
        if idx < 0 or idx >= len(detection_history):
            return
        entry = detection_history[idx]

        img_b64 = entry.get("image_b64")
        if img_b64:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
                img.thumbnail((220, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                detail_img_lbl.config(image=photo, text="",
                                      width=img.width, height=img.height)
                detail_img_lbl.image = photo
            except Exception:
                detail_img_lbl.config(image="", text="—")
        else:
            detail_img_lbl.config(image="", text="—")

        def _set(w, text):
            w.config(state=tk.NORMAL)
            w.delete("1.0", tk.END)
            w.insert(tk.END, text)
            w.config(state=tk.DISABLED)

        _set(vision_txt, entry.get("vision_result", ""))
        _set(judg_txt,   entry.get("text_result",   ""))

    listbox.bind("<<ListboxSelect>>", _on_select)
    if detection_history:
        listbox.selection_set(0)
        listbox.event_generate("<<ListboxSelect>>")

# ---------------------------------------------------------------
# QUEUE WORKER
# ---------------------------------------------------------------
def _set_stage(stage, ts_str=""):
    global _queue_stage
    _queue_stage = stage
    root.after(0, _refresh_queue_badge)
    label = {"diff": "🔬 Motion diff", "vision": "👁 Vision",
             "judgment": "🧠 Judgment"}.get(stage, "📡 Processing")
    root.after(0, lambda l=label, t=ts_str: status_var.set(
        f"{l} [{t}]" + (f" — {analysis_queue.qsize()} waiting" if analysis_queue.qsize() else "")))

def queue_worker():
    global _queue_current, _queue_stage
    while True:
        item = analysis_queue.get()
        _queue_current = item
        root.after(0, _refresh_queue_badge)
        try:
            ts_str        = item["ts_str"]
            vision_model  = item["vision_model"]
            text_model    = item["text_model"]
            vision_prompt = item["vision_prompt"]
            text_prompt   = item["text_prompt"]
            source        = item.get("source", "webhook")
            cam           = item.get("cam_ref")   # CameraTab the event belongs to

            if source == "manual":
                image_bytes = base64.b64decode(item["image_b64"])
                _set_stage("vision", ts_str)
                t0 = time.time()
                vision_result = analyze_image_bytes(image_bytes, vision_prompt, vision_model)
                synth.check(vision_result)

                _set_stage("judgment", ts_str)
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
                _ti = full_text_prompt
                _hist = {
                    "ts":            ts_str,
                    "cam_name":      cam.cam_name_var.get() if cam is not None else "—",
                    "vision_result": vision_result,
                    "text_result":   text_result,
                    "image_b64":     _b,
                    "elapsed":       time.time() - item["trigger_ts"],
                }
                detection_history.append(_hist)
                _save_history_entry(_hist)
                if cam is not None:
                    root.after(0, lambda b=_b, vr=vision_result, ti=_ti, tr=text_result,
                                        t=ts_str, e=elapsed, c=cam:
                               c.finish_analysis(b, vr, ti, tr, t, e))
                save_event_background(None, None, image_bytes, text_result, None, ts_str)
                continue

            # --- Use pre-captured frames (grabbed at enqueue time) ---
            frame_a = item.get("frame_a")
            frame_b = item.get("frame_b")

            if frame_a is None or frame_b is None:
                root.after(0, lambda t=ts_str: status_var.set(
                    f"⚠️ [{t}] No frames captured at trigger time"))
                continue

            # Motion tuning comes from the originating camera's own settings.
            mb_pct    = cam.min_box_pct_var.get() if cam is not None else 0.05
            pad_px    = cam.crop_padding_var.get() if cam is not None else 50
            far_zones = cam.far_zones if cam is not None else []

            _set_stage("diff", ts_str)
            cropped_bytes, debug_imgs, bbox = compute_motion_crop(
                frame_a, frame_b, mb_pct, pad_px
            )

            if debug_imgs:
                root.after(0, lambda d=debug_imgs: show_debug_window(d))
            if cropped_bytes is None:
                root.after(0, lambda t=ts_str: status_var.set(
                    f"⚠️ [{t}] No motion — using full frame"))
                cropped_bytes = frame_b

            crop_b64 = base64.b64encode(cropped_bytes).decode()
            if cam is not None:
                root.after(0, lambda b=crop_b64, c=cam: c.show_detected_image(b))

            distance = compute_distance(bbox, far_zones)
            if distance:
                root.after(0, lambda d=distance: _update_distance_display(d))

            _set_stage("vision", ts_str)
            t0 = time.time()
            vision_result = analyze_image_bytes(cropped_bytes, vision_prompt, vision_model)
            synth.check(vision_result)

            _set_stage("judgment", ts_str)
            chicago_now = datetime.now(CHICAGO_TZ).strftime("%A %B %d %Y  %I:%M:%S %p %Z")
            distance_line = f"\nDistance from house: {distance}" if distance else ""
            full_text_prompt = (
                f"{text_prompt}\n\n"
                f"Time: {chicago_now}\n\n"
                f"Home state:\n{build_ha_context()}{distance_line}\n\n"
                f"Visual observation:\n{vision_result}"
            )
            text_result = analyze_text(full_text_prompt, text_model)
            elapsed = time.time() - t0

            _b64 = crop_b64
            _ti  = full_text_prompt
            _hist = {
                "ts":            ts_str,
                "cam_name":      cam.cam_name_var.get() if cam is not None else "—",
                "vision_result": vision_result,
                "text_result":   text_result,
                "image_b64":     _b64,
                "elapsed":       time.time() - item["trigger_ts"],
            }
            detection_history.append(_hist)
            _save_history_entry(_hist)
            if cam is not None:
                root.after(0, lambda b=_b64, vr=vision_result, ti=_ti, tr=text_result,
                                    t=ts_str, e=elapsed, c=cam:
                           c.finish_analysis(b, vr, ti, tr, t, e))

            save_event_background(frame_a, frame_b, cropped_bytes,
                                  f"VISION:\n{vision_result}\n\nJUDGMENT:\n{text_result}",
                                  bbox, ts_str)

        except Exception as e:
            import traceback
            traceback.print_exc()
            root.after(0, lambda err=str(e): status_var.set(f"⚠️ Error: {err}"))
        finally:
            _queue_current = None
            _queue_stage   = ""
            analysis_queue.task_done()
            root.after(0, _refresh_queue_badge)
            root.after(0, update_queue_status)

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
        print(f"[Webhook] payload: {raw[:500]}")

        alarm = data.get("alarm", {})
        triggers = alarm.get("triggers", [])
        trigger_key = triggers[0].get("key", "unknown") if triggers else "unknown"
        # The triggering camera is in triggers[].device, NOT sources[]
        # (sources lists every camera in the alarm group).
        trigger_device = triggers[0].get("device", "").strip().upper() if triggers else ""
        print(f"[Webhook] trigger_key={trigger_key!r}  trigger_device={trigger_device!r}")

        # Route by trigger device — match against each camera tab's cam_id.
        matched = None
        for cam in cameras:
            cid = cam.cam_id_var.get().strip().upper()
            if cid and cid == trigger_device:
                matched = cam
                break

        if matched is not None:
            print(f"[Webhook] MATCHED camera '{matched.cam_name_var.get()}' "
                  f"(ID={matched.cam_id_var.get()}) — queuing analysis")
            ts_float = time.time()
            matched.enqueue_event(ts_float, ts_str, source="webhook")
            return "OK", 200
        else:
            configured_ids = [c.cam_id_var.get() for c in cameras]
            print(f"[Webhook] SKIPPED — trigger device {trigger_device!r} "
                  f"not in configured IDs {configured_ids!r}")
            root.after(0, lambda: status_var.set(
                f"Webhook received — no camera ID match for {trigger_device}"
            ))
            return "SKIP", 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Webhook error] {e}")
        return "ERROR", 500

def run_flask():
    flask_app.run(host="0.0.0.0", port=WEBHOOK_PORT, debug=False, use_reloader=False)

# ---------------------------------------------------------------
# CONFIG PERSISTENCE
# ---------------------------------------------------------------
_save_job = None
_loading  = False   # suppresses schedule_save while load_config is running

def save_config(*_):
    """Write global settings and every verified camera to config.json."""
    try:
        data = {
            "vision_model":    vision_model_var.get(),
            "text_model":      text_model_var.get(),
            "vision_prompt":   vision_prompt_text.get("1.0", tk.END).rstrip("\n"),
            "text_prompt":     text_prompt_text.get("1.0", tk.END).rstrip("\n"),
            "system_active":   system_active_var.get(),
            # Only persist cameras that have been verified by a live frame.
            "cameras":         [c.to_dict() for c in cameras if c.verified],
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[Config] Save error: {e}")

def schedule_save(*_):
    """Debounce saves — write 400 ms after the last change."""
    global _save_job
    if _loading:
        return
    if _save_job:
        root.after_cancel(_save_job)
    _save_job = root.after(400, save_config)

def load_config():
    """Read config.json and apply all stored values to the UI + rebuild camera tabs."""
    global _loading
    if not CONFIG_PATH.exists():
        return
    _loading = True
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
        if "system_active" in data:
            system_active_var.set(data["system_active"])
        # Rebuild each saved camera tab.
        for cam_data in data.get("cameras", []):
            cam = add_camera_tab(initial_data=cam_data, autostart=True)
            if cam is not None:
                cam.verified = True   # trust previously-saved cameras
        print(f"[Config] Loaded from {CONFIG_PATH}")
    except Exception as e:
        print(f"[Config] Load error: {e}")
    finally:
        _loading = False

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
root = tk.Tk()
root.title("Security Vision — Preston" + (" [DEBUG]" if DEBUG_MODE else ""))
root.geometry("2400x900")
root.configure(bg="#0a0a0a")
root.resizable(True, True)

# Force combobox dropdown list to use black text on white background (Windows fix)
root.option_add("*TCombobox*Listbox.background",       "#ffffff")
root.option_add("*TCombobox*Listbox.foreground",       "#000000")
root.option_add("*TCombobox*Listbox.selectBackground", "#cceecc")
root.option_add("*TCombobox*Listbox.selectForeground", "#000000")

# Synthetic sensor manager (UI label refs wired in after HA panel is built)
synth = SyntheticSensors(root, ha_state)

# --- Shared tunable vars (per-camera vars live on each CameraTab instance) ---
vision_model_var  = tk.StringVar(value=DEFAULT_VISION_MODEL)
text_model_var    = tk.StringVar(value=DEFAULT_TEXT_MODEL)
system_active_var = tk.BooleanVar(value=True)

for _v in (vision_model_var, text_model_var):
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
    fieldbackground="#f5f5f5", background="#f5f5f5",
    foreground="#000000", arrowcolor="#333333",
    selectbackground="#0078d7", selectforeground="#ffffff",
    bordercolor="#cccccc", lightcolor="#f5f5f5", darkcolor="#f5f5f5")

notebook = ttk.Notebook(root, style="Dark.TNotebook")
notebook.pack(fill=tk.BOTH, expand=True)

# Master tab is always present. Camera tabs are inserted BEFORE it at runtime.
# A ghost "+" tab lives after Master; selecting it creates a new camera tab.
tab_master = tk.Frame(notebook, bg="#0a0a0a")
notebook.add(tab_master, text="Master")
_plus_tab = tk.Frame(notebook, bg="#0a0a0a")
notebook.add(_plus_tab, text="  +  ")

# ═══════════════════════════════════════════════
# MASTER TAB  (queue badge | left controls | right HA panel)
# ═══════════════════════════════════════════════

# Queue status badge — top of Master tab, click to open detail window
_queue_badge_var = tk.StringVar(value="● idle")
queue_badge = tk.Button(
    tab_master, textvariable=_queue_badge_var,
    command=open_queue_window,
    bg="#0d0d0d", fg="#333333", activebackground="#1a1a1a", activeforeground="#00ff88",
    font=("Courier New", 9, "bold"), relief=tk.FLAT, anchor="w",
    padx=14, pady=6, cursor="hand2", bd=0)
queue_badge.pack(fill=tk.X, side=tk.TOP)
tk.Frame(tab_master, bg="#1a1a1a", height=1).pack(fill=tk.X, side=tk.TOP)

master_cols = tk.Frame(tab_master, bg="#0a0a0a")
master_cols.pack(fill=tk.X, expand=False)

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

# Models — two side-by-side
model_row = tk.Frame(master_left, bg="#0a0a0a")
model_row.pack(fill=tk.X, pady=(0, 4))

vision_col = tk.Frame(model_row, bg="#0a0a0a")
vision_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
tk.Label(vision_col, text="VISION MODEL", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
vision_model_dropdown = ttk.Combobox(vision_col, textvariable=vision_model_var,
                                      font=("Courier New", 10), style="Dark.TCombobox",
                                      state="readonly", width=28)
vision_model_dropdown.pack(anchor="w", pady=(0, 2))
vision_model_info_lbl = tk.Label(vision_col, text="—", bg="#0a0a0a", fg="#333333",
                                  font=("Courier New", 7), anchor="w")
vision_model_info_lbl.pack(anchor="w", pady=(0, 8))

text_col = tk.Frame(model_row, bg="#0a0a0a")
text_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
tk.Label(text_col, text="TEXT MODEL", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 4))
text_model_dropdown = ttk.Combobox(text_col, textvariable=text_model_var,
                                    font=("Courier New", 10), style="Dark.TCombobox",
                                    state="readonly", width=28)
text_model_dropdown.pack(anchor="w", pady=(0, 2))
text_model_info_lbl = tk.Label(text_col, text="—", bg="#0a0a0a", fg="#333333",
                                font=("Courier New", 7), anchor="w")
text_model_info_lbl.pack(anchor="w", pady=(0, 8))

# Refresh info labels whenever selection changes
vision_model_var.trace_add("write", _update_model_info_labels)
text_model_var.trace_add("write",   _update_model_info_labels)

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

# Helper to build a generic sensor row in the HA list
def _ha_sensor_row(label_text, init_val="—"):
    row = tk.Frame(ha_list, bg="#0d0d0d")
    row.pack(fill=tk.X, padx=12, pady=1)
    dot = tk.Label(row, text="○", bg="#0d0d0d", fg="#333333",
                   font=("Courier New", 10, "bold"), width=2)
    dot.pack(side=tk.LEFT)
    tk.Label(row, text=label_text, bg="#0d0d0d", fg="#555555",
             font=("Courier New", 8), width=16, anchor="w").pack(side=tk.LEFT)
    val = tk.Label(row, text=init_val, bg="#0d0d0d", fg="#444444",
                   font=("Courier New", 8), anchor="w")
    val.pack(side=tk.LEFT)
    return dot, val

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

    # Inject synthetic sensors under OCCUPANCY
    if group_name == "OCCUPANCY":
        _sc_dot, _sc_val = _ha_sensor_row("child detected", "clear")
        _sj_dot, _sj_val = _ha_sensor_row("jacob detected", "clear")
        _sl_dot, _sl_val = _ha_sensor_row("lauren detected","clear")
        # Emergency row
        emerg_row = tk.Frame(ha_list, bg="#0d0d0d")
        emerg_row.pack(fill=tk.X, padx=12, pady=1)
        _se_dot = tk.Label(emerg_row, text="○", bg="#0d0d0d", fg="#333333",
                           font=("Courier New", 10, "bold"), width=2)
        _se_dot.pack(side=tk.LEFT)
        tk.Label(emerg_row, text="⚠ child alone", bg="#0d0d0d", fg="#884400",
                 font=("Courier New", 8, "bold"), width=16, anchor="w").pack(side=tk.LEFT)
        _se_val = tk.Label(emerg_row, text="clear", bg="#0d0d0d", fg="#444444",
                           font=("Courier New", 8), anchor="w")
        _se_val.pack(side=tk.LEFT)
        synth.set_ui_refs(
            child_dot=_sc_dot,  child_val=_sc_val,
            jacob_dot=_sj_dot,  jacob_val=_sj_val,
            lauren_dot=_sl_dot, lauren_val=_sl_val,
            emerg_dot=_se_dot,  emerg_val=_se_val,
        )

# CAMERA group — distance sensor (populated by detection events)
tk.Label(ha_list, text="CAMERA", bg="#0d0d0d", fg="#333333",
         font=("Courier New", 7, "bold"), padx=12,
         anchor="w").pack(fill=tk.X, pady=(8, 2))
dist_row = tk.Frame(ha_list, bg="#0d0d0d")
dist_row.pack(fill=tk.X, padx=12, pady=1)
_ddot = tk.Label(dist_row, text="○", bg="#0d0d0d", fg="#444444",
                 font=("Courier New", 10, "bold"), width=2)
_ddot.pack(side=tk.LEFT)
tk.Label(dist_row, text="distance", bg="#0d0d0d", fg="#555555",
         font=("Courier New", 8), width=16, anchor="w").pack(side=tk.LEFT)
_dval = tk.Label(dist_row, text="—", bg="#0d0d0d", fg="#444444",
                 font=("Courier New", 8), anchor="w")
_dval.pack(side=tk.LEFT)
# Wire up to the module-level refs used by _update_distance_display
distance_dot_lbl = _ddot
distance_val_lbl  = _dval

ha_list.update_idletasks()
ha_canvas.configure(scrollregion=ha_canvas.bbox("all"))

# ═══════════════════════════════════════════════
# MASTER TAB — DETECTION RESULTS (bottom panel)
# ═══════════════════════════════════════════════
tk.Frame(tab_master, bg="#222222", height=1).pack(fill=tk.X, padx=14, pady=(4, 0))

det_outer = tk.Frame(tab_master, bg="#0a0a0a")
det_outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 10))

_det_hdr = tk.Frame(det_outer, bg="#0a0a0a")
_det_hdr.pack(fill=tk.X, pady=(0, 6))
tk.Label(_det_hdr, text="LAST DETECTION", bg="#0a0a0a", fg="#444444",
         font=("Courier New", 8, "bold")).pack(side=tk.LEFT)
tk.Button(_det_hdr, text="📋  HISTORY", command=open_history_window,
          bg="#111111", fg="#666666", font=("Courier New", 8, "bold"),
          relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
          activebackground="#1a1a1a", activeforeground="#00ff88", bd=0
          ).pack(side=tk.RIGHT)

det_inner = tk.Frame(det_outer, bg="#0a0a0a")
det_inner.pack(fill=tk.BOTH, expand=True)

# 1. Cropped image
det_img_col = tk.Frame(det_inner, bg="#0a0a0a")
det_img_col.pack(side=tk.LEFT, padx=(0, 10), anchor="n")
tk.Label(det_img_col, text="CROP", bg="#0a0a0a", fg="#333333",
         font=("Courier New", 7, "bold")).pack(anchor="w")
master_detected_label = tk.Label(det_img_col, bg="#111111",
                                  text="—", fg="#333333",
                                  font=("Courier New", 8),
                                  width=28, height=9, anchor="center")
master_detected_label.pack()

def _make_det_col(parent, title, font_size=8, fg="#888888"):
    col = tk.Frame(parent, bg="#0a0a0a")
    col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
    tk.Label(col, text=title, bg="#0a0a0a", fg="#333333",
             font=("Courier New", 7, "bold")).pack(anchor="w")
    inner = tk.Frame(col, bg="#0a0a0a")
    inner.pack(fill=tk.BOTH, expand=True)
    txt = tk.Text(inner, bg="#111111", fg=fg,
                  font=("Courier New", font_size), relief=tk.FLAT,
                  padx=6, pady=6, wrap=tk.WORD,
                  state=tk.DISABLED, selectbackground="#003322")
    scr = tk.Scrollbar(inner, command=txt.yview, bg="#111111")
    txt.configure(yscrollcommand=scr.set)
    scr.pack(side=tk.RIGHT, fill=tk.Y)
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return txt

# 2. Vision model result
master_vision_text = _make_det_col(det_inner, "VISION RESULT",  font_size=8, fg="#666666")

# 3. Text model input (full prompt sent to text model)
master_input_text  = _make_det_col(det_inner, "TEXT MODEL INPUT", font_size=7, fg="#444444")

# 4. Text model result
master_result_text = _make_det_col(det_inner, "JUDGMENT",       font_size=9, fg="#e0e0e0")

# ═══════════════════════════════════════════════
# CAMERA TAB MANAGEMENT  (multi-camera: "+" tab adds new cameras)
# ═══════════════════════════════════════════════

# app_refs is passed to every CameraTab so it can reach shared state.
app_refs = {
    "root":                  root,
    "analysis_queue":        analysis_queue,
    "status_var":            status_var,
    "vision_model_var":      vision_model_var,
    "text_model_var":        text_model_var,
    "vision_prompt_text":    vision_prompt_text,
    "text_prompt_text":      text_prompt_text,
    "schedule_save":         schedule_save,
    "_refresh_queue_badge":  _refresh_queue_badge,
    "master_detected_label": master_detected_label,
    "master_vision_text":    master_vision_text,
    "master_input_text":     master_input_text,
    "master_result_text":    master_result_text,
}


def add_camera_tab(initial_data=None, autostart=False):
    """
    Create a new CameraTab, insert it just before the Master tab, and select it.
    If initial_data is provided, seed the camera's fields from that dict.
    If autostart is True, immediately begin streaming (used when loading saved cameras).
    Returns the new CameraTab (or None on error).
    """
    try:
        cam = CameraTab(notebook, len(cameras), app_refs)
    except Exception as e:
        print(f"[Camera] Failed to build tab: {e}")
        return None

    # Insert the new tab just before '+'. Master stays at the far left.
    plus_pos = notebook.index(_plus_tab)
    tab_name = (initial_data.get("cam_name") if initial_data else None) \
               or f"Camera {len(cameras) + 1}"
    notebook.insert(plus_pos, cam.tab_frame, text=tab_name)

    cameras.append(cam)

    if initial_data:
        cam.from_dict(initial_data)

    notebook.select(cam.tab_frame)

    if autostart and cam.rtsp_url_var.get().strip():
        cam.start_stream()

    return cam


def _on_tab_changed(event):
    """Intercept selection of the ghost '+' tab and spawn a new camera instead."""
    try:
        current = notebook.select()
    except Exception:
        return
    if current == str(_plus_tab):
        add_camera_tab()

notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)


# ---------------------------------------------------------------
# START SERVICES
# ---------------------------------------------------------------
_load_history()   # restore past detections from disk
load_config()     # rebuilds camera tabs from saved config

# First run: no saved cameras yet → give the user an empty starting tab.
if not cameras:
    add_camera_tab()

threading.Thread(target=run_flask,    daemon=True).start()
threading.Thread(target=queue_worker, daemon=True).start()
threading.Thread(target=warmup_model, daemon=True).start()
threading.Thread(target=ha_worker,    daemon=True).start()

root.mainloop()
