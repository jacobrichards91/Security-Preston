"""
camera_tab.py — Per-camera tab widget, RTSP buffer worker, and motion helpers.
Imported by security_vision.py.
"""

import tkinter as tk
from tkinter import filedialog
import threading
import collections
import numpy as np
import cv2
from PIL import Image, ImageTk
import base64
import io
import time
from datetime import datetime

FRAME_INTERVAL = 0.2   # seconds between buffer grabs

# ── Pure image helpers ────────────────────────────────────────────────────────

def apply_mask(img_bgr, mask_rects):
    """Paint black over every rect in mask_rects (in-place on a copy)."""
    if not mask_rects:
        return img_bgr
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in mask_rects:
        out[y1:y2, x1:x2] = 0
    return out


def jpeg_apply_mask(jpeg_bytes, mask_rects):
    """Decode JPEG → apply mask → re-encode. Returns bytes."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    img = apply_mask(img, mask_rects)
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return enc.tobytes()


def compute_motion_crop(frame_a_bytes, frame_b_bytes, min_box_pct, crop_padding):
    """
    Diff two JPEG frames.
    Returns (cropped_bytes, debug_images_dict, bbox) where bbox is (x1,y1,x2,y2) or None.
    """
    arr_a = np.frombuffer(frame_a_bytes, dtype=np.uint8)
    arr_b = np.frombuffer(frame_b_bytes, dtype=np.uint8)
    img_a = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        return None, {}, None

    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    h, w = img_a.shape[:2]
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    blur_a = cv2.GaussianBlur(gray_a, (21, 21), 0)
    blur_b = cv2.GaussianBlur(gray_b, (21, 21), 0)
    diff   = cv2.absdiff(blur_a, blur_b)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = w * h * (min_box_pct / 100.0)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    bbox = None
    cropped_bytes = None

    if contours:
        x1 = min(cv2.boundingRect(c)[0] for c in contours)
        y1 = min(cv2.boundingRect(c)[1] for c in contours)
        x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
        pad  = int(crop_padding)
        x1p  = max(0, x1 - pad)
        y1p  = max(0, y1 - pad)
        x2p  = min(w, x2 + pad)
        y2p  = min(h, y2 + pad)
        bbox = (x1p, y1p, x2p, y2p)
        crop = img_b[y1p:y2p, x1p:x2p]
        _, crop_enc = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cropped_bytes = crop_enc.tobytes()

    def _cv2pil(img, gray=False):
        if gray:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    debug = {
        "Frame A\n(before)":  _cv2pil(img_a),
        "Frame B\n(after)":   _cv2pil(img_b),
        "Diff":               _cv2pil(diff,    gray=True),
        "Threshold":          _cv2pil(thresh,  gray=True),
        "Dilated\nmask":      _cv2pil(dilated, gray=True),
    }
    img_b_annot = img_b.copy()
    if bbox:
        x1p, y1p, x2p, y2p = bbox
        cv2.rectangle(img_b_annot, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)
        cv2.drawContours(img_b_annot, contours, -1, (0, 0, 255), 2)
    debug["Bounding\nbox"] = _cv2pil(img_b_annot)
    if cropped_bytes:
        debug["AI\nCrop"] = Image.open(io.BytesIO(cropped_bytes))

    return cropped_bytes, debug, bbox


# ── CameraTab ─────────────────────────────────────────────────────────────────

class CameraTab:
    """
    Encapsulates one camera tab: RTSP stream, frame buffer, mask/far zones,
    per-camera settings, and the full camera tab UI.

    app_refs dict must contain (at construction time):
      root, analysis_queue, system_active, status_var,
      vision_model_var, text_model_var,
      CHICAGO_TZ, SAVE_DIR, schedule_save, _refresh_queue_badge

    These may be None at construction but must be set before first use:
      vision_prompt_text, text_prompt_text,
      master_detected_label, master_vision_text,
      master_input_text, master_result_text
    """

    def __init__(self, notebook, tab_index, app_refs):
        self.notebook  = notebook
        self.app_refs  = app_refs
        self.tab_index = tab_index

        root = app_refs["root"]

        # ── Per-camera settings vars ──────────────────────────────────────
        self.rtsp_url_var     = tk.StringVar(root, value="")
        self.cam_name_var     = tk.StringVar(root, value=f"Camera {tab_index + 1}")
        self.cam_id_var       = tk.StringVar(root, value="")
        self.snap_before_var  = tk.DoubleVar(root, value=2.5)
        self.snap_after_var   = tk.DoubleVar(root, value=2.0)
        self.buffer_secs_var  = tk.DoubleVar(root, value=3.0)
        self.crop_padding_var = tk.IntVar(root,    value=50)
        self.min_box_pct_var  = tk.DoubleVar(root, value=0.05)

        # ── Buffer state ──────────────────────────────────────────────────
        self.frame_buffer = collections.deque()
        self.buffer_lock  = threading.Lock()
        self.mask_rects   = []
        self.far_zone     = None

        # ── Stream thread state ───────────────────────────────────────────
        self._stream_running = False
        self._stream_thread  = None

        # ── UI widget refs (set in build_ui) ──────────────────────────────
        self.tab_frame         = None
        self.stream_label      = None
        self.stream_status_var = None
        self.detected_label    = None
        self.output_text       = None
        self.timer_var         = None
        self._go_btn           = None
        self._rtsp_entry       = None

        # Debounce-save on any setting change
        ss = app_refs["schedule_save"]
        for v in (self.rtsp_url_var, self.cam_name_var, self.cam_id_var,
                  self.snap_before_var, self.snap_after_var, self.buffer_secs_var,
                  self.crop_padding_var, self.min_box_pct_var):
            v.trace_add("write", lambda *_, f=ss: f())

        # Update tab label when name changes
        self.cam_name_var.trace_add("write", self._on_cam_name)

        self.build_ui()

    # ── Name ──────────────────────────────────────────────────────────────
    def _on_cam_name(self, *_):
        try:
            self.notebook.tab(self.tab_frame, text=self.cam_name_var.get() or "Camera")
        except Exception:
            pass

    # ── UI build ──────────────────────────────────────────────────────────
    def build_ui(self):
        self.tab_frame = tk.Frame(self.notebook, bg="#0a0a0a")
        # Caller is responsible for calling notebook.add(self.tab_frame, ...)

        # ── RTSP row ──
        rtsp_row = tk.Frame(self.tab_frame, bg="#0a0a0a")
        rtsp_row.pack(fill=tk.X, padx=14, pady=(10, 2))
        tk.Label(rtsp_row, text="RTSP", bg="#0a0a0a", fg="#444444",
                 font=("Courier New", 8, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self._rtsp_entry = tk.Entry(
            rtsp_row, textvariable=self.rtsp_url_var,
            bg="#111111", fg="#00ff88", insertbackground="#00ff88",
            font=("Courier New", 9), relief=tk.FLAT, bd=2, width=52)
        self._rtsp_entry.pack(side=tk.LEFT, padx=(0, 6))
        self._go_btn = tk.Button(
            rtsp_row, text="▶  GO",
            command=self._toggle_stream,
            bg="#111111", fg="#00ff88", font=("Courier New", 9, "bold"),
            relief=tk.FLAT, padx=10, pady=4, cursor="hand2",
            activebackground="#003322", activeforeground="#00ff88", bd=0)
        self._go_btn.pack(side=tk.LEFT)
        self.stream_status_var = tk.StringVar(value="○ Not connected")
        tk.Label(rtsp_row, textvariable=self.stream_status_var,
                 bg="#0a0a0a", fg="#2a6a3a",
                 font=("Courier New", 8)).pack(side=tk.LEFT, padx=(10, 0))

        # ── Live stream + detected panels ──
        panels = tk.Frame(self.tab_frame, bg="#0a0a0a")
        panels.pack(fill=tk.X, padx=14, pady=(4, 4))

        stream_panel = tk.Frame(panels, bg="#0a0a0a")
        stream_panel.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(stream_panel, text="LIVE STREAM", bg="#0a0a0a", fg="#333333",
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        self.stream_label = tk.Label(
            stream_panel, bg="#0d1a0d",
            text="No stream — enter RTSP URL and click GO",
            fg="#2a5a2a", font=("Courier New", 9), anchor="center", relief=tk.FLAT)
        self.stream_label.pack()

        detected_panel = tk.Frame(panels, bg="#0a0a0a")
        detected_panel.pack(side=tk.LEFT)
        tk.Label(detected_panel, text="LAST DETECTED (AI CROP)", bg="#0a0a0a", fg="#333333",
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        self.detected_label = tk.Label(
            detected_panel, bg="#111111",
            text="Waiting for event...", fg="#333333",
            font=("Courier New", 9), anchor="center", relief=tk.FLAT)
        self.detected_label.pack()

        # ── Output ──
        tk.Label(self.tab_frame, text="OUTPUT", bg="#0a0a0a", fg="#333333",
                 font=("Courier New", 8, "bold"), anchor="w",
                 padx=14).pack(fill=tk.X, pady=(6, 0))
        out_frame = tk.Frame(self.tab_frame, bg="#0a0a0a")
        out_frame.pack(fill=tk.BOTH, expand=True, padx=14)
        self.output_text = tk.Text(
            out_frame, bg="#111111", fg="#e0e0e0",
            font=("Courier New", 11), relief=tk.FLAT,
            padx=10, pady=8, wrap=tk.WORD, height=6,
            state=tk.DISABLED, insertbackground="#00ff88",
            selectbackground="#003322")
        out_scroll = tk.Scrollbar(out_frame, command=self.output_text.yview, bg="#111111")
        self.output_text.configure(yscrollcommand=out_scroll.set)
        out_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Action row ──
        action_row = tk.Frame(self.tab_frame, bg="#0a0a0a")
        action_row.pack(fill=tk.X, padx=14, pady=(6, 4))
        tk.Button(action_row, text="📁  BROWSE",
                  command=self.browse_and_analyze,
                  bg="#111111", fg="#00ff88", font=("Courier New", 10, "bold"),
                  relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
                  activebackground="#003322", activeforeground="#00ff88", bd=0
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(action_row, text="📡  TRIGGER TEST",
                  command=self.snap_from_stream,
                  bg="#111111", fg="#00aaff", font=("Courier New", 10, "bold"),
                  relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
                  activebackground="#001a33", activeforeground="#00aaff", bd=0
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        tk.Button(action_row, text="⬛  MASK ZONES",
                  command=self.open_mask_wizard,
                  bg="#111111", fg="#ff8800", font=("Courier New", 10, "bold"),
                  relief=tk.FLAT, padx=12, pady=10, cursor="hand2",
                  activebackground="#2a1800", activeforeground="#ff8800", bd=0
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.timer_var = tk.StringVar(value="⏱  —")
        tk.Label(action_row, textvariable=self.timer_var,
                 bg="#0a0a0a", fg="#00aaff",
                 font=("Courier New", 12, "bold"), padx=14).pack(side=tk.RIGHT)

        # ── Settings row ──
        timing_frame = tk.Frame(self.tab_frame, bg="#0a0a0a")
        timing_frame.pack(fill=tk.X, padx=14, pady=(4, 10))
        tk.Label(timing_frame, text="SETTINGS", bg="#0a0a0a", fg="#444444",
                 font=("Courier New", 8, "bold")).pack(side=tk.LEFT, padx=(0, 14))
        self._spin(timing_frame, "buffer",   self.buffer_secs_var,  0.5,  30.0, 0.5)
        self._spin(timing_frame, "before",   self.snap_before_var,  0.1,  29.0, 0.1)
        self._spin(timing_frame, "after",    self.snap_after_var,   0.0,  29.0, 0.1)
        self._spin(timing_frame, "padding",  self.crop_padding_var, 0,    500,  10,  unit="px")
        self._spin(timing_frame, "min box",  self.min_box_pct_var,  0.0,  10.0, 0.01, unit="%")
        tk.Label(timing_frame, text="cam ID", bg="#0a0a0a", fg="#444444",
                 font=("Courier New", 8)).pack(side=tk.LEFT, padx=(12, 2))
        tk.Entry(timing_frame, textvariable=self.cam_id_var, width=14,
                 bg="#111111", fg="#00ff88", insertbackground="#00ff88",
                 font=("Courier New", 9), relief=tk.FLAT, bd=2
                 ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(timing_frame, text="name", bg="#0a0a0a", fg="#444444",
                 font=("Courier New", 8)).pack(side=tk.LEFT, padx=(0, 2))
        tk.Entry(timing_frame, textvariable=self.cam_name_var, width=16,
                 bg="#111111", fg="#00ff88", insertbackground="#00ff88",
                 font=("Courier New", 9), relief=tk.FLAT, bd=2
                 ).pack(side=tk.LEFT, padx=(0, 4))

    def _spin(self, parent, label, var, from_, to, increment, unit="s"):
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

    # ── Stream control ────────────────────────────────────────────────────
    def _toggle_stream(self):
        if self._stream_running:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        url = self.rtsp_url_var.get().strip()
        if not url:
            self.stream_status_var.set("⚠  Enter RTSP URL first")
            return
        if self._stream_running:
            return
        self._stream_running = True
        self._go_btn.config(text="■  STOP", fg="#ff4444",
                            activeforeground="#ff4444", activebackground="#2a0000")
        self._stream_thread = threading.Thread(target=self._buffer_worker, daemon=True)
        self._stream_thread.start()

    def stop_stream(self):
        self._stream_running = False
        self._go_btn.config(text="▶  GO", fg="#00ff88",
                            activeforeground="#00ff88", activebackground="#003322")
        self.stream_status_var.set("○ Stopped")

    def _buffer_worker(self):
        root = self.app_refs["root"]
        cap = None
        fail_count = 0
        last_frame_ts = 0.0
        first_success = False

        while self._stream_running:
            url = self.rtsp_url_var.get().strip()
            if not url:
                time.sleep(1)
                continue

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                root.after(0, lambda: self.stream_status_var.set("⏳ Connecting..."))
                print(f"[RTSP:{self.cam_name_var.get()}] Opening {url} "
                      f"(attempt {fail_count + 1})")
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    fail_count += 1
                    root.after(0, lambda n=fail_count: self.stream_status_var.set(
                        f"🔴 Can't connect (attempt {n}) — retrying..."))
                    time.sleep(2)
                    continue
                fail_count = 0

            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                root.after(0, lambda n=fail_count: self.stream_status_var.set(
                    f"🔴 Lost — reconnecting... ({n})"))
                cap.release()
                cap = None
                time.sleep(1)
                continue

            fail_count = 0
            now = time.time()
            if now - last_frame_ts < FRAME_INTERVAL:
                continue
            last_frame_ts = now

            _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            data = enc.tobytes()
            with self.buffer_lock:
                self.frame_buffer.append((now, data))
                cutoff = now - self.buffer_secs_var.get()
                while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
                    self.frame_buffer.popleft()

            b64 = base64.b64encode(data).decode()
            root.after(0, lambda b=b64: self.update_stream_preview(b))

            # Save config once on first successful frame (stream validated)
            if not first_success:
                first_success = True
                root.after(0, self.app_refs["schedule_save"])
                print(f"[RTSP:{self.cam_name_var.get()}] Stream connected — config saved")

        if cap is not None:
            cap.release()

    # ── Frame access ──────────────────────────────────────────────────────
    def get_frame_at(self, target_ts):
        with self.buffer_lock:
            if not self.frame_buffer:
                return None
            return min(self.frame_buffer, key=lambda x: abs(x[0] - target_ts))[1]

    # ── Distance ─────────────────────────────────────────────────────────
    def compute_distance(self, bbox):
        if self.far_zone is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        fx1, fy1, fx2, fy2 = self.far_zone
        if x1 >= fx1 and y1 >= fy1 and x2 <= fx2 and y2 <= fy2:
            return "more than 15 feet from house"
        return "closer than 15 feet to house"

    # ── Enqueue ───────────────────────────────────────────────────────────
    def enqueue_event(self, ts_float, ts_str, source="webhook", image_b64=None):
        app = self.app_refs
        frame_a = frame_b = None
        if source == "webhook":
            ts_a   = ts_float - self.snap_before_var.get()
            ts_b   = ts_float - self.snap_after_var.get()
            frame_a = self.get_frame_at(ts_a)
            frame_b = self.get_frame_at(ts_b)
            if frame_a and self.mask_rects:
                frame_a = jpeg_apply_mask(frame_a, self.mask_rects)
            if frame_b and self.mask_rects:
                frame_b = jpeg_apply_mask(frame_b, self.mask_rects)

        vpt = app.get("vision_prompt_text")
        tpt = app.get("text_prompt_text")
        item = {
            "trigger_ts":    ts_float,
            "ts_str":        ts_str,
            "vision_model":  app["vision_model_var"].get(),
            "text_model":    app["text_model_var"].get(),
            "vision_prompt": vpt.get("1.0", tk.END).strip() if vpt else "",
            "text_prompt":   tpt.get("1.0", tk.END).strip() if tpt else "",
            "source":        source,
            "frame_a":       frame_a,
            "frame_b":       frame_b,
            "enqueued_at":   ts_float,
            "cam_ref":       self,   # routes results back to this tab
        }
        if image_b64:
            item["image_b64"] = image_b64

        app["analysis_queue"].put(item, ts_float)
        app["root"].after(0, app["_refresh_queue_badge"])

    # ── UI update helpers ─────────────────────────────────────────────────
    def show_detected_image(self, image_b64):
        try:
            img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            img.thumbnail((320, 180), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.detected_label.config(image=photo, text="")
            self.detected_label.image = photo
        except Exception:
            pass

    def update_stream_preview(self, image_b64):
        try:
            img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            img.thumbnail((640, 360), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.stream_label.config(image=photo, text="")
            self.stream_label.image = photo
            self.stream_status_var.set(
                f"🟢 Live  {datetime.now().strftime('%H:%M:%S')}")
        except Exception:
            self.stream_status_var.set("⚠  Preview error")

    def finish_analysis(self, image_b64, vision_result,
                        text_input, text_result, ts, elapsed):
        """Update this camera's output panel and the shared Master tab panels."""
        self.timer_var.set(f"⏱  {elapsed:.2f}s")
        self.show_detected_image(image_b64)

        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.tag_configure(
            "dim",   foreground="#555555", font=("Courier New", 9))
        self.output_text.tag_configure(
            "label", foreground="#444444", font=("Courier New", 8, "bold"))
        self.output_text.tag_configure(
            "main",  foreground="#e0e0e0", font=("Courier New", 11))
        self.output_text.insert(tk.END, "👁  VISION\n", "label")
        self.output_text.insert(tk.END, vision_result + "\n\n", "dim")
        self.output_text.insert(tk.END, "🧠  JUDGMENT\n", "label")
        self.output_text.insert(tk.END, text_result, "main")
        self.output_text.config(state=tk.DISABLED)

        # ── Master tab panels ──
        app = self.app_refs
        master_lbl = app.get("master_detected_label")
        if master_lbl:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
                img.thumbnail((220, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                master_lbl.config(image=photo, text="",
                                  width=img.width, height=img.height)
                master_lbl.image = photo
            except Exception:
                pass

        def _set_master_text(key, content):
            w = app.get(key)
            if w:
                w.config(state=tk.NORMAL)
                w.delete("1.0", tk.END)
                w.insert(tk.END, content)
                w.config(state=tk.DISABLED)

        _set_master_text("master_vision_text", vision_result)
        _set_master_text("master_input_text",  text_input)
        _set_master_text("master_result_text", text_result)

    # ── Actions ───────────────────────────────────────────────────────────
    def snap_from_stream(self):
        ts_float = time.time()
        ts_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.enqueue_event(ts_float, ts_str, source="webhook")

    def browse_and_analyze(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All", "*.*")])
        if not path:
            return
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.enqueue_event(time.time(), ts_str, source="manual", image_b64=image_b64)

    # ── Mask / far-zone wizard ────────────────────────────────────────────
    def open_mask_wizard(self):
        root = self.app_refs["root"]
        with self.buffer_lock:
            snap = self.frame_buffer[-1][1] if self.frame_buffer else None
        if snap is None:
            self.app_refs["status_var"].set(
                "⚠  No frame in buffer — connect stream first")
            return

        arr    = np.frombuffer(snap, dtype=np.uint8)
        native = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if native is None:
            return

        native_h, native_w = native.shape[:2]
        disp_w, disp_h = 960, 540
        scale_x = native_w / disp_w
        scale_y = native_h / disp_h

        win = tk.Toplevel(root)
        win.title(f"Zone Editor — {self.cam_name_var.get()}")
        win.configure(bg="#0a0a0a")
        win.resizable(False, False)

        rgb    = cv2.cvtColor(native, cv2.COLOR_BGR2RGB)
        pil_bg = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)
        canvas = tk.Canvas(win, width=disp_w, height=disp_h,
                           bg="#111111", cursor="crosshair", highlightthickness=0)
        canvas.pack(padx=10, pady=(10, 4))
        info_var = tk.StringVar()
        tk.Label(win, textvariable=info_var, bg="#0a0a0a", fg="#555555",
                 font=("Courier New", 8)).pack()
        btn_row = tk.Frame(win, bg="#0a0a0a")
        btn_row.pack(fill=tk.X, padx=10, pady=(4, 10))

        _mode = {"v": "mask"}

        def _info_text():
            fz = "SET" if self.far_zone else "not set"
            m  = "MASK ZONE" if _mode["v"] == "mask" else "15+ FT ZONE"
            return (f"{len(self.mask_rects)} mask zone(s)   |   "
                    f"far zone: {fz}   |   mode: {m}   |   right-click to delete")

        def redraw():
            canvas.delete("all")
            tk_img = ImageTk.PhotoImage(pil_bg)
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            canvas._bg_ref = tk_img
            for i, (x1, y1, x2, y2) in enumerate(self.mask_rects):
                dx1 = int(x1 / scale_x); dy1 = int(y1 / scale_y)
                dx2 = int(x2 / scale_x); dy2 = int(y2 / scale_y)
                canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                        fill="black", outline="#ff4444", width=2)
                canvas.create_text(dx1 + 4, dy1 + 4, anchor="nw",
                                   text=str(i + 1), fill="#ff4444",
                                   font=("Courier New", 9, "bold"))
            if self.far_zone:
                fx1, fy1, fx2, fy2 = self.far_zone
                dx1 = int(fx1 / scale_x); dy1 = int(fy1 / scale_y)
                dx2 = int(fx2 / scale_x); dy2 = int(fy2 / scale_y)
                canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                        fill="", outline="#4488ff",
                                        width=3, dash=(8, 4))
                canvas.create_text(dx1 + 6, dy1 + 6, anchor="nw",
                                   text="15+ ft zone", fill="#4488ff",
                                   font=("Courier New", 9, "bold"))
            info_var.set(_info_text())

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
            if _mode["v"] == "mask":
                _draw["live_rect"] = canvas.create_rectangle(
                    x0, y0, e.x, e.y,
                    fill="black", outline="#ffaa00", width=2, stipple="gray50")
            else:
                _draw["live_rect"] = canvas.create_rectangle(
                    x0, y0, e.x, e.y,
                    fill="", outline="#4488ff", width=3, dash=(8, 4))

        def on_release(e):
            if _draw["start"] is None:
                return
            x0, y0 = _draw["start"]
            x1_d = min(x0, e.x); y1_d = min(y0, e.y)
            x2_d = max(x0, e.x); y2_d = max(y0, e.y)
            _draw["start"] = None
            if _draw["live_rect"]:
                canvas.delete(_draw["live_rect"])
                _draw["live_rect"] = None
            if abs(x2_d - x1_d) < 5 or abs(y2_d - y1_d) < 5:
                return
            nx1 = max(0, int(x1_d * scale_x));  ny1 = max(0, int(y1_d * scale_y))
            nx2 = min(native_w, int(x2_d * scale_x))
            ny2 = min(native_h, int(y2_d * scale_y))
            if _mode["v"] == "mask":
                self.mask_rects.append((nx1, ny1, nx2, ny2))
            else:
                self.far_zone = (nx1, ny1, nx2, ny2)
            redraw()
            self.app_refs["schedule_save"]()

        def on_right_click(e):
            if self.far_zone:
                fx1, fy1, fx2, fy2 = self.far_zone
                dx1 = int(fx1/scale_x); dy1 = int(fy1/scale_y)
                dx2 = int(fx2/scale_x); dy2 = int(fy2/scale_y)
                if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                    self.far_zone = None
                    redraw()
                    self.app_refs["schedule_save"]()
                    return
            for i, (x1, y1, x2, y2) in enumerate(self.mask_rects):
                dx1 = int(x1/scale_x); dy1 = int(y1/scale_y)
                dx2 = int(x2/scale_x); dy2 = int(y2/scale_y)
                if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                    self.mask_rects.pop(i)
                    redraw()
                    self.app_refs["schedule_save"]()
                    return

        canvas.bind("<ButtonPress-1>",   on_press)
        canvas.bind("<B1-Motion>",       on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        canvas.bind("<Button-3>",        on_right_click)

        _mode_btn = [None]

        def toggle_mode():
            _mode["v"] = "far" if _mode["v"] == "mask" else "mask"
            mb = _mode_btn[0]
            if mb:
                if _mode["v"] == "mask":
                    mb.config(text="MODE: MASK ZONE", fg="#ff4444",
                              activeforeground="#ff4444", activebackground="#2a0000")
                else:
                    mb.config(text="MODE: 15+ FT ZONE", fg="#4488ff",
                              activeforeground="#4488ff", activebackground="#00112a")
            info_var.set(_info_text())

        mb = tk.Button(btn_row, text="MODE: MASK ZONE", command=toggle_mode,
                       bg="#111111", fg="#ff4444", font=("Courier New", 9, "bold"),
                       relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                       activebackground="#2a0000", activeforeground="#ff4444", bd=0)
        mb.pack(side=tk.LEFT)
        _mode_btn[0] = mb

        tk.Button(btn_row, text="🗑  CLEAR MASKS",
                  command=lambda: (self.mask_rects.clear(), redraw(),
                                   self.app_refs["schedule_save"]()),
                  bg="#111111", fg="#ff4444", font=("Courier New", 9, "bold"),
                  relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                  activebackground="#2a0000", activeforeground="#ff4444", bd=0
                  ).pack(side=tk.LEFT, padx=(6, 0))

        def _clear_far():
            self.far_zone = None
            redraw()
            self.app_refs["schedule_save"]()

        tk.Button(btn_row, text="✕  CLEAR FAR ZONE", command=_clear_far,
                  bg="#111111", fg="#4488ff", font=("Courier New", 9, "bold"),
                  relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                  activebackground="#00112a", activeforeground="#4488ff", bd=0
                  ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(btn_row, text="✓  DONE", command=win.destroy,
                  bg="#111111", fg="#00ff88", font=("Courier New", 9, "bold"),
                  relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                  activebackground="#003322", activeforeground="#00ff88", bd=0
                  ).pack(side=tk.RIGHT)

        redraw()
        win.grab_set()

    # ── Serialization ─────────────────────────────────────────────────────
    def to_dict(self):
        return {
            "cam_name":    self.cam_name_var.get(),
            "cam_id":      self.cam_id_var.get(),
            "rtsp_url":    self.rtsp_url_var.get(),
            "snap_before": self.snap_before_var.get(),
            "snap_after":  self.snap_after_var.get(),
            "buffer_secs": self.buffer_secs_var.get(),
            "crop_padding": self.crop_padding_var.get(),
            "min_box_pct": self.min_box_pct_var.get(),
            "mask_rects":  [list(r) for r in self.mask_rects],
            "far_zone":    list(self.far_zone) if self.far_zone else None,
        }

    def from_dict(self, data):
        if "cam_name"     in data: self.cam_name_var.set(data["cam_name"])
        if "cam_id"       in data: self.cam_id_var.set(data["cam_id"])
        if "rtsp_url"     in data: self.rtsp_url_var.set(data["rtsp_url"])
        if "snap_before"  in data: self.snap_before_var.set(data["snap_before"])
        if "snap_after"   in data: self.snap_after_var.set(data["snap_after"])
        if "buffer_secs"  in data: self.buffer_secs_var.set(data["buffer_secs"])
        if "crop_padding" in data: self.crop_padding_var.set(data["crop_padding"])
        if "min_box_pct"  in data: self.min_box_pct_var.set(data["min_box_pct"])
        if "mask_rects"   in data:
            self.mask_rects.clear()
            self.mask_rects.extend(tuple(r) for r in data["mask_rects"])
        if data.get("far_zone"):
            self.far_zone = tuple(data["far_zone"])
