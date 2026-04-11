"""
camera_tab.py — Per-camera tab widget: RTSP stream, frame buffer, mask/far zones,
per-camera tuning, and the full camera tab UI. Imported by security_vision.py.
"""

import tkinter as tk
from tkinter import filedialog
import threading
import collections
import base64
import io
import time
from datetime import datetime
import cv2
from PIL import Image, ImageTk

from constants import (
    FRAME_INTERVAL, BUFFER_SECONDS, SNAP_BEFORE_SECS, SNAP_AFTER_SECS,
    CROP_PADDING, MIN_BOX_PCT,
)
from motion import jpeg_apply_mask
from mask_wizard import open_mask_wizard as _open_mask_wizard


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
        self.snap_before_var  = tk.DoubleVar(root, value=SNAP_BEFORE_SECS)
        self.snap_after_var   = tk.DoubleVar(root, value=SNAP_AFTER_SECS)
        self.buffer_secs_var  = tk.DoubleVar(root, value=BUFFER_SECONDS)
        self.crop_padding_var = tk.IntVar(root,    value=CROP_PADDING)
        self.min_box_pct_var  = tk.DoubleVar(root, value=MIN_BOX_PCT)

        # ── Buffer state ──────────────────────────────────────────────────
        self.frame_buffer = collections.deque()
        self.buffer_lock  = threading.Lock()
        self.mask_rects   = []
        self.far_zone     = None

        # ── Stream thread state ───────────────────────────────────────────
        self._stream_running = False
        self._stream_thread  = None

        # Camera is only persisted to config after its stream has produced
        # at least one frame — this prevents saving broken/typo RTSP URLs.
        # Cameras loaded from config start verified=True.
        self.verified = False

        # ── UI widget refs (set in build_ui) ──────────────────────────────
        self.tab_frame         = None
        self.stream_label      = None
        self.stream_status_var = None
        self.detected_label    = None
        self.output_text       = None
        self.timer_var         = None
        self._go_btn           = None
        self._rtsp_entry       = None

        # Debounce-save on any setting change (only after verification).
        for v in (self.rtsp_url_var, self.cam_name_var, self.cam_id_var,
                  self.snap_before_var, self.snap_after_var, self.buffer_secs_var,
                  self.crop_padding_var, self.min_box_pct_var):
            v.trace_add("write", lambda *_: self._maybe_save())

        # Update tab label when name changes
        self.cam_name_var.trace_add("write", self._on_cam_name)

        self.build_ui()

    # ── Name ──────────────────────────────────────────────────────────────
    def _on_cam_name(self, *_):
        try:
            self.notebook.tab(self.tab_frame, text=self.cam_name_var.get() or "Camera")
        except Exception:
            pass

    # ── Save gate ────────────────────────────────────────────────────────
    def _maybe_save(self):
        """Only debounce-save once this camera has been verified via a live frame."""
        if self.verified:
            self.app_refs["schedule_save"]()

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

            # Mark camera verified + save config on first successful frame.
            if not first_success:
                first_success = True
                self.verified = True
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
        _open_mask_wizard(
            self.app_refs["root"],
            self,                          # state: has mask_rects + far_zone
            self.frame_buffer,
            self.buffer_lock,
            self.app_refs["status_var"],
            self.app_refs["schedule_save"],
            title=f"Zone Editor — {self.cam_name_var.get()}",
        )

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
