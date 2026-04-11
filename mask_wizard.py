"""
mask_wizard.py — Toplevel zone editor: mask (exclusion) rects + 15-ft far zone.

Opens a live editor on the most recent buffered frame. Mutates `state` in place:
  state.mask_rects : list[tuple[int, int, int, int]]
  state.far_zones  : list[tuple[int, int, int, int]]
"""

import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk


def open_mask_wizard(root, state, frame_buffer, buffer_lock, status_var,
                     schedule_save, title=None):
    """
    Args:
      root           — Tk root window
      state          — object with mutable `mask_rects` list and `far_zones` list
      frame_buffer   — deque of (timestamp, jpeg_bytes)
      buffer_lock    — threading.Lock protecting frame_buffer
      status_var     — Tk StringVar for the status bar
      schedule_save  — function to debounce-save config
      title          — optional Toplevel window title
    """
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

    disp_w, disp_h = 960, 540
    scale_x = native_w / disp_w
    scale_y = native_h / disp_h

    win = tk.Toplevel(root)
    win.title(title or "Zone Editor — MASK (black exclusion) | FAR ZONE (15+ ft distance reference)")
    win.configure(bg="#0a0a0a")
    win.resizable(False, False)

    rgb = cv2.cvtColor(native, cv2.COLOR_BGR2RGB)
    pil_bg = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)

    canvas = tk.Canvas(win, width=disp_w, height=disp_h,
                       bg="#111111", cursor="crosshair", highlightthickness=0)
    canvas.pack(padx=10, pady=(10, 4))

    info_var = tk.StringVar()
    tk.Label(win, textvariable=info_var, bg="#0a0a0a", fg="#555555",
             font=("Courier New", 8)).pack()

    btn_row = tk.Frame(win, bg="#0a0a0a")
    btn_row.pack(fill=tk.X, padx=10, pady=(4, 10))

    # Current drawing mode: "mask" or "far"
    _mode = {"v": "mask"}

    def _info_text():
        nfz = len(state.far_zones)
        return (f"{len(state.mask_rects)} mask zone(s)   |   far zones: {nfz}"
                f"   |   mode: {'MASK ZONE' if _mode['v'] == 'mask' else '15+ FT ZONE'}"
                f"   |   right-click to delete")

    def redraw():
        canvas.delete("all")
        tk_img = ImageTk.PhotoImage(pil_bg)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas._bg_ref = tk_img

        # Mask zones — black fill, red outline
        for i, (x1, y1, x2, y2) in enumerate(state.mask_rects):
            dx1, dy1 = int(x1 / scale_x), int(y1 / scale_y)
            dx2, dy2 = int(x2 / scale_x), int(y2 / scale_y)
            canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                    fill="black", outline="#ff4444", width=2)
            canvas.create_text(dx1 + 4, dy1 + 4, anchor="nw",
                               text=str(i + 1), fill="#ff4444",
                               font=("Courier New", 9, "bold"))

        # Far zones — no fill, blue outline with label
        for i, (fx1, fy1, fx2, fy2) in enumerate(state.far_zones):
            dx1, dy1 = int(fx1 / scale_x), int(fy1 / scale_y)
            dx2, dy2 = int(fx2 / scale_x), int(fy2 / scale_y)
            canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                    fill="", outline="#4488ff", width=3,
                                    dash=(8, 4))
            canvas.create_text(dx1 + 6, dy1 + 6, anchor="nw",
                               text=f"15+ ft zone {i + 1}", fill="#4488ff",
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
        x1_d, y1_d = min(x0, e.x), min(y0, e.y)
        x2_d, y2_d = max(x0, e.x), max(y0, e.y)
        _draw["start"] = None
        if _draw["live_rect"]:
            canvas.delete(_draw["live_rect"])
            _draw["live_rect"] = None
        if abs(x2_d - x1_d) < 5 or abs(y2_d - y1_d) < 5:
            return
        nx1 = max(0, int(x1_d * scale_x))
        ny1 = max(0, int(y1_d * scale_y))
        nx2 = min(native_w, int(x2_d * scale_x))
        ny2 = min(native_h, int(y2_d * scale_y))
        if _mode["v"] == "mask":
            state.mask_rects.append((nx1, ny1, nx2, ny2))
        else:
            state.far_zones.append((nx1, ny1, nx2, ny2))
        redraw()
        schedule_save()

    def on_right_click(e):
        # Check far zones first
        for i, (fx1, fy1, fx2, fy2) in enumerate(state.far_zones):
            dx1, dy1 = int(fx1 / scale_x), int(fy1 / scale_y)
            dx2, dy2 = int(fx2 / scale_x), int(fy2 / scale_y)
            if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                state.far_zones.pop(i)
                redraw()
                schedule_save()
                return
        # Check mask zones
        for i, (x1, y1, x2, y2) in enumerate(state.mask_rects):
            dx1, dy1 = int(x1 / scale_x), int(y1 / scale_y)
            dx2, dy2 = int(x2 / scale_x), int(y2 / scale_y)
            if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                state.mask_rects.pop(i)
                redraw()
                schedule_save()
                return

    def clear_masks():
        state.mask_rects.clear()
        redraw()
        schedule_save()

    def clear_far():
        state.far_zones.clear()
        redraw()
        schedule_save()

    def toggle_mode():
        _mode["v"] = "far" if _mode["v"] == "mask" else "mask"
        if _mode["v"] == "mask":
            mode_btn.config(text="MODE: MASK ZONE", fg="#ff4444",
                            activeforeground="#ff4444", activebackground="#2a0000")
        else:
            mode_btn.config(text="MODE: 15+ FT ZONE", fg="#4488ff",
                            activeforeground="#4488ff", activebackground="#00112a")
        info_var.set(_info_text())

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>",        on_right_click)

    mode_btn = tk.Button(btn_row, text="MODE: MASK ZONE", command=toggle_mode,
              bg="#111111", fg="#ff4444", font=("Courier New", 9, "bold"),
              relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
              activebackground="#2a0000", activeforeground="#ff4444", bd=0)
    mode_btn.pack(side=tk.LEFT)

    tk.Button(btn_row, text="🗑  CLEAR MASKS", command=clear_masks,
              bg="#111111", fg="#ff4444", font=("Courier New", 9, "bold"),
              relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
              activebackground="#2a0000", activeforeground="#ff4444", bd=0
              ).pack(side=tk.LEFT, padx=(6, 0))

    tk.Button(btn_row, text="✕  CLEAR FAR ZONES", command=clear_far,
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
