"""
mask_wizard.py — Toplevel zone editor: mask rects, far zones, and motion zone polygon.

Opens a live editor on the most recent buffered frame. Mutates `state` in place:
  state.mask_rects  : list[tuple[int, int, int, int]]
  state.far_zones   : list[tuple[int, int, int, int]]
  state.motion_zone : list[tuple[int, int]]  — polygon points (native res)
"""

import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk


_HANDLE_RADIUS = 6   # px radius for draggable polygon handles


def open_mask_wizard(root, state, frame_buffer, buffer_lock, status_var,
                     schedule_save, title=None):
    """
    Args:
      root           — Tk root window
      state          — object with mutable mask_rects, far_zones, motion_zone
      frame_buffer   — deque of (timestamp, jpeg_bytes)
      buffer_lock    — threading.Lock protecting frame_buffer
      status_var     — Tk StringVar for the status bar
      schedule_save  — function to debounce-save config
      title          — optional Toplevel window title
    """
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
    win.title(title or "Zone Editor")
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

    # Drawing modes: "mask", "far", "motion"
    _mode = {"v": "mask"}
    # Polygon drag state
    _poly_drag = {"idx": None}

    # ── helpers ──────────────────────────────────────────────────────

    def _to_disp(nx, ny):
        return int(nx / scale_x), int(ny / scale_y)

    def _to_native(dx, dy):
        return max(0, min(native_w, int(dx * scale_x))), \
               max(0, min(native_h, int(dy * scale_y)))

    def _mode_label():
        m = _mode["v"]
        if m == "mask":    return "MASK ZONE"
        if m == "far":     return "15+ FT ZONE"
        return "MOTION ZONE"

    def _info_text():
        nfz = len(state.far_zones)
        mz  = len(state.motion_zone)
        return (f"{len(state.mask_rects)} mask | {nfz} far | motion: {mz} pts"
                f"   |   mode: {_mode_label()}"
                f"   |   right-click to delete")

    # ── redraw ───────────────────────────────────────────────────────

    def redraw():
        canvas.delete("all")
        tk_img = ImageTk.PhotoImage(pil_bg)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas._bg_ref = tk_img

        # Mask zones — black fill, red outline
        for i, (x1, y1, x2, y2) in enumerate(state.mask_rects):
            dx1, dy1 = _to_disp(x1, y1)
            dx2, dy2 = _to_disp(x2, y2)
            canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                    fill="black", outline="#ff4444", width=2)
            canvas.create_text(dx1 + 4, dy1 + 4, anchor="nw",
                               text=str(i + 1), fill="#ff4444",
                               font=("Courier New", 9, "bold"))

        # Far zones — dashed blue outline
        for i, (fx1, fy1, fx2, fy2) in enumerate(state.far_zones):
            dx1, dy1 = _to_disp(fx1, fy1)
            dx2, dy2 = _to_disp(fx2, fy2)
            canvas.create_rectangle(dx1, dy1, dx2, dy2,
                                    fill="", outline="#4488ff", width=3,
                                    dash=(8, 4))
            canvas.create_text(dx1 + 6, dy1 + 6, anchor="nw",
                               text=f"15+ ft zone {i + 1}", fill="#4488ff",
                               font=("Courier New", 9, "bold"))

        # Motion zone polygon — green filled outline with draggable handles
        if state.motion_zone:
            disp_pts = [_to_disp(nx, ny) for nx, ny in state.motion_zone]
            flat = [c for pt in disp_pts for c in pt]
            if len(disp_pts) >= 3:
                canvas.create_polygon(flat, fill="", outline="#00ff88",
                                      width=2, dash=(6, 3))
            elif len(disp_pts) == 2:
                canvas.create_line(flat, fill="#00ff88", width=2, dash=(6, 3))
            # Handles
            for i, (dx, dy) in enumerate(disp_pts):
                r = _HANDLE_RADIUS
                canvas.create_oval(dx - r, dy - r, dx + r, dy + r,
                                   fill="#00ff88", outline="#003322", width=1)
                canvas.create_text(dx, dy, text=str(i + 1), fill="#003322",
                                   font=("Courier New", 7, "bold"))

        info_var.set(_info_text())

    # ── rect drawing (mask / far modes) ──────────────────────────────

    _draw = {"start": None, "live_rect": None}

    def on_press(e):
        if _mode["v"] == "motion":
            # Check if clicking near an existing handle to drag it
            for i, (nx, ny) in enumerate(state.motion_zone):
                dx, dy = _to_disp(nx, ny)
                if abs(e.x - dx) <= _HANDLE_RADIUS + 3 and abs(e.y - dy) <= _HANDLE_RADIUS + 3:
                    _poly_drag["idx"] = i
                    return
            # Not near a handle — add a new point (up to 10)
            if len(state.motion_zone) < 10:
                nx, ny = _to_native(e.x, e.y)
                state.motion_zone.append((nx, ny))
                redraw()
                schedule_save()
            return
        _draw["start"] = (e.x, e.y)
        if _draw["live_rect"]:
            canvas.delete(_draw["live_rect"])
            _draw["live_rect"] = None

    def on_drag(e):
        if _mode["v"] == "motion":
            idx = _poly_drag["idx"]
            if idx is not None:
                nx, ny = _to_native(e.x, e.y)
                state.motion_zone[idx] = (nx, ny)
                redraw()
            return
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
        if _mode["v"] == "motion":
            if _poly_drag["idx"] is not None:
                _poly_drag["idx"] = None
                schedule_save()
            return
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
        nx1, ny1 = _to_native(x1_d, y1_d)
        nx2, ny2 = _to_native(x2_d, y2_d)
        if _mode["v"] == "mask":
            state.mask_rects.append((nx1, ny1, nx2, ny2))
        else:
            state.far_zones.append((nx1, ny1, nx2, ny2))
        redraw()
        schedule_save()

    def on_right_click(e):
        # In motion mode — right-click a handle to remove that point
        if _mode["v"] == "motion":
            for i, (nx, ny) in enumerate(state.motion_zone):
                dx, dy = _to_disp(nx, ny)
                if abs(e.x - dx) <= _HANDLE_RADIUS + 5 and abs(e.y - dy) <= _HANDLE_RADIUS + 5:
                    state.motion_zone.pop(i)
                    redraw()
                    schedule_save()
                    return
            return
        # Check far zones
        for i, (fx1, fy1, fx2, fy2) in enumerate(state.far_zones):
            dx1, dy1 = _to_disp(fx1, fy1)
            dx2, dy2 = _to_disp(fx2, fy2)
            if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                state.far_zones.pop(i)
                redraw()
                schedule_save()
                return
        # Check mask zones
        for i, (x1, y1, x2, y2) in enumerate(state.mask_rects):
            dx1, dy1 = _to_disp(x1, y1)
            dx2, dy2 = _to_disp(x2, y2)
            if dx1 <= e.x <= dx2 and dy1 <= e.y <= dy2:
                state.mask_rects.pop(i)
                redraw()
                schedule_save()
                return

    # ── clear functions ──────────────────────────────────────────────

    def clear_masks():
        state.mask_rects.clear()
        redraw()
        schedule_save()

    def clear_far():
        state.far_zones.clear()
        redraw()
        schedule_save()

    def clear_motion():
        state.motion_zone.clear()
        redraw()
        schedule_save()

    # ── mode cycling ─────────────────────────────────────────────────

    _modes = ["mask", "far", "motion"]
    _mode_colors = {"mask": "#ff4444", "far": "#4488ff", "motion": "#00ff88"}
    _mode_labels = {"mask": "MODE: MASK ZONE", "far": "MODE: 15+ FT ZONE",
                    "motion": "MODE: MOTION ZONE"}
    _mode_bg     = {"mask": "#2a0000", "far": "#00112a", "motion": "#003322"}

    def toggle_mode():
        idx = (_modes.index(_mode["v"]) + 1) % len(_modes)
        _mode["v"] = _modes[idx]
        c = _mode_colors[_mode["v"]]
        mode_btn.config(text=_mode_labels[_mode["v"]], fg=c,
                        activeforeground=c, activebackground=_mode_bg[_mode["v"]])
        info_var.set(_info_text())

    # ── bindings ─────────────────────────────────────────────────────

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>",        on_right_click)

    # ── buttons ──────────────────────────────────────────────────────

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

    tk.Button(btn_row, text="✕  CLEAR MOTION", command=clear_motion,
              bg="#111111", fg="#00ff88", font=("Courier New", 9, "bold"),
              relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
              activebackground="#003322", activeforeground="#00ff88", bd=0
              ).pack(side=tk.LEFT, padx=(6, 0))

    tk.Button(btn_row, text="✓  DONE", command=win.destroy,
              bg="#111111", fg="#00ff88", font=("Courier New", 9, "bold"),
              relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
              activebackground="#003322", activeforeground="#00ff88", bd=0
              ).pack(side=tk.RIGHT)

    redraw()
    win.grab_set()
