"""
string_detector.py — Detection String manager.

A "string" is a cluster of rapid-fire detections (more than 1 per minute).
Each observation that arrives within 30s of the previous one extends the
active string.  A string auto-closes after 30s of silence or 2 min total.

When 2+ unique cameras contribute to a string, ADVANCED MODE activates:
continuous motion scanning on all cameras until the string closes.

The string judgment (text model) re-runs every time a new observation is
added, seeing ALL observations in the string.  Output goes only to the
Strings tab.
"""

import threading
import time
import json
from datetime import datetime


# ── Observation (one detection added to a string) ─────────────────────────────

def _make_observation(ts_str, cam_name, vision_result, distance, image_b64):
    return {
        "ts":            ts_str,
        "cam_name":      cam_name,
        "vision_result": vision_result,
        "distance":      distance,       # str or None
        "image_b64":     image_b64,
    }


# ── DetectionString (one cluster) ─────────────────────────────────────────────

class DetectionString:
    """One cluster of rapid-fire detections."""

    def __init__(self, start_ts_str):
        # Name like  String:4.12:19:35:23  (month.day:H:M:S)
        try:
            dt = datetime.strptime(start_ts_str, "%Y-%m-%d %H:%M:%S")
            self.name = f"String:{dt.month}.{dt.day}:{dt.strftime('%H:%M:%S')}"
        except Exception:
            self.name = f"String:{start_ts_str}"

        self.start_ts_str   = start_ts_str
        self.observations   = []            # list of observation dicts
        self.judgment_text  = ""            # latest AI judgment result
        self.closed         = False
        self.advanced_mode  = False         # True once 2+ unique cameras seen
        self._unique_cams   = set()
        self._last_obs_time = time.time()   # wall-clock of last added obs

    def add_observation(self, obs):
        self.observations.append(obs)
        self._last_obs_time = time.time()
        self._unique_cams.add(obs.get("cam_name", ""))

    @property
    def unique_camera_count(self):
        return len(self._unique_cams)

    def seconds_since_last(self):
        return time.time() - self._last_obs_time

    def total_duration(self):
        """Seconds between first and last observation."""
        if len(self.observations) < 2:
            return 0.0
        try:
            t0 = datetime.strptime(self.observations[0]["ts"],  "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(self.observations[-1]["ts"], "%Y-%m-%d %H:%M:%S")
            return (t1 - t0).total_seconds()
        except Exception:
            return 0.0

    def to_dict(self):
        return {
            "type":          "string",
            "name":          self.name,
            "start_ts":      self.start_ts_str,
            "observations":  self.observations,
            "judgment_text": self.judgment_text,
            "closed":        self.closed,
            "advanced_mode": self.advanced_mode,
        }


# ── StringManager ─────────────────────────────────────────────────────────────

STRING_IDLE_TIMEOUT  = 30    # close after 30s with no new observation
STRING_MAX_DURATION  = 120   # hard close after 2 minutes

class StringManager:
    """
    Tracks detection strings.  Thread-safe — called from the queue_worker
    thread and the main (UI) thread.

    Constructor args:
      root               — Tk root (for root.after scheduling)
      text_model_var     — tk.StringVar with current text model name
      string_prompt_text — tk.Text widget holding the string judgment prompt
      build_ha_context   — callable returning HA state string
      analyze_text_fn    — callable(prompt, model) → result string
      on_judgment_ready  — callable(string) on main thread after judgment
      on_string_opened   — callable(string) on main thread when new string starts
      on_string_closed   — callable(string) on main thread when string closes
      on_advanced_start  — callable(string) on main thread when advanced mode activates
      on_advanced_stop   — callable(string) on main thread when advanced mode deactivates
      chicago_tz         — timezone object
      save_fn            — callable(string_dict) to persist a closed string
    """

    def __init__(self, *, root, text_model_var, string_prompt_text,
                 build_ha_context, analyze_text_fn,
                 on_judgment_ready, on_string_opened, on_string_closed,
                 on_advanced_start, on_advanced_stop,
                 chicago_tz, save_fn):
        self._root              = root
        self._text_model_var    = text_model_var
        self._string_prompt_text = string_prompt_text
        self._build_ha          = build_ha_context
        self._analyze           = analyze_text_fn
        self._on_judgment       = on_judgment_ready
        self._on_opened         = on_string_opened
        self._on_closed         = on_string_closed
        self._on_adv_start      = on_advanced_start
        self._on_adv_stop       = on_advanced_stop
        self._tz                = chicago_tz
        self._save_fn           = save_fn

        self._lock              = threading.Lock()
        self._active            = None        # DetectionString | None
        self.history            = []          # list of closed DetectionString
        self._judgment_pending  = False       # judgment model currently running
        self._judgment_queued   = False       # another observation arrived while running
        self._close_timer       = None

    # ── Public: feed a completed detection ──────────────────────────────

    def feed(self, ts_str, cam_name, vision_result, distance, image_b64):
        """
        Called from queue_worker after each completed vision analysis.
        Decides whether to start / extend a string, then triggers judgment.
        """
        obs = _make_observation(ts_str, cam_name, vision_result, distance, image_b64)

        with self._lock:
            now = time.time()

            if self._active is not None and not self._active.closed:
                # Extend active string
                was_advanced = self._active.advanced_mode
                self._active.add_observation(obs)
                self._schedule_close_check()
                self._request_judgment()
                # Check if we just crossed into advanced mode (2+ unique cameras)
                if not was_advanced and self._active.unique_camera_count >= 2:
                    self._active.advanced_mode = True
                    s = self._active
                    self._root.after(0, lambda: self._on_adv_start(s))
                return

            # No active string — need 2 detections within idle timeout to start.
            if not hasattr(self, '_pending_obs') or self._pending_obs is None:
                self._pending_obs = (obs, now)
                return

            prev_obs, prev_time = self._pending_obs
            if now - prev_time <= STRING_IDLE_TIMEOUT:
                # Two detections within timeout — start a string!
                s = DetectionString(prev_obs["ts"])
                s.add_observation(prev_obs)
                s.add_observation(obs)
                self._active = s
                self._pending_obs = None
                self._schedule_close_check()
                self._root.after(0, lambda: self._on_opened(s))
                self._request_judgment()
                # Check if already 2 unique cameras at birth
                if s.unique_camera_count >= 2:
                    s.advanced_mode = True
                    self._root.after(0, lambda: self._on_adv_start(s))
            else:
                # Gap too long — replace pending with new obs
                self._pending_obs = (obs, now)

    # ── Close check ─────────────────────────────────────────────────────

    def _schedule_close_check(self):
        """Schedule an idle-timeout check on the main thread."""
        if self._close_timer is not None:
            self._root.after_cancel(self._close_timer)
        # Check every 3 seconds
        self._close_timer = self._root.after(3000, self._check_close)

    def _check_close(self):
        with self._lock:
            s = self._active
            if s is None or s.closed:
                return
            idle = s.seconds_since_last()
            total = s.total_duration()
            if idle >= STRING_IDLE_TIMEOUT or total >= STRING_MAX_DURATION:
                was_advanced = s.advanced_mode
                s.closed = True
                self.history.append(s)
                self._active = None
                self._save_fn(s.to_dict())
                self._root.after(0, lambda: self._on_closed(s))
                if was_advanced:
                    self._root.after(0, lambda: self._on_adv_stop(s))
                return
        # Still active — keep checking
        self._schedule_close_check()

    # ── Judgment worker ─────────────────────────────────────────────────

    def _request_judgment(self):
        """Request a judgment run.  If one is already running, queue another."""
        if self._judgment_pending:
            self._judgment_queued = True
            return
        self._judgment_pending = True
        threading.Thread(target=self._run_judgment, daemon=True).start()

    def _run_judgment(self):
        """Run the text model on the full string context. Worker thread."""
        while True:
            with self._lock:
                self._judgment_queued = False
                s = self._active
                if s is None or s.closed:
                    self._judgment_pending = False
                    return

            try:
                prompt = self._build_string_prompt(s)
                model = self._text_model_var.get()
                result = self._analyze(prompt, model)
                with self._lock:
                    if self._active is s:
                        s.judgment_text = result
                self._root.after(0, lambda: self._on_judgment(s))
            except Exception as e:
                print(f"[String] Judgment error: {e}")

            with self._lock:
                if self._judgment_queued:
                    # New obs arrived while we were running — go again
                    continue
                self._judgment_pending = False
                return

    def _build_string_prompt(self, s):
        """Assemble the full prompt sent to the text model for a string."""
        # Get the user's custom string prompt
        try:
            user_prompt = self._string_prompt_text.get("1.0", "end-1c").strip()
        except Exception:
            user_prompt = ""

        chicago_now = datetime.now(self._tz).strftime("%A %B %d %Y  %I:%M:%S %p %Z")

        obs_lines = []
        for o in s.observations:
            dist = f"  ({o['distance']})" if o.get("distance") else ""
            obs_lines.append(f"  {o['ts']}  [{o['cam_name']}]{dist}\n"
                             f"    {o['vision_result']}")

        ha_context = self._build_ha()

        mode_label = "ADVANCED MODE — " if s.advanced_mode else ""

        return (
            f"{user_prompt}\n\n"
            f"Time: {chicago_now}\n\n"
            f"Home state:\n{ha_context}\n\n"
            f"{mode_label}STRING ANALYSIS — {len(s.observations)} detections "
            f"over {s.total_duration():.0f} seconds\n\n"
            + "\n\n".join(obs_lines)
        )

    # ── Accessors ───────────────────────────────────────────────────────

    def get_active(self):
        with self._lock:
            return self._active

    def get_active_observations(self):
        with self._lock:
            if self._active and not self._active.closed:
                return list(self._active.observations)
            return []
