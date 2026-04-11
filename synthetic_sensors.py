"""
synthetic_sensors.py — Derived presence sensors based on vision model output.

Parses the vision description to detect:
  - child            (no HA gate)
  - jacob detected   (gated on input_boolean.jacob_is_home == on)
  - lauren detected  (gated on input_boolean.lauren_is_home == on)
  - emergency child alone  (child AND NOT (jacob OR lauren))

Each positive detection stays active for 5 minutes, then auto-clears.
UI label refs are wired after the HA panel is built via set_ui_refs().
"""

import threading

from constants import CHILD_KEYWORDS, JACOB_PATTERNS, LAUREN_PATTERNS


class SyntheticSensors:
    def __init__(self, root, ha_state, auto_off_seconds=300):
        self._root = root
        self._ha_state = ha_state
        self._auto_off = auto_off_seconds

        self._state = {"child": False, "jacob": False, "lauren": False}
        self._timers = {"child": None, "jacob": None, "lauren": None}

        # UI label refs (set via set_ui_refs)
        self.child_dot  = None; self.child_val  = None
        self.jacob_dot  = None; self.jacob_val  = None
        self.lauren_dot = None; self.lauren_val = None
        self.emerg_dot  = None; self.emerg_val  = None

    # ── UI wiring ────────────────────────────────────────────────────
    def set_ui_refs(self, *, child_dot, child_val,
                    jacob_dot, jacob_val,
                    lauren_dot, lauren_val,
                    emerg_dot, emerg_val):
        self.child_dot  = child_dot;  self.child_val  = child_val
        self.jacob_dot  = jacob_dot;  self.jacob_val  = jacob_val
        self.lauren_dot = lauren_dot; self.lauren_val = lauren_val
        self.emerg_dot  = emerg_dot;  self.emerg_val  = emerg_val

    def get_state(self):
        """Return (child, jacob, lauren, emergency) flags."""
        c = self._state["child"]
        j = self._state["jacob"]
        l = self._state["lauren"]
        e = c and not (j or l)
        return c, j, l, e

    # ── UI refresh ───────────────────────────────────────────────────
    @staticmethod
    def _apply_row(dot, val, active, text, alert=False):
        if dot is None:
            return
        if alert and active:
            dot.config(text="●", fg="#ff2222")
            val.config(text=text, fg="#ff2222")
        elif active:
            dot.config(text="●", fg="#00ff88")
            val.config(text=text, fg="#00ff88")
        else:
            dot.config(text="○", fg="#444444")
            val.config(text=text, fg="#666666")

    def _refresh_ui(self):
        """Recompute derived states and update all 4 rows. Main thread only."""
        child, jacob, lauren, emerg = self.get_state()
        self._apply_row(self.child_dot,  self.child_val,
                        child,  "detected" if child  else "clear")
        self._apply_row(self.jacob_dot,  self.jacob_val,
                        jacob,  "detected" if jacob  else "clear")
        self._apply_row(self.lauren_dot, self.lauren_val,
                        lauren, "detected" if lauren else "clear")
        self._apply_row(self.emerg_dot,  self.emerg_val,
                        emerg,  "ACTIVE"   if emerg  else "clear", alert=emerg)

    def _set(self, key, value):
        """Set one sensor flag and refresh UI. Must run on main thread."""
        self._state[key] = value
        self._refresh_ui()

    def _arm_timer(self, key):
        """Cancel existing timer for key and start a fresh auto-off timer."""
        old = self._timers.get(key)
        if old:
            old.cancel()
        t = threading.Timer(
            self._auto_off,
            lambda: self._root.after(0, lambda: self._set(key, False))
        )
        t.daemon = True
        t.start()
        self._timers[key] = t

    # ── Main entry — called from the queue worker ────────────────────
    def check(self, vision_result):
        """
        Parse vision model output and update synthetic sensors.
        Jacob/Lauren are gated on their respective HA home-presence sensor.
        Safe to call from any thread — UI updates marshalled via root.after.
        """
        text = vision_result.lower()

        # Child (no HA gate)
        if any(kw in text for kw in CHILD_KEYWORDS):
            self._arm_timer("child")
            self._root.after(0, lambda: self._set("child", True))

        # Jacob — gated on jacob_is_home=on
        if self._ha_state.get("input_boolean.jacob_is_home", "off") == "on":
            if any(p.search(text) for p in JACOB_PATTERNS):
                self._arm_timer("jacob")
                self._root.after(0, lambda: self._set("jacob", True))

        # Lauren — gated on lauren_is_home=on
        if self._ha_state.get("input_boolean.lauren_is_home", "off") == "on":
            if any(p.search(text) for p in LAUREN_PATTERNS):
                self._arm_timer("lauren")
                self._root.after(0, lambda: self._set("lauren", True))
