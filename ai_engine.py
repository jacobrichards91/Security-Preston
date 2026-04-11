"""
ai_engine.py — Ollama AI analysis, synthetic sensor detection, and prompt defaults.

Imports ha_integration for ha_state and HA_GROUPS/HA_NAMES (no circular dependency).

Exposes:
  OLLAMA_URL
  DEFAULT_VISION_PROMPT, DEFAULT_TEXT_PROMPT
  analyze_image_bytes(image_bytes, prompt, model) → str
  analyze_text(prompt, model) → str
  build_ha_context() → str
  check_synthetic_sensors(vision_result)
  init(root)          — call once after Tk root is created
  set_synth_labels(child_dot, child_val, jacob_dot, jacob_val,
                   lauren_dot, lauren_val, emerg_dot, emerg_val)
                      — call after the HA panel UI is built
"""

import re
import requests
import threading

import ha_integration

# ── Ollama endpoint ───────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"

# ── Default prompts ───────────────────────────────────────────────────────────
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

# ── Synthetic sensor state ────────────────────────────────────────────────────
_synth       = {"child": False, "jacob": False, "lauren": False}
_child_timer  = None
_jacob_timer  = None
_lauren_timer = None

# Synth sensor UI label refs — set via set_synth_labels() after UI is built
synth_child_dot  = None;  synth_child_val  = None
synth_jacob_dot  = None;  synth_jacob_val  = None
synth_lauren_dot = None;  synth_lauren_val = None
synth_emerg_dot  = None;  synth_emerg_val  = None

# ── Module ref ────────────────────────────────────────────────────────────────
_root = None


def init(root):
    """Call once after the Tk root window is created."""
    global _root
    _root = root


def set_synth_labels(child_dot, child_val, jacob_dot, jacob_val,
                     lauren_dot, lauren_val, emerg_dot, emerg_val):
    """Wire up the synthetic-sensor label widgets after the HA panel is built."""
    global synth_child_dot, synth_child_val
    global synth_jacob_dot, synth_jacob_val
    global synth_lauren_dot, synth_lauren_val
    global synth_emerg_dot, synth_emerg_val
    synth_child_dot   = child_dot;   synth_child_val   = child_val
    synth_jacob_dot   = jacob_dot;   synth_jacob_val   = jacob_val
    synth_lauren_dot  = lauren_dot;  synth_lauren_val  = lauren_val
    synth_emerg_dot   = emerg_dot;   synth_emerg_val   = emerg_val


# ── Synthetic sensor keyword / pattern lists ──────────────────────────────────
_CHILD_KEYWORDS = [
    "child", "children", "baby", "babies", "toddler", "toddlers",
    "infant", "infants", "kid ", "kids ", "young child", "small child",
    "little one", "little ones", "youngster", "youngsters", "minor",
]

_JACOB_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"man.{0,50}brown\s*hair",                 r"brown\s*hair.{0,50}man",
    r"man.{0,50}black\s*hair",                 r"black\s*hair.{0,50}man",
    r"man.{0,50}dark\s*hair",                  r"dark\s*hair.{0,50}man",
    r"male.{0,50}brown\s*hair",                r"brown\s*hair.{0,50}male",
    r"male.{0,50}black\s*hair",                r"black\s*hair.{0,50}male",
    r"male.{0,50}dark\s*hair",                 r"dark\s*hair.{0,50}male",
    r"guy.{0,50}(brown|black|dark)\s*hair",    r"(brown|black|dark)\s*hair.{0,50}guy",
    r"brown[\-\s]haired\s+\w*\s*(man|male|guy|gentleman)",
    r"black[\-\s]haired\s+\w*\s*(man|male|guy|gentleman)",
    r"dark[\-\s]haired\s+\w*\s*(man|male|guy|gentleman)",
    r"(man|male|guy)\s+\w*\s*brown[\-\s]hair",
    r"(man|male|guy)\s+\w*\s*black[\-\s]hair",
    r"(man|male|guy)\s+\w*\s*dark[\-\s]hair",
    r"adult\s+male.{0,50}(brown|black|dark)\s*hair",
    r"(brown|black|dark)\s*hair.{0,50}adult\s+male",
    r"man\s+with\s+(short|long|medium|curly|straight|wavy)?\s*(brown|black|dark)\s*hair",
    r"(brown|black|dark)[\-\s]haired\s+adult",
    r"individual.{0,30}man.{0,30}(brown|black|dark)\s*hair",
    r"person.{0,20}appears\s+to\s+be\s+(a\s+)?male.{0,50}(brown|black|dark)\s*hair",
]]

_LAUREN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"woman.{0,50}brown\s*hair",               r"brown\s*hair.{0,50}woman",
    r"female.{0,50}brown\s*hair",              r"brown\s*hair.{0,50}female",
    r"lady.{0,50}brown\s*hair",                r"brown\s*hair.{0,50}lady",
    r"girl.{0,50}brown\s*hair",                r"brown\s*hair.{0,50}girl",
    r"woman.{0,50}dark\s*hair",                r"dark\s*hair.{0,50}woman",
    r"female.{0,50}dark\s*hair",               r"dark\s*hair.{0,50}female",
    r"brunette",
    r"brown[\-\s]haired\s+\w*\s*(woman|female|lady|girl)",
    r"dark[\-\s]haired\s+\w*\s*(woman|female|lady|girl)",
    r"(woman|female|lady|girl)\s+\w*\s*brown[\-\s]hair",
    r"adult\s+female.{0,50}(brown|dark)\s*hair",
    r"(brown|dark)\s*hair.{0,50}adult\s+female",
    r"woman\s+with\s+(short|long|medium|curly|straight|wavy)?\s*(brown|dark)\s*hair",
    r"woman.{0,40}(in her |aged? )?(20s|twenties|30s|thirties)",
    r"(20s|twenties|30s|thirties).{0,40}woman",
    r"female.{0,40}(in her |aged? )?(20s|twenties|30s|thirties)",
    r"(20s|twenties|30s|thirties).{0,40}female",
    r"(woman|female|lady).{0,30}mid[\-\s]?(twenties|20s)",
    r"(woman|female|lady).{0,30}late[\-\s]?(twenties|20s)",
    r"(woman|female|lady).{0,30}early[\-\s]?(thirties|30s)",
    r"(woman|female|lady).{0,30}mid[\-\s]?(thirties|30s)",
    r"20[\-\s]?something.{0,30}(woman|female|lady)",
    r"30[\-\s]?something.{0,30}(woman|female|lady)",
    r"(woman|female|lady).{0,30}20[\-\s]?something",
    r"(woman|female|lady).{0,30}30[\-\s]?something",
    r"young\s+adult\s+(woman|female|lady)",
    r"(woman|female|lady).{0,20}young\s+adult",
    r"(woman|female|lady).{0,40}(approximately|about|around|age\s+)?(2[0-9]|3[0-5])\s*(year|yr|y\.?o)",
    r"(approximately|about|around|age\s+)?(2[0-9]|3[0-5])\s*(year|yr|y\.?o).{0,40}(woman|female|lady)",
    r"(woman|female).{0,40}between\s+\d+\s+and\s+3[0-5]",
    r"(woman|female).{0,40}appears\s+to\s+be\s+(in\s+her\s+)?(20|25|30|35)",
]]


# ── Synthetic sensor UI helpers ───────────────────────────────────────────────

def _apply_synth_row(dot, val, active, text, alert=False):
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


def _refresh_synth_ui():
    """Recompute derived states and repaint all four synthetic sensor rows.
    Must be called on the main thread."""
    child  = _synth["child"]
    jacob  = _synth["jacob"]
    lauren = _synth["lauren"]
    emerg  = child and not (jacob or lauren)
    _apply_synth_row(synth_child_dot,  synth_child_val,
                     child,  "detected" if child  else "clear")
    _apply_synth_row(synth_jacob_dot,  synth_jacob_val,
                     jacob,  "detected" if jacob  else "clear")
    _apply_synth_row(synth_lauren_dot, synth_lauren_val,
                     lauren, "detected" if lauren else "clear")
    _apply_synth_row(synth_emerg_dot,  synth_emerg_val,
                     emerg,  "ACTIVE"   if emerg  else "clear", alert=emerg)


def _set_synth(key, value):
    """Set one synthetic sensor flag and refresh UI. Must run on main thread."""
    _synth[key] = value
    _refresh_synth_ui()


def _arm_timer(key, timer_attr):
    """Cancel the existing auto-off timer for key and start a fresh 5-minute one."""
    global _child_timer, _jacob_timer, _lauren_timer
    timers = {"child": "_child_timer", "jacob": "_jacob_timer", "lauren": "_lauren_timer"}
    attr   = timers[key]
    old    = globals().get(attr)
    if old:
        old.cancel()
    t = threading.Timer(300, lambda: _root.after(0, lambda: _set_synth(key, False)))
    t.daemon = True
    t.start()
    globals()[attr] = t


def check_synthetic_sensors(vision_result):
    """Parse vision model output and update synthetic sensors.
    Safe to call from any thread — UI updates are marshalled via root.after."""
    text = vision_result.lower()

    # Child (no HA gate)
    if any(kw in text for kw in _CHILD_KEYWORDS):
        _arm_timer("child", "_child_timer")
        _root.after(0, lambda: _set_synth("child", True))

    # Jacob: man with brown/black/dark hair, gated on jacob_is_home
    if ha_integration.ha_state.get("input_boolean.jacob_is_home", "off") == "on":
        if any(p.search(text) for p in _JACOB_PATTERNS):
            _arm_timer("jacob", "_jacob_timer")
            _root.after(0, lambda: _set_synth("jacob", True))

    # Lauren: woman (brown hair OR aged 20-35), gated on lauren_is_home
    if ha_integration.ha_state.get("input_boolean.lauren_is_home", "off") == "on":
        if any(p.search(text) for p in _LAUREN_PATTERNS):
            _arm_timer("lauren", "_lauren_timer")
            _root.after(0, lambda: _set_synth("lauren", True))


# ── Ollama analysis ───────────────────────────────────────────────────────────

def analyze_image_bytes(image_bytes, prompt, model):
    """Run a vision model with an image attached. Returns the response string."""
    import base64
    image_b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model":      model,
        "prompt":     prompt,
        "images":     [image_b64],
        "stream":     False,
        "keep_alive": -1,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "No response received.")


def analyze_text(prompt, model):
    """Run a text-only model (no image). Returns the response string."""
    payload = {
        "model":      model,
        "prompt":     prompt,
        "stream":     False,
        "keep_alive": -1,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "No response received.")


def build_ha_context():
    """Format current HA state into a readable string for the text model."""
    lines = []
    child  = _synth["child"]
    jacob  = _synth["jacob"]
    lauren = _synth["lauren"]
    emerg  = child and not (jacob or lauren)
    lines.append(
        f"SYNTHETIC SENSORS: "
        f"child_detected={'on' if child else 'off'}, "
        f"jacob_detected={'on' if jacob else 'off'}, "
        f"lauren_detected={'on' if lauren else 'off'}, "
        f"emergency_child_alone={'on' if emerg else 'off'}"
    )
    for group_name, entities in ha_integration.HA_GROUPS:
        parts = [
            f"{ha_integration.HA_NAMES.get(e, e)}="
            f"{ha_integration.ha_state.get(e, 'unknown')}"
            for e in entities
        ]
        lines.append(f"{group_name}: {', '.join(parts)}")
    return "\n".join(lines)
