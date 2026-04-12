"""
constants.py — Static configuration, prompts, regex patterns, and HA entity maps.
No mutable state and no tkinter/UI dependencies.
"""

import os
import re
from pathlib import Path
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo


# --- Ollama ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_VISION_MODEL = "minicpm-v:latest"
DEFAULT_TEXT_MODEL   = "minicpm-v:latest"

# --- Time zone (Windows needs tzdata: pip install tzdata) ---
try:
    CHICAGO_TZ = ZoneInfo("America/Chicago")
except Exception:
    print("[Warning] tzdata not installed — run: pip install tzdata")
    print("[Warning] Falling back to UTC-5 (CDT). Install tzdata for correct DST handling.")
    CHICAGO_TZ = timezone(timedelta(hours=-5))

# --- Paths & network ---
WEBHOOK_PORT = 8765
SAVE_DIR = Path(os.path.expanduser("~")) / "SecurityEvents"
SAVE_DIR.mkdir(exist_ok=True)
CONFIG_PATH = Path.home() / "Documents" / "GitHub" / "security_preston_config.json"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

RTSP_URL = "rtsp://192.168.0.166:7447/YD4arutidcyKjQvI"
STREAM_PREVIEW_INTERVAL = 3000  # ms between live preview refreshes

# --- Buffer & motion defaults ---
BUFFER_SECONDS   = 3.0   # rolling buffer window
FRAME_INTERVAL   = 0.2   # grab a frame every N seconds
SNAP_BEFORE_SECS = 2.5   # frame A offset
SNAP_AFTER_SECS  = 2.0   # frame B offset
MIN_BOX_PCT      = 0.05  # minimum contour size as % of frame area
CROP_PADDING     = 50    # px padding around bounding box

DEBUG_MODE = True  # show debug windows

# --- Prompts ---
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

DEFAULT_STRING_PROMPT = """You are a home security analyst reviewing a CLUSTER of rapid-fire detections across multiple cameras.

You will receive:
1. The current date and time (Chicago)
2. The live state of all sensors, doors, locks, and occupancy in the home
3. A timeline of visual observations from multiple cameras, each with timestamps and distance info

Analyze the MOVEMENT PATTERN across cameras:
- Is the same person/vehicle appearing across multiple cameras?
- Are they approaching or leaving the house?
- What direction are they moving?
- Is this behavior normal (resident, delivery, neighbor) or suspicious?

Be concise and direct. Focus on the cross-camera story, not individual frames."""


# ---------------------------------------------------------------
# SYNTHETIC SENSOR KEYWORDS / PATTERNS
# ---------------------------------------------------------------
CHILD_KEYWORDS = [
    "child", "children", "baby", "babies", "toddler", "toddlers",
    "infant", "infants", "kid ", "kids ", "young child", "small child",
    "little one", "little ones", "youngster", "youngsters", "minor",
]

# Jacob: man with brown / black / dark hair (gated on jacob_is_home)
JACOB_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
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

# Lauren: woman with brown/dark hair OR woman aged 20-35 (gated on lauren_is_home)
LAUREN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    # Brown / dark hair
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
    # Age 20-35
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
    # Numeric age 20-35
    r"(woman|female|lady).{0,40}(approximately|about|around|age\s+)?(2[0-9]|3[0-5])\s*(year|yr|y\.?o)",
    r"(approximately|about|around|age\s+)?(2[0-9]|3[0-5])\s*(year|yr|y\.?o).{0,40}(woman|female|lady)",
    r"(woman|female).{0,40}between\s+\d+\s+and\s+3[0-5]",
    r"(woman|female).{0,40}appears\s+to\s+be\s+(in\s+her\s+)?(20|25|30|35)",
]]


# ---------------------------------------------------------------
# HOME ASSISTANT CONFIG
# ---------------------------------------------------------------

# Person-detected sensors — one per physical camera in Home Assistant.
# Used to populate the HA sensor dropdown in each camera tab.
PERSON_DETECTED_SENSORS = [
    ("", "— none —"),
    ("binary_sensor.front_person_detected",                "front"),
    ("binary_sensor.front_door_person_detected",           "front door"),
    ("binary_sensor.side_henrys_room_person_detected",     "henry's room"),
    ("binary_sensor.side_yard_cul_de_sac_person_detected", "cul-de-sac"),
    ("binary_sensor.side_yard_street_person_detected",     "street"),
    ("binary_sensor.patio2_person_detected",               "patio"),
    ("binary_sensor.backyard_person_detected",             "backyard"),
]

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
