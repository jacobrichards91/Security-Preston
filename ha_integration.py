"""
ha_integration.py — Home Assistant websocket worker, entity state management,
and all HA configuration constants.

Exposes:
  HA_HOST, HA_TOKEN, WATCHED_ENTITIES, HA_NAMES, HA_GROUPS
  ha_state       — dict {entity_id: state_string}
  ha_row_labels  — dict {entity_id: {"dot": Label, "val": Label}}
  init(root, ha_status_var)  — call once after Tk root is created
  ha_worker()    — run in a daemon thread
  _ha_apply_entity(entity_id, state)  — update state dict + UI row
"""

import json
import time
import websocket

# ── Connection config ─────────────────────────────────────────────────────────
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

# ── Shared mutable state ──────────────────────────────────────────────────────
ha_state      = {}   # entity_id → state string (e.g. "on", "off")
ha_row_labels = {}   # entity_id → {"dot": tk.Label, "val": tk.Label}

# ── Module refs (set via init()) ──────────────────────────────────────────────
_root          = None
_ha_status_var = None


def init(root, ha_status_var):
    """Call once after the Tk root window and ha_status_var StringVar are created."""
    global _root, _ha_status_var
    _root          = root
    _ha_status_var = ha_status_var


# ── State helpers ─────────────────────────────────────────────────────────────

def _ha_is_active(state):
    return state in ("on", "open", "unlocked", "detected", "playing", "home", "true")


def _ha_apply_entity(entity_id, state):
    """Update the state dict and the matching UI row. Must run on main thread."""
    ha_state[entity_id] = state
    if entity_id not in ha_row_labels:
        return
    row    = ha_row_labels[entity_id]
    active = _ha_is_active(state)
    row["dot"].config(text="●" if active else "○",
                      fg="#00ff88" if active else "#444444")
    row["val"].config(text=state,
                      fg="#00ff88" if active else "#666666")


# ── Websocket worker ──────────────────────────────────────────────────────────

def ha_worker():
    """Connect to Home Assistant, authenticate, and stream state-change events.
    Designed to run as a daemon thread.  Retries indefinitely on failure."""
    while True:
        try:
            ws = websocket.create_connection(
                f"ws://{HA_HOST}:8123/api/websocket",
                timeout=15
            )
            _root.after(0, lambda: _ha_status_var.set("⟳  Authenticating..."))

            # 1. auth_required → send token
            msg = json.loads(ws.recv())
            assert msg.get("type") == "auth_required", \
                f"Expected auth_required, got {msg}"
            ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))

            # 2. auth_ok
            msg = json.loads(ws.recv())
            assert msg.get("type") == "auth_ok", f"Auth failed: {msg}"
            _root.after(0, lambda: _ha_status_var.set("⟳  Fetching states..."))

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
                    for s in msg.get("result", []):
                        eid   = s.get("entity_id", "")
                        if eid in WATCHED_ENTITIES:
                            state = s.get("state", "unknown")
                            _root.after(0, lambda e=eid, st=state:
                                        _ha_apply_entity(e, st))
                    _root.after(0, lambda: _ha_status_var.set("● Connected"))
                    print("[HA] Initial states loaded")

                elif msg.get("type") == "event":
                    edata = msg.get("event", {}).get("data", {})
                    eid   = edata.get("entity_id", "")
                    if eid in WATCHED_ENTITIES:
                        state = (edata.get("new_state") or {}).get("state", "unknown")
                        _root.after(0, lambda e=eid, st=state:
                                    _ha_apply_entity(e, st))
                        print(f"[HA] {eid} → {state}")

        except Exception as e:
            print(f"[HA] Disconnected: {e} — retrying in 10s")
            if _root and _ha_status_var:
                _root.after(0, lambda: _ha_status_var.set("○ Disconnected — retrying..."))
            try:
                ws.close()
            except Exception:
                pass
            time.sleep(10)
