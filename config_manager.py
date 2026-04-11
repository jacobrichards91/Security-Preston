"""
config_manager.py — Configuration persistence for Security Vision.

Saves and loads all tunable settings (per-camera and global) to/from a JSON
file in the user's Documents/GitHub folder.

Call init(...) once after the Tk root and all UI variables are created.
Then call schedule_save() / load_config() as needed.

Config file format (v2 — multi-camera):
{
  "vision_model":  str,
  "text_model":    str,
  "vision_prompt": str,
  "text_prompt":   str,
  "system_active": bool,
  "cameras": [
    {
      "name":        str,
      "rtsp":        str,
      "cam_id":      str,
      "snap_before": float,
      "snap_after":  float,
      "buffer_secs": float,
      "crop_padding":int,
      "min_box_pct": float,
      "mask_rects":  [[x1,y1,x2,y2], ...],
      "far_zone":    [x1,y1,x2,y2] | null
    },
    ...
  ]
}
"""

import json
from pathlib import Path

import ai_engine   # for DEFAULT_VISION_PROMPT / DEFAULT_TEXT_PROMPT

# ── Config file path ──────────────────────────────────────────────────────────
CONFIG_PATH = Path.home() / "Documents" / "GitHub" / "security_preston_config.json"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Module refs (set via init()) ──────────────────────────────────────────────
_root              = None
_camera_tabs       = None   # list of CameraTab instances (live reference)
_vision_model_var  = None
_text_model_var    = None
_system_active_var = None
_app_refs          = None   # shared dict; keys: vision_prompt_text, text_prompt_text

_save_job = None


def init(root, camera_tabs, vision_model_var, text_model_var,
         system_active_var, app_refs):
    """Call once after root and all tkinter vars are created."""
    global _root, _camera_tabs, _vision_model_var, _text_model_var
    global _system_active_var, _app_refs
    _root              = root
    _camera_tabs       = camera_tabs
    _vision_model_var  = vision_model_var
    _text_model_var    = text_model_var
    _system_active_var = system_active_var
    _app_refs          = app_refs


# ── Save ──────────────────────────────────────────────────────────────────────

def save_config(*_):
    """Write all tunable settings to the config JSON file."""
    try:
        vpt = _app_refs.get("vision_prompt_text") if _app_refs else None
        tpt = _app_refs.get("text_prompt_text")   if _app_refs else None

        cam_data = []
        for cam in (_camera_tabs or []):
            cam_data.append({
                "name":         cam.cam_name_var.get(),
                "rtsp":         cam.rtsp_url_var.get(),
                "cam_id":       cam.cam_id_var.get(),
                "snap_before":  cam.snap_before_var.get(),
                "snap_after":   cam.snap_after_var.get(),
                "buffer_secs":  cam.buffer_secs_var.get(),
                "crop_padding": cam.crop_padding_var.get(),
                "min_box_pct":  cam.min_box_pct_var.get(),
                "mask_rects":   [list(r) for r in cam.mask_rects],
                "far_zone":     list(cam.far_zone) if cam.far_zone else None,
            })

        data = {
            "vision_model":  _vision_model_var.get()  if _vision_model_var  else "",
            "text_model":    _text_model_var.get()    if _text_model_var    else "",
            "vision_prompt": (vpt.get("1.0", "end").rstrip("\n")
                              if vpt else ai_engine.DEFAULT_VISION_PROMPT),
            "text_prompt":   (tpt.get("1.0", "end").rstrip("\n")
                              if tpt else ai_engine.DEFAULT_TEXT_PROMPT),
            "system_active": _system_active_var.get() if _system_active_var else True,
            "cameras":       cam_data,
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[Config] Save error: {e}")


def schedule_save(*_):
    """Debounce saves — write 400 ms after the last change."""
    global _save_job
    if _save_job:
        _root.after_cancel(_save_job)
    _save_job = _root.after(400, save_config)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_config(add_camera_tab_fn):
    """Read the config file and apply all stored values.

    add_camera_tab_fn(name) is called to create additional camera tabs beyond
    the first one that already exists when the app starts.
    """
    if not CONFIG_PATH.exists():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text())

        if "vision_model" in data and _vision_model_var:
            _vision_model_var.set(data["vision_model"])
        if "text_model" in data and _text_model_var:
            _text_model_var.set(data["text_model"])

        vpt = _app_refs.get("vision_prompt_text") if _app_refs else None
        tpt = _app_refs.get("text_prompt_text")   if _app_refs else None
        if vpt and "vision_prompt" in data:
            vpt.delete("1.0", "end")
            vpt.insert("1.0", data["vision_prompt"])
        if tpt and "text_prompt" in data:
            tpt.delete("1.0", "end")
            tpt.insert("1.0", data["text_prompt"])

        if "system_active" in data and _system_active_var:
            _system_active_var.set(data["system_active"])

        # ── Multi-camera restore ──────────────────────────────────────────
        if "cameras" in data:
            saved = data["cameras"]
            for i, cd in enumerate(saved):
                if i < len(_camera_tabs):
                    cam = _camera_tabs[i]
                else:
                    cam = add_camera_tab_fn(cd.get("name", f"Camera {i+1}"))
                _apply_camera_data(cam, cd)

        else:
            # Legacy single-camera format (no "cameras" key)
            if _camera_tabs:
                cam = _camera_tabs[0]
                legacy = {
                    "name":         data.get("cam_name",     "Front Door"),
                    "rtsp":         data.get("rtsp_url",     ""),
                    "cam_id":       data.get("cam_id",       ""),
                    "snap_before":  data.get("snap_before",  2.5),
                    "snap_after":   data.get("snap_after",   2.0),
                    "buffer_secs":  data.get("buffer_secs",  3.0),
                    "crop_padding": data.get("crop_padding", 50),
                    "min_box_pct":  data.get("min_box_pct",  0.05),
                    "mask_rects":   data.get("mask_rects",   []),
                    "far_zone":     data.get("far_zone",     None),
                }
                _apply_camera_data(cam, legacy)

        print(f"[Config] Loaded from {CONFIG_PATH}")
    except Exception as e:
        print(f"[Config] Load error: {e}")


def _apply_camera_data(cam, cd):
    """Apply a single camera config dict to a CameraTab instance."""
    cam.cam_name_var.set(   cd.get("name",         cam.cam_name_var.get()))
    cam.rtsp_url_var.set(   cd.get("rtsp",         ""))
    cam.cam_id_var.set(     cd.get("cam_id",       ""))
    cam.snap_before_var.set( cd.get("snap_before",  2.5))
    cam.snap_after_var.set(  cd.get("snap_after",   2.0))
    cam.buffer_secs_var.set( cd.get("buffer_secs",  3.0))
    cam.crop_padding_var.set(cd.get("crop_padding", 50))
    cam.min_box_pct_var.set( cd.get("min_box_pct",  0.05))
    cam.mask_rects.clear()
    cam.mask_rects.extend(tuple(r) for r in cd.get("mask_rects", []))
    fz = cd.get("far_zone")
    cam.far_zone = tuple(fz) if fz else None
