"""
ollama_api.py — Thin wrappers around the Ollama HTTP API.
"""

import base64
import requests

from constants import OLLAMA_URL


def fmt_size(gb):
    """Format a size-in-GB value as a human-readable string."""
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def list_models(timeout=10):
    """
    Fetch installed model metadata from Ollama.
    Returns a dict: {name: {param_size, quantization, size_gb, family}}.
    Raises on HTTP errors.
    """
    resp = requests.get("http://localhost:11434/api/tags", timeout=timeout)
    resp.raise_for_status()
    raw = resp.json().get("models", [])
    out = {}
    for m in raw:
        name    = m.get("name", "")
        details = m.get("details", {})
        out[name] = {
            "param_size":   details.get("parameter_size", ""),
            "quantization": details.get("quantization_level", ""),
            "size_gb":      m.get("size", 0) / (1024 ** 3),
            "family":       details.get("family", ""),
        }
    return out


def warmup(model, timeout=60):
    """Send a tiny request to keep a model loaded in memory."""
    requests.post(OLLAMA_URL, json={
        "model": model, "prompt": "ready", "stream": False, "keep_alive": -1
    }, timeout=timeout)


def analyze_image_bytes(image_bytes, prompt, model, timeout=120):
    """Run a vision model with an image attached."""
    image_b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": -1,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("response", "No response received.")


def analyze_text(prompt, model, timeout=120):
    """Run a text-only model (no image)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("response", "No response received.")
