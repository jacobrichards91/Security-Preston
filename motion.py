"""
motion.py — Pure image processing: mask application and motion detection / crop.
All functions are stateless; callers pass in mask rectangles and tuning values.
"""

import io
import numpy as np
import cv2
from PIL import Image


def apply_mask(img_bgr, mask_rects):
    """Paint black over every rect in mask_rects. Returns a copy."""
    if not mask_rects:
        return img_bgr
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in mask_rects:
        out[y1:y2, x1:x2] = 0
    return out


def jpeg_apply_mask(jpeg_bytes, mask_rects):
    """Decode JPEG bytes → apply mask → re-encode. Returns bytes."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    img = apply_mask(img, mask_rects)
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return enc.tobytes()


def compute_motion_crop(frame_a_bytes, frame_b_bytes, min_box_pct, crop_padding):
    """
    Diff two JPEG frames. Returns:
      - cropped_bytes: JPEG of the motion crop from frame_b (or None)
      - debug_images:  dict of labeled PIL images for debug view
      - bbox:          (x1, y1, x2, y2) or None

    min_box_pct  — minimum contour size as percent of frame area (e.g. 0.05)
    crop_padding — pixels to pad around the motion bounding box
    """
    # Decode to numpy
    arr_a = np.frombuffer(frame_a_bytes, dtype=np.uint8)
    arr_b = np.frombuffer(frame_b_bytes, dtype=np.uint8)
    img_a = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        return None, {}, None

    # Resize to same size if different (shouldn't happen but safety)
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    h, w = img_a.shape[:2]

    # Grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur_a = cv2.GaussianBlur(gray_a, (21, 21), 0)
    blur_b = cv2.GaussianBlur(gray_b, (21, 21), 0)

    # Absolute diff
    diff = cv2.absdiff(blur_a, blur_b)

    # Threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small noise contours
    min_area = w * h * (min_box_pct / 100.0)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    bbox = None
    cropped_bytes = None

    if contours:
        # Bounding box enclosing all motion contours
        x1 = min(cv2.boundingRect(c)[0] for c in contours)
        y1 = min(cv2.boundingRect(c)[1] for c in contours)
        x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

        # Add padding, clamp to frame
        pad = int(crop_padding)
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)
        y2p = min(h, y2 + pad)
        bbox = (x1p, y1p, x2p, y2p)

        # Crop from frame_b
        crop = img_b[y1p:y2p, x1p:x2p]
        _, crop_enc = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cropped_bytes = crop_enc.tobytes()

    # --- Debug images ---
    debug = {}

    def cv2_to_pil(img, gray=False):
        if gray:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    debug["Frame A\n(t-1s)"]   = cv2_to_pil(img_a)
    debug["Frame B\n(t+0.5s)"] = cv2_to_pil(img_b)
    debug["Diff"]              = cv2_to_pil(diff,    gray=True)
    debug["Threshold"]         = cv2_to_pil(thresh,  gray=True)
    debug["Dilated\nmask"]     = cv2_to_pil(dilated, gray=True)

    # Draw bounding box on frame B copy
    img_b_annot = img_b.copy()
    if bbox:
        x1p, y1p, x2p, y2p = bbox
        cv2.rectangle(img_b_annot, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)
        cv2.drawContours(img_b_annot, contours, -1, (0, 0, 255), 2)
    debug["Bounding\nbox"] = cv2_to_pil(img_b_annot)

    if cropped_bytes:
        crop_pil = Image.open(io.BytesIO(cropped_bytes))
        debug["AI\nCrop"] = crop_pil

    return cropped_bytes, debug, bbox


def compute_distance(bbox, far_zones):
    """
    Classify a motion bounding box by distance relative to configured far_zones.

    Returns:
      "more than 15 feet from house"  — bbox is entirely inside any far zone
      "closer than 15 feet to house"  — bbox is not inside any far zone
      None                            — no far zones configured or no bbox
    """
    if not far_zones or bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    for fz in far_zones:
        fx1, fy1, fx2, fy2 = fz
        if x1 >= fx1 and y1 >= fy1 and x2 <= fx2 and y2 <= fy2:
            return "more than 15 feet from house"
    return "closer than 15 feet to house"


def compute_motion_in_zone(frame_a_bytes, frame_b_bytes, polygon_points,
                           threshold_pct, crop_padding=50):
    """
    Detect motion within a polygon-defined zone.

    Args:
      polygon_points — list of (x, y) tuples in native resolution
      threshold_pct  — minimum motion as % of polygon area to trigger

    Returns:
      (motion_pct, cropped_bytes, diff_b64, bbox)
      motion_pct   — float, percent of polygon area with motion
      cropped_bytes — JPEG bytes of the motion crop (or None)
      diff_b64     — base64 string of the thresholded diff image (for debug)
      bbox         — (x1, y1, x2, y2) or None
    """
    import base64 as _b64

    arr_a = np.frombuffer(frame_a_bytes, dtype=np.uint8)
    arr_b = np.frombuffer(frame_b_bytes, dtype=np.uint8)
    img_a = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)
    if img_a is None or img_b is None:
        return 0.0, None, None, None
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    h, w = img_a.shape[:2]

    # Create polygon mask
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [pts], 255)
    zone_area = cv2.countNonZero(zone_mask)
    if zone_area == 0:
        return 0.0, None, None, None

    # Motion detection
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    blur_a = cv2.GaussianBlur(gray_a, (21, 21), 0)
    blur_b = cv2.GaussianBlur(gray_b, (21, 21), 0)
    diff = cv2.absdiff(blur_a, blur_b)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Mask to polygon zone only
    zone_motion = cv2.bitwise_and(dilated, zone_mask)
    motion_pixels = cv2.countNonZero(zone_motion)
    motion_pct = (motion_pixels / zone_area) * 100.0

    # Encode diff image for display
    diff_small = cv2.resize(zone_motion, (320, 180))
    _, diff_enc = cv2.imencode('.jpg', diff_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    diff_b64 = _b64.b64encode(diff_enc.tobytes()).decode()

    cropped_bytes = None
    bbox = None

    if motion_pct >= threshold_pct:
        # Find bounding box of motion within the zone
        contours, _ = cv2.findContours(zone_motion, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x1 = min(cv2.boundingRect(c)[0] for c in contours)
            y1 = min(cv2.boundingRect(c)[1] for c in contours)
            x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
            y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
            pad = int(crop_padding)
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
            bbox = (x1p, y1p, x2p, y2p)
            crop = img_b[y1p:y2p, x1p:x2p]
            _, crop_enc = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cropped_bytes = crop_enc.tobytes()

    return motion_pct, cropped_bytes, diff_b64, bbox
