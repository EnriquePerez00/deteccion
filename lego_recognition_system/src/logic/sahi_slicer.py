"""
🔪 SAHI Slicer — Slicing Aided Hyper Inference for LEGO Detection
=================================================================
Optimised for 24MP iPhone 16 images (5712×4284) captured at 70cm
zenithal over a 50×50cm surface.

Key parameters derived from physical setup:
  - Training density: 40 px/cm (800px render / 20cm surface)
  - Capture density: ~114 px/cm (5712px / 50cm)
  - Optimal tile: 1824×1824px (covers ~16cm, matches training density when YOLO resizes to 640)
"""

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────
# 1. SLICE GENERATOR
# ─────────────────────────────────────────────────────────────────────
def generate_slices(image, slice_size=1024, overlap_ratio=0.30):
    """
    Splits a PIL image into overlapping tiles.

    Args:
        image: PIL Image (RGB).
        slice_size: Tile side in pixels (default 1024).
        overlap_ratio: Fractional overlap between adjacent tiles (default 0.20).

    Returns:
        List of dicts: [{'tile': PIL.Image, 'x': int, 'y': int, 'w': int, 'h': int}, ...]
        where (x, y) is the top-left offset and (w, h) is the tile size.
    """
    img_w, img_h = image.size
    stride = int(slice_size * (1.0 - overlap_ratio))

    slices = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            # Clamp to image boundaries
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            # Ensure minimum tile size (at least 50% of slice_size)
            tile_w = x2 - x
            tile_h = y2 - y
            if tile_w < slice_size * 0.5 or tile_h < slice_size * 0.5:
                # Shift origin backwards so the tile is full-sized
                x = max(0, img_w - slice_size) if tile_w < slice_size * 0.5 else x
                y = max(0, img_h - slice_size) if tile_h < slice_size * 0.5 else y
                x2 = min(x + slice_size, img_w)
                y2 = min(y + slice_size, img_h)
                tile_w = x2 - x
                tile_h = y2 - y

            tile = image.crop((x, y, x2, y2))
            slices.append({
                'tile': tile,
                'x': x,
                'y': y,
                'w': tile_w,
                'h': tile_h,
            })

            if x2 >= img_w:
                break
            x += stride

        if y2 >= img_h:
            break
        y += stride

    return slices


# ─────────────────────────────────────────────────────────────────────
# 2. DETECTION REMAPPER
# ─────────────────────────────────────────────────────────────────────
def remap_detections(yolo_result, slice_info, center_bonus=0.05):
    """
    Converts YOLO detections from tile-local coords to global image coords.
    Applies a confidence bonus to detections whose centre falls in the
    non-overlapping core of the tile.

    Args:
        yolo_result: Ultralytics Results object for one tile.
        slice_info: Dict with 'x', 'y', 'w', 'h' of the tile origin.
        center_bonus: Extra confidence for detections centred in the tile core.

    Returns:
        List of dicts:
          [{'box': [x1,y1,x2,y2], 'conf': float, 'polygon': np.array|None}, ...]
    """
    if yolo_result is None or yolo_result.boxes is None:
        return []

    ox, oy = slice_info['x'], slice_info['y']
    tw, th = slice_info['w'], slice_info['h']
    margin = 0.15  # 15% from each edge is "overlap zone"

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    confs = yolo_result.boxes.conf.cpu().numpy()

    has_masks = hasattr(yolo_result, 'masks') and yolo_result.masks is not None
    polygons = yolo_result.masks.xy if has_masks else [None] * len(boxes)

    detections = []
    for i, box in enumerate(boxes):
        # Remap box to global coordinates
        gx1 = box[0] + ox
        gy1 = box[1] + oy
        gx2 = box[2] + ox
        gy2 = box[3] + oy

        # Centre of detection in tile-local space
        cx_local = (box[0] + box[2]) / 2
        cy_local = (box[1] + box[3]) / 2

        # Is the centre in the "core" (non-overlap) zone?
        in_core = (
            cx_local > tw * margin and cx_local < tw * (1 - margin) and
            cy_local > th * margin and cy_local < th * (1 - margin)
        )

        conf = float(confs[i])
        if in_core:
            conf = min(1.0, conf + center_bonus)

        # Remap polygon to global coordinates
        poly_global = None
        if has_masks and polygons[i] is not None and len(polygons[i]) > 0:
            poly_global = np.array(polygons[i]) + np.array([ox, oy])

        detections.append({
            'box': [float(gx1), float(gy1), float(gx2), float(gy2)],
            'conf': conf,
            'polygon': poly_global,
        })

    return detections


# ─────────────────────────────────────────────────────────────────────
# 3. GLOBAL NMS (Non-Maximum Suppression)
# ─────────────────────────────────────────────────────────────────────
def _compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def global_nms(all_detections, iou_threshold=0.5):
    """
    Applies confidence-weighted NMS across all detections from all tiles.
    Higher-confidence detections suppress overlapping lower-confidence ones.

    Args:
        all_detections: List of dicts from remap_detections (across all tiles).
        iou_threshold: IoU above which the weaker detection is suppressed.

    Returns:
        Filtered list of detections (same dict format).
    """
    if not all_detections:
        return []

    # Sort by confidence descending
    dets = sorted(all_detections, key=lambda d: d['conf'], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        # Suppress anything that overlaps too much with `best`
        remaining = []
        for d in dets:
            if _compute_iou(best['box'], d['box']) < iou_threshold:
                remaining.append(d)
        dets = remaining

    return keep


# ─────────────────────────────────────────────────────────────────────
# 4. HIGH-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────────
def run_sahi_inference(yolo_model, image, conf_threshold=0.25,
                       slice_size=1024, overlap_ratio=0.20,
                       iou_threshold=0.5, center_bonus=0.05,
                       progress_callback=None):
    """
    Full SAHI pipeline: slice → infer → remap → NMS.

    Args:
        yolo_model: Loaded Ultralytics YOLO model.
        image: PIL Image (full 24MP resolution).
        conf_threshold: Minimum YOLO confidence.
        slice_size: Tile size in pixels.
        overlap_ratio: Overlap fraction between tiles.
        iou_threshold: NMS IoU threshold.
        center_bonus: Confidence bonus for core-zone detections.
        progress_callback: Optional callable(current, total) for UI updates.

    Returns:
        List of final detections after global NMS.
    """
    slices = generate_slices(image, slice_size, overlap_ratio)
    total = len(slices)

    all_detections = []
    for idx, s in enumerate(slices):
        # Run YOLO on tile
        # We explicitly force the internal YOLO resizer to match our tile size precisely
        results = yolo_model.predict(
            s['tile'], conf=conf_threshold, imgsz=1024, agnostic_nms=True, verbose=False
        )
        result = results[0] if results else None

        # Remap to global coords
        tile_dets = remap_detections(result, s, center_bonus=center_bonus)
        all_detections.extend(tile_dets)

        if progress_callback:
            progress_callback(idx + 1, total)

    # Global deduplication
    final = global_nms(all_detections, iou_threshold=iou_threshold)
    return final
