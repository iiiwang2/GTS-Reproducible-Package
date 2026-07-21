import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


"""Merge generated grid JSON files with YOLO cell-segmentation labels."""

def load_grid(grid_path: Path):
    """Load grid JSON and extract line positions plus image size."""
    data = json.loads(grid_path.read_text(encoding="utf-8"))
    v_lines = [int(x) for x in data.get("vertical_lines", [])]
    h_lines = [int(y) for y in data.get("horizontal_lines", [])]
    col_rows = data.get("column_rows", {})
    w = int(data.get("imageWidth", 0))
    h = int(data.get("imageHeight", 0))
    return data, v_lines, h_lines, col_rows, w, h


def parse_yolo_seg(txt_path: Path):
    """Parse YOLO segmentation labels into polygons, bboxes, and confidence."""
    items = []
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        # Support optional confidence at the end: class x1 y1 ... xn yn conf
        if (len(parts) - 2) % 2 == 0:
            conf = float(parts[-1])
            coords = list(map(float, parts[1:-1]))
        else:
            conf = 1.0
            coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        items.append({"class": cls, "points": pts, "conf": conf})
    return items


def denorm_points(points, width, height):
    """Convert normalized YOLO polygon points to pixel coordinates."""
    return [(x * width, y * height) for x, y in points]


def bbox_from_points(points):
    """Compute an axis-aligned bounding box from polygon points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_iou(a, b):
    """Compute IoU between two axis-aligned bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_items(items, iou_thresh=0.3):
    """Apply confidence-sorted non-maximum suppression to cell detections."""
    if not items:
        return items
    # items must include bbox_raw and conf
    items_sorted = sorted(items, key=lambda x: x.get("conf", 1.0), reverse=True)
    kept = []
    for it in items_sorted:
        keep = True
        for k in kept:
            if bbox_iou(it["bbox_raw"], k["bbox_raw"]) >= iou_thresh:
                keep = False
                break
        if keep:
            kept.append(it)
    return kept


def _bbox_overlap(a, b):
    """Return whether two axis-aligned boxes overlap."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1)


def _adjust_bbox_min_area_loss(bbox, keep_bbox):
    """Shrink a box away from a kept box while minimizing retained-area loss."""
    bx0, by0, bx1, by1 = bbox
    kx0, ky0, kx1, ky1 = keep_bbox
    if not _bbox_overlap(bbox, keep_bbox):
        return bbox

    options = []
    # Move left edge to the right of keep box
    if kx1 < bx1:
        new_bbox = [kx1, by0, bx1, by1]
        loss = (kx1 - bx0) * (by1 - by0) if kx1 > bx0 else 0.0
        options.append((loss, abs(kx1 - bx0), new_bbox))
    # Move right edge to the left of keep box
    if kx0 > bx0:
        new_bbox = [bx0, by0, kx0, by1]
        loss = (bx1 - kx0) * (by1 - by0) if kx0 < bx1 else 0.0
        options.append((loss, abs(bx1 - kx0), new_bbox))
    # Move top edge below keep box
    if ky1 < by1:
        new_bbox = [bx0, ky1, bx1, by1]
        loss = (ky1 - by0) * (bx1 - bx0) if ky1 > by0 else 0.0
        options.append((loss, abs(ky1 - by0), new_bbox))
    # Move bottom edge above keep box
    if ky0 > by0:
        new_bbox = [bx0, by0, bx1, ky0]
        loss = (by1 - ky0) * (bx1 - bx0) if ky0 < by1 else 0.0
        options.append((loss, abs(by1 - ky0), new_bbox))

    if not options:
        return None

    # minimize area loss, then minimize shift distance
    _, _, best = min(options, key=lambda t: (t[0], t[1]))
    nx0, ny0, nx1, ny1 = best
    if nx0 >= nx1 or ny0 >= ny1:
        return None
    return best


def resolve_overlaps_by_conf(items):
    """Resolve overlapping cell boxes by confidence order."""
    if not items:
        return items
    items_sorted = sorted(items, key=lambda x: x.get("conf", 1.0), reverse=True)
    kept = []
    for it in items_sorted:
        bbox = it["bbox_raw"]
        valid = True
        for k in kept:
            bbox = _adjust_bbox_min_area_loss(bbox, k["bbox_raw"])
            if bbox is None:
                valid = False
                break
        if not valid:
            continue
        if bbox is None:
            continue
        it["bbox_raw"] = [float(v) for v in bbox]
        kept.append(it)
    return kept


def _nearest(val, candidates):
    """Return the candidate position nearest to `val`."""
    return min(candidates, key=lambda x: abs(x - val))


def remove_border_lines(v_lines, h_lines, width, height, border=1):
    """Remove line positions that lie on the outer image border."""
    v_filtered = [x for x in v_lines if border < x < width - border]
    h_filtered = [y for y in h_lines if border < y < height - border]
    return v_filtered, h_filtered


def filter_rows_by_support(h_lines, col_rows, min_support):
    """Keep horizontal lines supported by enough per-column rows."""
    if not h_lines:
        return h_lines, col_rows
    support = {int(r): 0 for r in h_lines}
    for rows in col_rows.values():
        for r in rows:
            r_int = int(r)
            if r_int in support:
                support[r_int] += 1
    kept = [r for r in h_lines if support.get(int(r), 0) >= min_support]
    kept_set = {int(r) for r in kept}
    col_rows_filtered = {k: [int(r) for r in rows if int(r) in kept_set] for k, rows in col_rows.items()}
    return kept, col_rows_filtered


def snap_bbox_to_grid(bbox, v_lines, h_lines, col_rows=None, snap_thresh=30, snap_always=False):
    """Snap a YOLO cell bbox to nearby grid lines and infer row/column span."""
    x0, y0, x1, y1 = bbox
    v_sorted = sorted(v_lines)
    h_sorted = sorted(h_lines)
    if not v_sorted or not h_sorted:
        return bbox, None, None

    vx0 = _nearest(x0, v_sorted)
    vx1 = _nearest(x1, v_sorted)
    hy0 = _nearest(y0, h_sorted)
    hy1 = _nearest(y1, h_sorted)

    # If both sides snap to the same line, fall back to enclosing lines around center.
    if vx0 == vx1:
        cx = 0.5 * (x0 + x1)
        left = max([v for v in v_sorted if v <= cx], default=v_sorted[0])
        right = min([v for v in v_sorted if v >= cx], default=v_sorted[-1])
        vx0, vx1 = left, right
    if hy0 == hy1:
        cy = 0.5 * (y0 + y1)
        top = max([h for h in h_sorted if h <= cy], default=h_sorted[0])
        bottom = min([h for h in h_sorted if h >= cy], default=h_sorted[-1])
        hy0, hy1 = top, bottom

    # Select horizontal candidates within the column span if column_rows exists
    h_candidates = h_sorted
    if col_rows:
        try:
            idx_left = v_sorted.index(vx0)
            idx_right = v_sorted.index(vx1)
            col_min = min(idx_left, idx_right)
            col_max = max(idx_left, idx_right) - 1
            rows = []
            for c in range(col_min, col_max + 1):
                rows.extend(col_rows.get(str(c), []))
            rows = sorted({int(r) for r in rows})
            if rows:
                h_candidates = rows
        except ValueError:
            pass

    hy0 = _nearest(y0, h_candidates)
    hy1 = _nearest(y1, h_candidates)
    if hy0 == hy1:
        cy = 0.5 * (y0 + y1)
        top = max([h for h in h_candidates if h <= cy], default=h_candidates[0])
        bottom = min([h for h in h_candidates if h >= cy], default=h_candidates[-1])
        hy0, hy1 = top, bottom

    # Apply snapping threshold
    if snap_always:
        x0n, x1n, y0n, y1n = vx0, vx1, hy0, hy1
        x0_line, x1_line, y0_line, y1_line = vx0, vx1, hy0, hy1
    else:
        x0n = vx0 if abs(vx0 - x0) <= snap_thresh else x0
        x1n = vx1 if abs(vx1 - x1) <= snap_thresh else x1
        y0n = hy0 if abs(hy0 - y0) <= snap_thresh else y0
        y1n = hy1 if abs(hy1 - y1) <= snap_thresh else y1
        x0_line = vx0 if x0n == vx0 else None
        x1_line = vx1 if x1n == vx1 else None
        y0_line = hy0 if y0n == hy0 else None
        y1_line = hy1 if y1n == hy1 else None

    # Ensure ordering
    if x0n >= x1n:
        x0n, x1n = min(x0, x1), max(x0, x1)
    if y0n >= y1n:
        y0n, y1n = min(y0, y1), max(y0, y1)

    # Indices for row/col (based only on the chosen boundary lines)
    col_idx = None
    row_idx = None
    if x0_line in v_sorted and x1_line in v_sorted:
        col_idx = v_sorted.index(x0_line)
    if y0_line in h_sorted and y1_line in h_sorted:
        row_idx = h_sorted.index(y0_line)

    # logical span indices: [start, end] inclusive (based only on chosen lines)
    sc = er = sr = ec = None
    if x0_line in v_sorted and x1_line in v_sorted:
        sc = v_sorted.index(x0_line)
        ec = max(sc, v_sorted.index(x1_line) - 1)
    if y0_line in h_sorted and y1_line in h_sorted:
        sr = h_sorted.index(y0_line)
        er = max(sr, h_sorted.index(y1_line) - 1)

    return [x0n, y0n, x1n, y1n], row_idx, col_idx, sr, er, sc, ec, x0_line, x1_line, y0_line, y1_line


def visualize(image, items, title):
    """Visualize snapped cell boxes on an image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image, cmap="gray")
    for it in items:
        x0, y0, x1, y1 = it["bbox_snapped"]
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1.5)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    """Merge grid-extraction results with YOLO cell detections."""
    parser = argparse.ArgumentParser(description="Merge grid with YOLOv11 cell segmentation results.")
    parser.add_argument("--grid-dir", required=True, help="Folder containing grid.json files")
    parser.add_argument("--yolo-dir", required=True, help="Folder containing YOLO txt files")
    parser.add_argument("--images-dir", default="", help="Optional images folder for visualization")
    parser.add_argument("--out-dir", required=True, help="Output folder for corrected cell json")
    parser.add_argument("--snap-thresh", type=int, default=30, help="Max snap distance in pixels")
    parser.add_argument("--snap-always", action="store_true", help="Always snap bbox edges to nearest grid lines")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument("--nms", action="store_true", help="Apply NMS on YOLO bboxes before snapping")
    parser.add_argument("--row-support", type=float, default=0.3,
                        help="Min column support ratio for keeping horizontal lines")
    parser.add_argument("--visualize", action="store_true", help="Show visualization (not saved)")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    yolo_dir = Path(args.yolo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([p for p in yolo_dir.iterdir() if p.suffix.lower() == ".txt"])
    for txt_path in txt_files:
        raw_stem = txt_path.stem
        stem = raw_stem + "_OUT"
        grid_candidates = [
            grid_dir / stem / "grid.json",
            grid_dir / raw_stem / "grid.json",
            grid_dir / f"{stem}.json",
            grid_dir / f"{raw_stem}.json",
        ]
        grid_path = next((p for p in grid_candidates if p.exists()), grid_candidates[0])
        if not grid_path.exists():
            print(f"[WARN] grid.json not found for {stem}")
            continue

        grid_data, v_lines, h_lines, col_rows, w, h = load_grid(grid_path)
        yolo_items = parse_yolo_seg(txt_path)
        if not yolo_items:
            continue

        if w == 0 or h == 0:
            # Try from image
            if args.images_dir:
                img_path = next(Path(args.images_dir).glob(raw_stem + ".*"), None)
                if img_path:
                    img = Image.open(img_path)
                    w, h = img.size
            if w == 0 or h == 0:
                print(f"[WARN] image size missing for {stem}")
                continue

        # Remove image border lines from grid
        # v_lines, h_lines = remove_border_lines(v_lines, h_lines, w, h, border=1)
        # Filter horizontal lines by column support to reduce false positives
        col_count = max(1, len(col_rows))
        min_support = max(2, int(col_count * args.row_support))
        # h_lines, col_rows = filter_rows_by_support(h_lines, col_rows, min_support=min_support)

        prepared = []
        for item in yolo_items:
            pts_abs = denorm_points(item["points"], w, h)
            bbox = bbox_from_points(pts_abs)
            prepared.append({
                "class": item["class"],
                "points": pts_abs,
                "bbox_raw": [float(v) for v in bbox],
                "conf": float(item.get("conf", 1.0)),
            })

        # postprocess: resolve overlaps by shifting lower-confidence boxes with minimal area loss
        prepared = resolve_overlaps_by_conf(prepared)

        corrected = []
        for item in prepared:
            bbox = item["bbox_raw"]
            snapped_bbox, row_idx, col_idx, sr, er, sc, ec, x0_line, x1_line, y0_line, y1_line = snap_bbox_to_grid(
                bbox, v_lines, h_lines, col_rows=col_rows,
                snap_thresh=args.snap_thresh, snap_always=args.snap_always
            )
            corrected.append({
                "class": item["class"],
                "polygon": [[float(x), float(y)] for x, y in item["points"]],
                "bbox_raw": [float(v) for v in bbox],
                "bbox_snapped": [float(v) for v in snapped_bbox],
                "sr": sr,
                "er": er,
                "sc": sc,
                "ec": ec,
                "conf": float(item.get("conf", 1.0)),
                "_x0_line": x0_line,
                "_x1_line": x1_line,
                "_y0_line": y0_line,
                "_y1_line": y1_line,
            })

        if args.nms:
            corrected = nms_items(corrected, iou_thresh=args.iou_thresh)

        # Re-index rows/cols based on all chosen boundary lines in this image
        chosen_v = sorted({v for it in corrected for v in (it.get("_x0_line"), it.get("_x1_line")) if v is not None})
        chosen_h = sorted({v for it in corrected for v in (it.get("_y0_line"), it.get("_y1_line")) if v is not None})
        map_v = {v: i for i, v in enumerate(chosen_v)}
        map_h = {v: i for i, v in enumerate(chosen_h)}
        for it in corrected:
            x0_line = it.get("_x0_line")
            x1_line = it.get("_x1_line")
            y0_line = it.get("_y0_line")
            y1_line = it.get("_y1_line")
            if x0_line in map_v and x1_line in map_v:
                sc = map_v[x0_line]
                ec = max(sc, map_v[x1_line] - 1)
            else:
                sc = ec = None
            if y0_line in map_h and y1_line in map_h:
                sr = map_h[y0_line]
                er = max(sr, map_h[y1_line] - 1)
            else:
                sr = er = None
            it["sr"] = sr
            it["er"] = er
            it["sc"] = sc
            it["ec"] = ec
            it.pop("_x0_line", None)
            it.pop("_x1_line", None)
            it.pop("_y0_line", None)
            it.pop("_y1_line", None)

        out_path = out_dir / f"{stem}.json"
        out_path.write_text(
            json.dumps({"imageWidth": w, "imageHeight": h, "cells": corrected}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.visualize:
            if args.images_dir:
                img_path = next(Path(args.images_dir).glob(raw_stem + ".*"), None)
                if img_path:
                    img = Image.open(img_path).convert("RGB")
                else:
                    img = Image.new("RGB", (w, h), (0, 0, 0))
            else:
                img = Image.new("RGB", (w, h), (0, 0, 0))
            visualize(img, corrected, title=stem)


if __name__ == "__main__":
    main()
