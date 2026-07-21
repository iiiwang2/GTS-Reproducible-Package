"""
Parse ruler cells, infer table-cell value bounds, and OCR text cells.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from ocr_compat import create_paddle_ocr, ocr_detect_text

# A single OCR engine is reused across all cells to avoid repeated model loading.
ocr = create_paddle_ocr(lang="ch", use_gpu=False)


def _read_image_bgr(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read an image as BGR while supporting non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: str, image: np.ndarray) -> bool:
    """Write an image while supporting non-ASCII Windows paths."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def shrink_box_xy(box, sx=0.75, sy=0.9):
    """
    Shrink an OCR text polygon before masking text from the ruler crop.

    `sx` and `sy` control the horizontal and vertical shrink ratios.
    """
    box = np.array(box, dtype=np.float32)
    cx = np.mean(box[:, 0])
    cy = np.mean(box[:, 1])

    box[:, 0] = cx + (box[:, 0] - cx) * sx
    box[:, 1] = cy + (box[:, 1] - cy) * sy

    return box.astype(np.int32)


def detect_minor_ticks(
        img_path_or_img: Union[str, np.ndarray],
        remove_text: bool = True,
        orientation: str = "vertical",  # "vertical" or "horizontal"
        match_text: bool = True,
        visualize: bool = False
) -> Dict[str, Any]:
    """
    Detect ruler tick positions and match nearby OCR numbers.

    For a vertical ruler, horizontal ticks are detected by row projection.
    For a horizontal ruler, vertical ticks are detected by column projection.
    """

    # ======================
    # 0. Read image
    # ======================
    if isinstance(img_path_or_img, (str, Path)):
        img = _read_image_bgr(img_path_or_img)
    else:
        img = img_path_or_img
    if img is None:
        raise ValueError("Failed to load image for ruler detection.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # OCR is used both for text removal and numeric tick matching.
    result = None
    if remove_text or match_text:
        result = ocr_detect_text(ocr, img_rgb)

    if remove_text:
        text_mask = np.zeros((h, w), np.uint8)

        if result is not None:
            for line in result:
                for box, (txt, score) in line:
                    pts = shrink_box_xy(box, sx=0.75, sy=0.9)
                    cv2.fillPoly(text_mask, [pts], 255)

        # Slight dilation helps remove the full text stroke before thresholding.
        text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)

        gray_proc = gray.copy()
        gray_proc[text_mask == 255] = 255
    else:
        text_mask = np.zeros((h, w), np.uint8)
        gray_proc = gray.copy()

    # Binarize the ruler crop while preserving thin tick marks.
    bin_img = cv2.adaptiveThreshold(
        gray_proc, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    # Keep thin line structures in the expected tick direction.
    if orientation == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    # Project the binary image along the ruler direction.
    if orientation == "vertical":
        # Row projection detects horizontal ticks.
        projection = np.sum(bin_img > 0, axis=1)
        axis_len = h
    else:
        # Column projection detects vertical ticks.
        projection = np.sum(bin_img > 0, axis=0)
        axis_len = w

    projection_smooth = cv2.GaussianBlur(
        projection.astype(np.float32).reshape(-1, 1),
        (1, 9), 0
    ).flatten()

    proj_norm = projection_smooth / (np.max(projection_smooth) + 1e-6)

    # Projection peaks are candidate tick positions.
    peaks, _ = find_peaks(
        proj_norm,
        distance=max(3, axis_len // 500),
        prominence=0.04,
        height=0.12
    )

    minor_ticks = peaks.tolist()

    # Match OCR numbers to detected ticks and optionally draw debug output.
    tick_values = None
    matched_pairs = []
    ocr_numbers = []

    def _normalize_ocr_num(text):
        # Fix common OCR confusions for numbers.
        text = text.replace("O", "0").replace("o", "0")
        text = text.replace("I", "1").replace("l", "1")
        text = text.replace(",", ".")
        return text

    if match_text and result is not None:
        for line in result:
            for box, (txt, score) in line:
                raw = _normalize_ocr_num(txt.strip())
                m = re.search(r"-?\d+(?:\.\d+)?", raw)
                if not m:
                    continue
                try:
                    value = float(m.group(0))
                except ValueError:
                    continue

                box_np = np.array(box, dtype=np.float32)
                cx = float(np.mean(box_np[:, 0]))
                cy = float(np.mean(box_np[:, 1]))
                pos = cy if orientation == "vertical" else cx
                ocr_numbers.append({"value": value, "pos": pos, "box": box_np})

        if minor_ticks and ocr_numbers:
            match_tol = max(12, axis_len * 0.05)
            for item in ocr_numbers:
                pos = item["pos"]
                nearest = min(minor_ticks, key=lambda t: abs(t - pos))
                if abs(nearest - pos) <= match_tol:
                    matched_pairs.append((nearest, item["value"]))

        if matched_pairs:
            tick_values = matched_pairs[:]

    vis = img_rgb.copy()

    matched_ticks = [p for p, v in matched_pairs]
    draw_ticks = matched_ticks if (match_text and matched_ticks) else minor_ticks

    if orientation == "vertical":
        for y in draw_ticks:
            cv2.line(vis, (0, y), (w - 1, y), (255, 0, 0), 1)
    else:
        for x in draw_ticks:
            cv2.line(vis, (x, 0), (x, h - 1), (255, 0, 0), 1)

    for item in ocr_numbers:
        box = item["box"].astype(np.int32)
        cv2.polylines(vis, [box], True, (255, 255, 0), 1)

    if tick_values is not None:
        if orientation == "vertical":
            text_x = max(5, w - 90)
            for y, v in tick_values:
                if match_text and matched_ticks and y not in matched_ticks:
                    continue
                label = str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.2f}"
                cv2.putText(vis, label, (text_x, int(y) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            text_y = 15
            for x, v in tick_values:
                if match_text and matched_ticks and x not in matched_ticks:
                    continue
                label = str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.2f}"
                cv2.putText(vis, label, (int(x) - 10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    if visualize:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("Text Mask")
        plt.imshow(text_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("Gray (After Text Removal)")
        plt.imshow(gray_proc, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title("Binary Image")
        plt.imshow(bin_img, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.title("Projection + Peaks")

        if orientation == "vertical":
            plt.plot(proj_norm, np.arange(h))
            plt.gca().invert_yaxis()
            for y in minor_ticks:
                plt.axhline(y, color="r", linestyle="--", linewidth=0.6)
        else:
            plt.plot(np.arange(w), proj_norm)
            for x in minor_ticks:
                plt.axvline(x, color="r", linestyle="--", linewidth=0.6)

        plt.subplot(2, 3, 6)
        plt.title("Detected Minor Ticks")
        plt.imshow(vis)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # Return the structured result used by the value-mapping step.
    matched_ticks = [p for p, v in matched_pairs]

    if orientation == "vertical":
        return {
            "orientation": "vertical",
            "rows": matched_ticks if (match_text and matched_ticks) else minor_ticks,
            "matched_pairs": matched_pairs,
            "tick_values": tick_values
        }
    else:
        return {
            "orientation": "horizontal",
            "cols": matched_ticks if (match_text and matched_ticks) else minor_ticks,
            "matched_pairs": matched_pairs,
            "tick_values": tick_values
        }


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    """Write a JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _find_image_for_json(image_dir: str, stem: str) -> Optional[str]:
    """Find the source image that corresponds to an output JSON stem."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    stem = stem.split("_OUT")[0]
    for ext in exts:
        cand = os.path.join(image_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def _iter_batch_pairs(image_dir: str, json_dir: str) -> Iterable[Tuple[str, str, str]]:
    """Yield matching image/JSON pairs for batch processing."""
    for name in sorted(os.listdir(json_dir)):
        if not name.lower().endswith(".json"):
            continue
        json_path = os.path.join(json_dir, name)
        stem = os.path.splitext(name)[0]
        img_path = _find_image_for_json(image_dir, stem)
        if img_path is None:
            print(f"[WARN] image not found for {json_path}")
            continue
        yield img_path, json_path, stem


def _clamp_bbox(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a bounding box to image bounds and return integer coordinates."""
    x1, y1, x2, y2 = bbox
    x1_i = int(round(x1))
    y1_i = int(round(y1))
    x2_i = int(round(x2))
    y2_i = int(round(y2))
    x1_i = max(0, min(x1_i, width - 1))
    y1_i = max(0, min(y1_i, height - 1))
    x2_i = max(0, min(x2_i, width))
    y2_i = max(0, min(y2_i, height))
    return x1_i, y1_i, x2_i, y2_i


def _select_ruler_cell(cells: List[Dict[str, Any]], ruler_type: str) -> Optional[Dict[str, Any]]:
    """Select the largest cell whose coarse type is the ruler class."""
    rulers = [c for c in cells if c.get("type") == ruler_type]
    if not rulers:
        return None

    def _area(cell: Dict[str, Any]) -> float:
        """Return the area of a candidate ruler cell."""
        bbox = cell.get("bbox_snapped") or cell.get("bbox_raw") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    return max(rulers, key=_area)


def _fit_ruler_mapping(
        matched_pairs: List[Tuple[float, float]],
        axis_offset: float,
        pos_min: Optional[float] = None,
        pos_max: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Fit a linear mapping from image coordinates to ruler values."""
    if len(matched_pairs) < 2:
        return None
    axis_positions = [axis_offset + float(pos) for pos, _ in matched_pairs]
    values = [float(v) for _, v in matched_pairs]
    slope, intercept = np.polyfit(axis_positions, values, 1)
    fit_pos_min = min(axis_positions)
    fit_pos_max = max(axis_positions)
    pos_min = fit_pos_min if pos_min is None else float(pos_min)
    pos_max = fit_pos_max if pos_max is None else float(pos_max)
    val_min = slope * pos_min + intercept
    val_max = slope * pos_max + intercept
    pairs = [{"pos": p, "value": v} for p, v in zip(axis_positions, values)]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pos_min": float(pos_min),
        "pos_max": float(pos_max),
        "value_min": float(val_min),
        "value_max": float(val_max),
        "fit_pos_min": float(fit_pos_min),
        "fit_pos_max": float(fit_pos_max),
        "pairs": pairs,
    }


def _infer_cell_bounds(
        bbox: List[float],
        axis: str,
        mapping: Optional[Dict[str, Any]]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Map a cell bbox to ruler-coordinate bounds when it falls inside the ruler range."""
    if not bbox or len(bbox) != 4:
        return None, None, None, None
    x1, y1, x2, y2 = bbox
    if axis == "y":
        pos_top = float(min(y1, y2))
        pos_bottom = float(max(y1, y2))
    else:
        pos_top = float(min(x1, x2))
        pos_bottom = float(max(x1, x2))

    if mapping is None:
        return pos_top, pos_bottom, None, None

    pos_min = mapping["pos_min"]
    pos_max = mapping["pos_max"]

    tolerance = 5.0
    if pos_top < (pos_min - tolerance) or pos_bottom > (pos_max + tolerance):
        return pos_top, pos_bottom, None, None

    pos_top_clamped = min(max(pos_top, pos_min), pos_max)
    pos_bottom_clamped = min(max(pos_bottom, pos_min), pos_max)

    slope = mapping["slope"]
    intercept = mapping["intercept"]
    val_top = slope * pos_top_clamped + intercept
    val_bottom = slope * pos_bottom_clamped + intercept
    return pos_top, pos_bottom, float(val_top), float(val_bottom)


def _process_one(
        image_path: str,
        json_path: str,
        out_json: str,
        out_vis: Optional[str],
        ruler_type: str,
        orientation_arg: str,
        remove_text: bool,
        match_text: bool,
        visualize: bool,
) -> None:
    """Run ruler parsing and text OCR for one image/JSON pair."""
    data = _load_json(json_path)
    cells = data.get("cells")
    if not isinstance(cells, list):
        raise ValueError("JSON does not contain a list field named 'cells'.")

    img = _read_image_bgr(image_path)
    if img is None:
        raise ValueError("Failed to load image.")
    height, width = img.shape[:2]

    ruler_cell = _select_ruler_cell(cells, ruler_type)
    mapping = None
    ruler_info: Dict[str, Any] = {"status": "missing"}

    if ruler_cell is not None:
        bbox = ruler_cell.get("bbox_snapped") or ruler_cell.get("bbox_raw")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
            crop = img[y1:y2, x1:x2]
            if orientation_arg == "auto":
                orientation = "vertical" if (y2 - y1) >= (x2 - x1) else "horizontal"
            else:
                orientation = orientation_arg

            result = detect_minor_ticks(
                crop,
                remove_text=remove_text,
                orientation=orientation,
                match_text=match_text,
                visualize=visualize,
            )
            axis_offset = y1 if orientation == "vertical" else x1
            pos_min = y1 if orientation == "vertical" else x1
            pos_max = y2 if orientation == "vertical" else x2
            mapping = _fit_ruler_mapping(
                result.get("matched_pairs", []),
                axis_offset,
                pos_min=pos_min,
                pos_max=pos_max
            )
            axis = "y" if orientation == "vertical" else "x"
            if mapping is not None:
                ruler_info = {
                    "status": "ok",
                    "orientation": orientation,
                    "axis": axis,
                    "bbox_snapped": [x1, y1, x2, y2],
                    "pos_min": mapping["pos_min"],
                    "pos_max": mapping["pos_max"],
                    "value_min": mapping["value_min"],
                    "value_max": mapping["value_max"],
                    "fit_pos_min": mapping["fit_pos_min"],
                    "fit_pos_max": mapping["fit_pos_max"],
                    "pairs": mapping["pairs"],
                }
            else:
                ruler_info = {
                    "status": "no_mapping",
                    "orientation": orientation,
                    "axis": axis,
                    "bbox_snapped": [x1, y1, x2, y2],
                    "reason": "matched_pairs < 2"
                }

    axis = ruler_info.get("axis", "y")
    for cell in cells:
        bbox = cell.get("bbox_snapped") or cell.get("bbox_raw")
        pos_top, pos_bottom, val_top, val_bottom = _infer_cell_bounds(bbox, axis, mapping)
        cell["ruler_pos_top"] = pos_top
        cell["ruler_pos_bottom"] = pos_bottom
        cell["ruler_value_top"] = val_top
        cell["ruler_value_bottom"] = val_bottom
        if cell.get("type") == "text":
            cell["text"] = _ocr_cell_text(img, bbox)

    data["ruler"] = ruler_info
    _save_json(out_json, data)
    print(f"Done. Updated JSON saved to: {out_json}")

    if out_vis:
        vis = img.copy()
        for cell in cells:
            bbox = cell.get("bbox_snapped") or cell.get("bbox_raw")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
            val_top = cell.get("ruler_value_top")
            val_bottom = cell.get("ruler_value_bottom")
            if val_top is None or val_bottom is None:
                color = (0, 0, 255)
                label = "out"
            else:
                color = (0, 200, 0)
                label = f"{val_top:.2f}->{val_bottom:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

            if axis == "y":
                text_x = min(x2 + 3, width - 5)
                text_y = max(10, y1 + 12)
            else:
                text_x = max(2, x1)
                text_y = max(10, y1 - 4)
            cv2.putText(
                vis,
                label,
                (int(text_x), int(text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

        out_dir = os.path.dirname(out_vis)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _write_image(out_vis, vis)
        print(f"Visualization saved to: {out_vis}")


def _ocr_cell_text(img_bgr: np.ndarray, bbox: List[float]) -> str:
    """OCR one text cell and concatenate detected text in reading order."""
    if img_bgr is None or not bbox or len(bbox) != 4:
        return ""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    result = ocr_detect_text(ocr, crop_rgb)
    items: List[Tuple[float, float, str]] = []
    for line in result or []:
        for box, (txt, score) in line:
            if not txt:
                continue
            box_np = np.array(box, dtype=np.float32)
            cx = float(np.mean(box_np[:, 0]))
            cy = float(np.mean(box_np[:, 1]))
            items.append((cy, cx, txt.strip()))
    if not items:
        return ""
    items.sort(key=lambda x: (x[0], x[1]))
    return "".join([t for _, _, t in items]).strip()


def main() -> None:
    """Parse command-line arguments and run single-image or batch ruler parsing."""
    parser = argparse.ArgumentParser(description="Parse ruler cells and infer value bounds for all cells.")
    parser.add_argument("--image", help="Path to the source image.")
    parser.add_argument("--json", help="Path to the input JSON.")
    parser.add_argument("--image-dir", help="Directory with images for batch processing.")
    parser.add_argument("--json-dir", help="Directory with JSON files for batch processing.")
    parser.add_argument("--out-json", help="Path to write the updated JSON.")
    parser.add_argument("--out-dir", help="Directory to write updated JSON files for batch.")
    parser.add_argument("--out-vis", help="Path to save visualization image.")
    parser.add_argument("--out-vis-dir", help="Directory to save visualization images for batch.")
    parser.add_argument("--ruler-type", default="ruler", help="Cell type name for ruler.")
    parser.add_argument("--orientation", default="auto", choices=["auto", "vertical", "horizontal"],
                        help="Ruler orientation.")
    parser.add_argument("--no-remove-text", action="store_true", help="Do not remove text before tick detection.")
    parser.add_argument("--no-match-text", action="store_true", help="Do not match OCR numbers to ticks.")
    parser.add_argument("--visualize", action="store_true", help="Show debug visualizations.")
    args = parser.parse_args()
    if args.image_dir or args.json_dir:
        if not args.image_dir or not args.json_dir:
            raise ValueError("Both --image-dir and --json-dir are required for batch processing.")
        out_dir = args.out_dir or args.json_dir
        os.makedirs(out_dir, exist_ok=True)
        vis_dir = args.out_vis_dir
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
        total = 0
        for img_path, json_path, stem in _iter_batch_pairs(args.image_dir, args.json_dir):
            out_json = os.path.join(out_dir, stem + ".json")
            out_vis = os.path.join(vis_dir, stem + ".png") if vis_dir else None
            _process_one(
                img_path,
                json_path,
                out_json,
                out_vis,
                args.ruler_type,
                args.orientation,
                not args.no_remove_text,
                not args.no_match_text,
                args.visualize,
            )
            total += 1
        print(f"Done. Updated JSON saved to: {out_dir}. Files: {total}")
        return

    if not args.image or not args.json or not args.out_json:
        raise ValueError("--image, --json, and --out-json are required for single processing.")

    _process_one(
        args.image,
        args.json,
        args.out_json,
        args.out_vis,
        args.ruler_type,
        args.orientation,
        not args.no_remove_text,
        not args.no_match_text,
        args.visualize,
    )


if __name__ == "__main__":
    main()
