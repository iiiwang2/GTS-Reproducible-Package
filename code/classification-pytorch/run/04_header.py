import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from difflib import SequenceMatcher
from PIL import Image

from ocr_compat import create_paddle_ocr, ocr_text


"""Mark header-row cells in merged JSON using OCR and the header dictionary."""

SEP_RE = re.compile(r"[,\\|;:，；：\\\\/\\-_]+")
WS_RE = re.compile(r"\\s+")


def parse_args() -> argparse.Namespace:
    """Parse header-OCR command-line arguments."""
    parser = argparse.ArgumentParser(description="Mark header-row cells in JSON using OCR + header dict.")
    parser.add_argument("--image", help="Path to the source image.")
    parser.add_argument("--json", help="Path to input JSON with cells.")
    parser.add_argument("--image-dir", help="Directory with images for batch processing.")
    parser.add_argument("--json-dir", help="Directory with JSON files for batch processing.")
    parser.add_argument("--out-json", help="Path to write updated JSON.")
    parser.add_argument("--out-dir", help="Directory to write updated JSON files for batch.")
    parser.add_argument("--header-dict", required=True, help="Txt file with standard headers per line.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold.")
    parser.add_argument("--lang", default="ch", help="PaddleOCR language code.")
    parser.add_argument("--max-row", type=int, default=3, help="Only consider rows <= max-row for header.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU OCR.")
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    """Write a JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _find_image_for_json(image_dir: str, stem: str) -> Optional[str]:
    """Find the source image corresponding to one JSON stem."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    stem = stem.split("_OUT")[0]
    for ext in exts:
        cand = os.path.join(image_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def _iter_batch_pairs(image_dir: str, json_dir: str) -> Iterable[Tuple[str, str, str]]:
    """Yield matching image/JSON pairs for batch header detection."""
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


def _normalize_text(text: str) -> str:
    """Normalize OCR text before dictionary similarity matching."""
    text = text.strip().lower()
    text = WS_RE.sub("", text)
    text = SEP_RE.sub("/", text)
    text = text.strip("/")
    return text


def _load_header_dict(path: str) -> List[str]:
    """Load and normalize standard header strings from a text dictionary."""
    headers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            headers.append(_normalize_text(line))
    return headers


def _best_similarity(query_text: str, dict_texts: List[str]) -> float:
    """Return the best sequence-similarity score against dictionary entries."""
    if not dict_texts:
        return 0.0
    return max(SequenceMatcher(None, query_text, cand).ratio() for cand in dict_texts)


def _clamp_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    """Clamp a cell bounding box to image bounds."""
    x1, y1, x2, y2 = bbox
    x1_i = int(round(x1))
    y1_i = int(round(y1))
    x2_i = int(round(x2))
    y2_i = int(round(y2))
    x1_i = max(0, min(x1_i, width - 1))
    y1_i = max(0, min(y1_i, height - 1))
    x2_i = max(0, min(x2_i, width))
    y2_i = max(0, min(y2_i, height))
    return [x1_i, y1_i, x2_i, y2_i]


def _ocr_crop(ocr_engine, img: Image.Image, bbox: List[float]) -> str:
    """OCR one cropped cell image and return recognized text."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = img.crop((x1, y1, x2, y2))
    np_img = np.array(crop)
    return ocr_text(ocr_engine, np_img)


def _process_one(
        image_path: str,
        json_path: str,
        out_json: str,
        ocr_engine,
        header_texts: List[str],
        threshold: float,
        max_row: int,
) -> None:
    """Mark header rows in one image/JSON pair using OCR dictionary matching."""
    data = _load_json(json_path)
    cells = data.get("cells")
    if not isinstance(cells, list):
        raise ValueError("JSON does not contain a list field named 'cells'.")

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    row_hits: Dict[int, bool] = {}
    for cell in cells:
        row = cell.get("sr")
        if row is None:
            continue
        if row > max_row:
            continue
        bbox = cell.get("bbox_snapped") or cell.get("bbox_raw")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
        text = _ocr_crop(ocr_engine, img, [x1, y1, x2, y2])
        norm = _normalize_text(text)
        if not norm:
            continue
        score = _best_similarity(norm, header_texts)
        if score >= threshold:
            row_hits[row] = True

    for cell in cells:
        row = cell.get("sr")
        cell["is_header"] = bool(row is not None and row_hits.get(row))

    _save_json(out_json, data)
    print(f"Done. Updated JSON saved to: {out_json}")


def main() -> None:
    """Run single-image or batch header detection."""
    args = parse_args()
    ocr_engine = create_paddle_ocr(lang=args.lang, use_gpu=not args.cpu, show_log=False)

    header_texts = _load_header_dict(args.header_dict)

    if args.image_dir or args.json_dir:
        if not args.image_dir or not args.json_dir:
            raise ValueError("Both --image-dir and --json-dir are required for batch processing.")
        out_dir = args.out_dir or args.json_dir
        os.makedirs(out_dir, exist_ok=True)
        total = 0
        for img_path, json_path, stem in _iter_batch_pairs(args.image_dir, args.json_dir):
            out_json = os.path.join(out_dir, stem + ".json")
            _process_one(
                img_path,
                json_path,
                out_json,
                ocr_engine,
                header_texts,
                args.threshold,
                args.max_row,
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
        ocr_engine,
        header_texts,
        args.threshold,
        args.max_row,
    )


if __name__ == "__main__":
    main()
