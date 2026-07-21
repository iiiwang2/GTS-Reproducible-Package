import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classification import Classification

"""Classify graph cells into fine-grained chart types."""



def _clamp_bbox(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
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
    return x1_i, y1_i, x2_i, y2_i


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    """Write a JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _iter_cells(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the cell list from a merged structure JSON."""
    cells = data.get("cells")
    if not isinstance(cells, list):
        raise ValueError("JSON does not contain a list field named 'cells'.")
    return cells


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
    """Yield matching image/JSON pairs for batch graph classification."""
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


def _process_one(
        image_path: str,
        json_path: str,
        out_json: str,
        classification: Classification,
        target_type: str,
        out_field: str,
) -> int:
    """Classify graph cells in one image and write the updated JSON."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    data = _load_json(json_path)
    cells = _iter_cells(data)

    skipped = 0
    for cell in cells:
        if cell.get("type") != target_type:
            continue
        bbox = cell.get("bbox_snapped") or cell.get("bbox_raw")
        if not bbox or len(bbox) != 4:
            skipped += 1
            continue

        x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        crop = image.crop((x1, y1, x2, y2))
        class_name = classification.detect_image(crop)
        cell[out_field] = class_name

    _save_json(out_json, data)
    return skipped


def main() -> None:
    """Parse command-line arguments and run graph-cell fine classification."""
    parser = argparse.ArgumentParser(
        description="Classify graph cells by bbox_snapped and write fine-grained types to JSON."
    )
    parser.add_argument("--image", help="Path to the source image.")
    parser.add_argument("--json", help="Path to the input JSON with cells and bbox_snapped.")
    parser.add_argument("--image-dir", help="Directory with images for batch processing.")
    parser.add_argument("--json-dir", help="Directory with JSON files for batch processing.")
    parser.add_argument(
        "--weights",
        default="../../models/graph_mobilenet_7class_best.pth",
        help="Path to model weights (.pth).",
    )
    parser.add_argument(
        "--classes",
        default="../cls_classes_7.txt",
        help="Classes file path.",
    )
    parser.add_argument("--out-json", help="Path to write the updated JSON.")
    parser.add_argument("--out-dir", help="Directory to write updated JSON files for batch.")
    parser.add_argument("--target-type", default="graph", help="Only classify cells with this type.")
    parser.add_argument("--out-field", default="type_fine", help="Field name to store fine-grained type.")
    parser.add_argument("--scale-class", help="(unused) kept for CLI compatibility", default=None)
    parser.add_argument("--scale-out-dir", help="(unused) kept for CLI compatibility", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()

    classification = Classification(
        model_path=args.weights,
        classes_path=args.classes,
        cuda=not args.cpu,
    )

    if args.image_dir or args.json_dir:
        if not args.image_dir or not args.json_dir:
            raise ValueError("Both --image-dir and --json-dir are required for batch processing.")
        out_dir = args.out_dir or args.json_dir
        os.makedirs(out_dir, exist_ok=True)
        total = 0
        skipped_total = 0
        for img_path, json_path, stem in _iter_batch_pairs(args.image_dir, args.json_dir):
            out_json = os.path.join(out_dir, stem + ".json")
            skipped = _process_one(
                img_path,
                json_path,
                out_json,
                classification,
                args.target_type,
                args.out_field,
            )
            skipped_total += skipped
            total += 1
        print(f"Done. Updated JSON saved to: {out_dir}. Files: {total}. Skipped cells: {skipped_total}")
        return

    if not args.image or not args.json or not args.out_json:
        raise ValueError("--image, --json, and --out-json are required for single processing.")

    skipped = _process_one(
        args.image,
        args.json,
        args.out_json,
        classification,
        args.target_type,
        args.out_field,
    )
    print(f"Done. Updated JSON saved to: {args.out_json}. Skipped cells: {skipped}")


if __name__ == "__main__":
    main()
