import argparse
import os
from pathlib import Path


def parse_args():
    """Parse YOLO prediction command-line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO cell segmentation and save labels/images.")
    parser.add_argument("--weights", required=True, help="Path to YOLO segmentation weights.")
    parser.add_argument("--input", required=True, help="Input image or image directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--save-conf", action="store_true", help="Save confidence values in txt labels.")
    return parser.parse_args()


def iter_images(path: Path):
    """Return all supported image files under a file path or directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def configure_ultralytics():
    """Use a repository-local Ultralytics settings directory."""
    config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", Path(__file__).resolve().parents[1] / ".ultralytics"))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def main():
    """Run YOLO inference and export labels plus visualized predictions."""
    args = parse_args()
    configure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(args.weights)

    input_path = Path(args.input)
    output_root = Path(args.output)
    labels_dir = output_root / "labels"
    images_dir = output_root / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No input images found: {input_path}")

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            save=False,
            conf=args.conf,
            iou=args.iou,
        )
        out_txt = labels_dir / f"{image_path.stem}.txt"
        out_img = images_dir / image_path.name
        if out_txt.exists():
            out_txt.unlink()
        for result in results:
            result.save_txt(out_txt, save_conf=args.save_conf)
            result.save(str(out_img), color_mode="instance")

    print(f"YOLO labels saved to: {labels_dir}")
    print(f"YOLO visualizations saved to: {images_dir}")


if __name__ == "__main__":
    main()
