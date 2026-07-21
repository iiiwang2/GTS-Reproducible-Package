import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command, cwd):
    """Print and run one pipeline command."""
    clean_command = [str(x) for x in command if str(x)]
    print("\n[quick-test] " + " ".join(clean_command))
    subprocess.run(clean_command, cwd=str(cwd), check=True)


def parse_args():
    """Parse quick-test command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the GTS reproducibility quick test.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--require-full", action="store_true", help="Fail if the 4-class style model is missing.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip ruler/header OCR stages.")
    parser.add_argument("--output-dir", help="Directory for quick-test outputs. Defaults to outputs/quick_test.")
    return parser.parse_args()


def main():
    """Run the complete reproducibility workflow for all quick-test images."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable

    image_dir = root / "data" / "quick_test" / "images"
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in image_exts])
    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = root / out
    else:
        out = root / "outputs" / "quick_test"
    line_masks = out / "01_line_masks"
    grid_json = out / "02_grid_json"
    yolo_out = out / "03_yolo"
    merged = out / "04_merged"
    style_out = out / "05_style"
    ruler_out = out / "06_ruler"
    graph_out = out / "07_graph"
    header_out = out / "08_header"
    for d in [line_masks, grid_json, yolo_out, merged, style_out, ruler_out, graph_out, header_out]:
        d.mkdir(parents=True, exist_ok=True)

    required = {
        "quick-test image directory": image_dir,
        "line model": root / "models" / "line_unet_best.pth",
        "cell YOLO model": root / "models" / "cell_yolo_best.pt",
        "graph 7-class model": root / "models" / "graph_mobilenet_7class_best.pth",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if not images:
        raise FileNotFoundError(f"No quick-test images found in {image_dir}")

    style_model = root / "models" / "style_mobilenet_4class_best.pth"
    if args.require_full and not style_model.exists():
        raise FileNotFoundError(
            "Missing models/style_mobilenet_4class_best.pth. "
            "This 4-class style model is required for a full end-to-end run."
        )

    run(
        [
            py,
            root / "code" / "Pytorch-UNet" / "run" / "01_predict.py",
            "--model",
            root / "models" / "line_unet_best.pth",
            "--input",
            image_dir,
            "--output",
            line_masks,
        ],
        root,
    )

    run(
        [
            py,
            root / "code" / "Pytorch-UNet" / "run" / "02_test_vertical_lines.py",
            "--mask-dir",
            line_masks,
            "--out-dir",
            grid_json,
            "--name-suffix",
            "_OUT",
        ],
        root,
    )

    run(
        [
            py,
            root / "scripts" / "predict_yolo_cells.py",
            "--weights",
            root / "models" / "cell_yolo_best.pt",
            "--input",
            image_dir,
            "--output",
            yolo_out,
        ],
        root,
    )

    run(
        [
            py,
            root / "code" / "Pytorch-UNet" / "run" / "03_merge_grid_yolo.py",
            "--grid-dir",
            grid_json,
            "--yolo-dir",
            yolo_out / "labels",
            "--images-dir",
            image_dir,
            "--out-dir",
            merged,
            "--snap-always",
        ],
        root,
    )

    status = {
        "structure_pipeline": "completed if no error was raised before this point",
        "style_model_found": style_model.exists(),
        "note": "",
    }
    if not style_model.exists():
        status["note"] = (
            "The 4-class style model is missing. Add models/style_mobilenet_4class_best.pth "
            "to continue with text/ruler/graph/miss classification and downstream content parsing."
        )
        (out / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.require_full:
            raise FileNotFoundError(status["note"])
        print("\n[quick-test] Structure stages finished. Content stages skipped:")
        print(status["note"])
        return

    run(
        [
            py,
            root / "code" / "classification-pytorch" / "run" / "01_predict_cells.py",
            "--image-dir",
            image_dir,
            "--json-dir",
            merged,
            "--weights",
            style_model,
            "--classes",
            root / "code" / "classification-pytorch" / "cls_classes_4.txt",
            "--out-dir",
            style_out,
            "--scale-class",
            "ruler",
            "--scale-out-dir",
            style_out / "ruler_crops",
            "--cpu" if args.device == "cpu" else "",
        ],
        root,
    )

    current_json_dir = style_out
    if not args.skip_ocr:
        run(
            [
                py,
                root / "code" / "classification-pytorch" / "run" / "02_ruler.py",
                "--image-dir",
                image_dir,
                "--json-dir",
                current_json_dir,
                "--out-dir",
                ruler_out,
            ],
            root,
        )
        current_json_dir = ruler_out

    run(
        [
            py,
            root / "code" / "classification-pytorch" / "run" / "03_graph_cells.py",
            "--image-dir",
            image_dir,
            "--json-dir",
            current_json_dir,
            "--out-dir",
            graph_out,
            "--weights",
            root / "models" / "graph_mobilenet_7class_best.pth",
            "--classes",
            root / "code" / "classification-pytorch" / "cls_classes_7.txt",
            "--cpu" if args.device == "cpu" else "",
        ],
        root,
    )
    current_json_dir = graph_out

    if not args.skip_ocr:
        run(
            [
                py,
                root / "code" / "classification-pytorch" / "run" / "04_header.py",
                "--image-dir",
                image_dir,
                "--json-dir",
                current_json_dir,
                "--out-dir",
                header_out,
                "--header-dict",
                root / "code" / "classification-pytorch" / "run" / "header_dict.txt",
                "--cpu" if args.device == "cpu" else "",
            ],
            root,
        )

    status["note"] = "Full quick-test completed."
    status["image_count"] = len(images)
    status["images"] = [p.name for p in images]
    status["final_output_dir"] = str(header_out if not args.skip_ocr else graph_out)
    (out / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[quick-test] Done. Outputs: {out}")


if __name__ == "__main__":
    main()
