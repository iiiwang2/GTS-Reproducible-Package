import importlib.util
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    ("quick-test image directory", ROOT / "data/quick_test/images"),
    ("quick-test source metadata", ROOT / "data/quick_test/SOURCE_METADATA.csv"),
    ("line model", ROOT / "models/line_unet_best.pth"),
    ("cell YOLO model", ROOT / "models/cell_yolo_best.pt"),
    ("graph 7-class model", ROOT / "models/graph_mobilenet_7class_best.pth"),
]

OPTIONAL_BUT_NEEDED_FOR_FULL_RUN = [
    ("style 4-class model", ROOT / "models/style_mobilenet_4class_best.pth"),
]

PACKAGES = [
    ("numpy", True),
    ("PIL", True),
    ("cv2", True),
    ("torch", True),
    ("torchvision", True),
    ("ultralytics", True),
    ("scipy", True),
    ("shapely", False),
    ("paddleocr", False),
    ("paddle", False),
]

VERSION_CHECKS = [
    ("paddleocr", "paddleocr", 2),
    ("paddle", "paddlepaddle", 2),
]


def package_status(module_name):
    """Return whether a Python module can be imported."""
    return importlib.util.find_spec(module_name) is not None


def major_version(distribution_name):
    """Return the installed version string and parsed major version."""
    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None, None
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        major = None
    return version, major


def main():
    """Print file, dependency, and OCR-version checks for the package."""
    print("GTS reproducibility package check\n")

    print("[Files]")
    for label, path in REQUIRED_FILES:
        print(f"{'OK' if path.exists() else 'MISSING':8} {label:24} {path.relative_to(ROOT)}")
    image_dir = ROOT / "data/quick_test/images"
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_count = len([p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]) if image_dir.exists() else 0
    print(f"{'OK' if image_count else 'MISSING':8} {'quick-test image count':24} {image_count}")

    print("\n[Files needed for complete content recognition]")
    for label, path in OPTIONAL_BUT_NEEDED_FOR_FULL_RUN:
        print(f"{'OK' if path.exists() else 'MISSING':8} {label:24} {path.relative_to(ROOT)}")

    print("\n[Python packages]")
    for module_name, required in PACKAGES:
        ok = package_status(module_name)
        need = "required" if required else "optional"
        print(f"{'OK' if ok else 'MISSING':8} {module_name:16} {need}")

    print("\n[OCR version check]")
    for module_name, distribution_name, expected_major in VERSION_CHECKS:
        if not package_status(module_name):
            print(f"MISSING  {distribution_name:16} needed only for OCR stages")
            continue
        version, major = major_version(distribution_name)
        if major == expected_major:
            print(f"OK       {distribution_name:16} {version}")
        else:
            print(
                f"WARN     {distribution_name:16} {version} "
                f"(expected {expected_major}.x for the OCR quick-test)"
            )

    print("\nWhen all required files are OK, install the missing Python packages and run:")
    print("python quick_test.py --device cpu --require-full")


if __name__ == "__main__":
    main()
