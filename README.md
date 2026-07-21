# Geological Time Scale Recognition Reproducible Package

This repository contains the code, model weights, and a small quick-test image
set for reproducing the geological time scale table recognition workflow
described in the accompanying manuscript.

The quick test runs the full inference pipeline on the sample images under
`data/quick_test/images/`:

```text
UNet line/border segmentation
-> grid JSON generation
-> YOLO cell instance segmentation
-> grid and YOLO result fusion
-> 4-class cell style classification
-> ruler OCR and value mapping
-> 7-class graph-cell classification
-> header OCR and header marking
```

## Environment

Python 3.10 is recommended. The OCR stages are tested with PaddleOCR 2.x; the
version is pinned in `requirements.txt` because PaddleOCR 3.x uses a different
runtime/API stack.

```bash
cd GTS_Reproducible_Package
python -m venv .venv

# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n gts-repro python=3.10 -y
conda activate gts-repro
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducibility Check

Check that the required files and packages are available:

```bash
python scripts/check_reproducibility.py
```

Run the full quick test:

```bash
python quick_test.py --device cpu --require-full
```

To save outputs to a separate directory:

```bash
python quick_test.py --device cpu --require-full --output-dir outputs/quick_test_run
```

If OCR dependencies are unavailable, the non-OCR core pipeline can be tested
with:

```bash
python quick_test.py --device cpu --require-full --skip-ocr
```

## Expected Result

A successful full quick test ends with:

```text
[quick-test] Done. Outputs: outputs/quick_test
```

The final JSON files are written to:

```text
outputs/quick_test/08_header/
```

The intermediate outputs are:

```text
outputs/quick_test/
  01_line_masks/      UNet line/border masks
  02_grid_json/       grid extraction results
  03_yolo/            YOLO labels and visualizations
  04_merged/          fused grid-cell structure JSON
  05_style/           4-class cell style classification JSON
  06_ruler/           ruler OCR and value mapping JSON
  07_graph/           graph-cell fine classification JSON
  08_header/          final JSON with header marks
```

## Repository Layout

```text
GTS_Reproducible_Package/
  quick_test.py
  requirements.txt
  README.md
  configs/
    table-cells.yaml
  scripts/
    check_reproducibility.py
    predict_yolo_cells.py
  code/
    Pytorch-UNet/
      run/01_predict.py
      run/02_test_vertical_lines.py
      run/03_merge_grid_yolo.py
    classification-pytorch/
      classification.py
      run/01_predict_cells.py
      run/02_ruler.py
      run/03_graph_cells.py
      run/04_header.py
      run/ocr_compat.py
  models/
    README.md
    line_unet_best.pth
    cell_yolo_best.pt
    style_mobilenet_4class_best.pth
    graph_mobilenet_7class_best.pth
  data/
    README.md
    quick_test/SOURCE_METADATA.csv
    quick_test/
      images/
      ground_truth/
```

## Data

This package includes only a small quick-test image set required to verify that
the code and model weights can run end to end. The full training and evaluation
datasets are not distributed with this repository because the original images
come from multiple published sources. Bibliographic and copyright notes for the
example images are recorded in `data/quick_test/SOURCE_METADATA.csv`.

## Models

The quick test requires the following model files:

```text
models/line_unet_best.pth
models/cell_yolo_best.pt
models/style_mobilenet_4class_best.pth
models/graph_mobilenet_7class_best.pth
```

`models/line_unet_best.pth` is larger than the 100 MB file-size limit for
regular GitHub uploads. If this repository is hosted on GitHub with the model
weights included, use Git LFS for the files matched by `.gitattributes`.

## Notes

- Generated outputs are ignored by Git through `.gitignore`.
- The code in this package is arranged for inference reproducibility. Training
  scripts and full experimental data splits should be documented separately if
  they are released.
