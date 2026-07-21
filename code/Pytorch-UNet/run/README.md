# Structure Recognition Scripts

These scripts implement the structure-recognition part of the quick test.

```text
01_predict.py
  Predict line/border masks from input GTS images using the trained UNet weight.

02_test_vertical_lines.py
  Convert predicted masks into vertical/horizontal separators and grid JSON.

03_merge_grid_yolo.py
  Merge the grid JSON with YOLO cell detections and infer corrected cell boxes
  plus row/column indices.
```

YOLO cell prediction is executed by `scripts/predict_yolo_cells.py` in the
repository root, so there is no separate `04_*.py` script in this directory.
