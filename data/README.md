# Data

This directory contains the minimal quick-test data used to verify the
end-to-end inference workflow.

```text
data/quick_test/
  SOURCE_METADATA.csv
  images/
    ics_chronostrat_chart_2026_06.png
    f24-9.png
    p7-3-10.png
    DCXZ202302003_07100.jpg
  ground_truth/yolo/
    f24-9.txt
    p7-3-10.txt
    DCXZ202302003_07100.txt
  ground_truth/labelme/
    f24-9.labelme.json
    p7-3-10.labelme.json
    DCXZ202302003_07100.labelme.json
```

The full training and evaluation datasets are not distributed with this
package. The quick-test images are provided as example inputs for
reproducibility review, and their bibliographic and copyright information is
listed in `quick_test/SOURCE_METADATA.csv`.

The ICS chart image is an unannotated inference sample. The three local GTS
samples include YOLO cell labels and sanitized LabelMe cell-box annotations.
The LabelMe `imageData` fields are set to `null`; the corresponding images are
stored separately under `quick_test/images/`.
