# Model Weights

The quick test requires the following model files:

```text
line_unet_best.pth
cell_yolo_best.pt
style_mobilenet_4class_best.pth
graph_mobilenet_7class_best.pth
```

## Weight Roles

| File | Role |
|---|---|
| `line_unet_best.pth` | UNet line/border segmentation |
| `cell_yolo_best.pt` | YOLO cell instance segmentation |
| `style_mobilenet_4class_best.pth` | 4-class coarse cell style classification: `graph`, `miss`, `ruler`, `text` |
| `graph_mobilenet_7class_best.pth` | 7-class fine graph-cell classification |

The two MobileNetV2 classification weights were checked by their final
classifier layers:

```text
style_mobilenet_4class_best.pth   classifier output: 4 classes
graph_mobilenet_7class_best.pth   classifier output: 7 classes
```

If any model weight cannot be redistributed publicly, remove it from the public
repository and explain the access condition in the manuscript and root README.
