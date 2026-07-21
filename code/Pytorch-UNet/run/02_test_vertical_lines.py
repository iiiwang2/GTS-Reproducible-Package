import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def otsu_threshold(values: np.ndarray) -> int:
    """Compute an Otsu threshold for a one-dimensional projection."""
    values = values.astype(np.int64)
    if values.size == 0:
        return 0
    max_val = int(values.max())
    if max_val == 0:
        return 0
    hist = np.bincount(values, minlength=max_val + 1).astype(np.float64)
    total = values.size
    sum_total = np.dot(np.arange(max_val + 1), hist)

    weight_bg = 0.0
    sum_bg = 0.0
    var_max = -1.0
    threshold = 0
    for t in range(max_val + 1):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold


def find_peaks_1d(values: np.ndarray, thr: int):
    """Find plateau-aware peaks above a global threshold."""
    peaks = []
    n = len(values)
    i = 0
    while i < n:
        if values[i] < thr:
            i += 1
            continue
        # plateau peak handling
        start = i
        while i + 1 < n and values[i + 1] == values[start]:
            i += 1
        end = i
        left = values[start - 1] if start - 1 >= 0 else -1
        right = values[end + 1] if end + 1 < n else -1
        if values[start] >= left and values[end] >= right:
            peaks.append((start + end) // 2)
        i += 1
    return peaks


def enforce_min_distance(peaks, min_dist: int):
    """Keep peaks separated by at least `min_dist` pixels."""
    if not peaks:
        return peaks
    peaks_sorted = sorted(peaks)
    kept = [peaks_sorted[0]]
    for p in peaks_sorted[1:]:
        if p - kept[-1] >= min_dist:
            kept.append(p)
    return kept


def enforce_min_height(peaks, values: np.ndarray, min_height: int):
    """Remove peaks whose projection height is too small."""
    if not peaks:
        return peaks
    return [p for p in peaks if values[p] >= min_height]


def find_peaks_1d_local(values: np.ndarray, thresholds: np.ndarray):
    """Find peaks using a per-position local threshold."""
    peaks = []
    n = len(values)
    i = 0
    while i < n:
        if values[i] < thresholds[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and values[i + 1] == values[start]:
            i += 1
        end = i
        left = values[start - 1] if start - 1 >= 0 else -1
        right = values[end + 1] if end + 1 < n else -1
        if values[start] >= left and values[end] >= right:
            peaks.append((start + end) // 2)
        i += 1
    return peaks


def local_otsu_thresholds(values: np.ndarray, window: int) -> np.ndarray:
    """Compute sliding-window Otsu thresholds for a projection."""
    if window < 3:
        raise ValueError("window must be >= 3")
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    thresholds = np.zeros_like(values, dtype=np.int64)
    for i in range(len(values)):
        seg = padded[i:i + window]
        thresholds[i] = otsu_threshold(seg)
    return thresholds


def find_vertical_lines(
    mask: np.ndarray,
    alpha: float = 0.7,
    window: int = 201,
    min_dist: int = 10,
    min_height: int = 50,
):
    """Detect vertical grid-line positions from a predicted mask."""
    binary = (mask > 0).astype(np.uint8)
    col_sums = binary.sum(axis=0)
    thr = otsu_threshold(col_sums)
    thr_relaxed = int(thr * alpha)
    positions = find_peaks_1d(col_sums, thr_relaxed)
    positions = enforce_min_height(positions, col_sums, min_height)
    positions = enforce_min_distance(positions, min_dist)
    local_thr = local_otsu_thresholds(col_sums, window=window)
    positions_local = find_peaks_1d_local(col_sums, local_thr)
    positions_local = enforce_min_height(positions_local, col_sums, min_height)
    positions_local = enforce_min_distance(positions_local, min_dist)

    return col_sums, thr, thr_relaxed, positions, local_thr, positions_local


def find_horizontal_lines(
    mask: np.ndarray,
    alpha: float = 0.5,
    window: int = 201,
    min_dist: int = 10,
    min_height: int = 50,
    min_height_ratio: float = 0.2,
):
    """Detect horizontal grid-line positions from a predicted mask."""
    binary = (mask > 0).astype(np.uint8)
    row_sums = binary.sum(axis=1)
    adaptive_min_height = max(1, int(row_sums.max() * min_height_ratio)) if row_sums.size else min_height
    min_height_eff = min(min_height, adaptive_min_height)
    thr = otsu_threshold(row_sums)
    thr_relaxed = int(thr * alpha)
    positions = find_peaks_1d(row_sums, thr_relaxed)
    positions = enforce_min_height(positions, row_sums, min_height_eff)
    positions = enforce_min_distance(positions, min_dist)
    local_thr = local_otsu_thresholds(row_sums, window=window)
    positions_local = find_peaks_1d_local(row_sums, local_thr)
    positions_local = enforce_min_height(positions_local, row_sums, min_height_eff)
    positions_local = enforce_min_distance(positions_local, min_dist)

    return row_sums, thr, thr_relaxed, positions, local_thr, positions_local


def find_horizontal_lines_fixed(
    mask: np.ndarray,
    alpha: float = 0.7,
    window: int = 201,
    min_dist: int = 10,
    min_height: int = 50,
):
    """Detect horizontal lines with fixed thresholds for a column interval."""
    # Same logic as find_vertical_lines, but applied to row sums within a fixed column interval
    binary = (mask > 0).astype(np.uint8)
    row_sums = binary.sum(axis=1)
    thr = otsu_threshold(row_sums)
    thr_relaxed = int(thr * alpha)
    positions = find_peaks_1d(row_sums, thr_relaxed)
    positions = enforce_min_height(positions, row_sums, min_height)
    positions = enforce_min_distance(positions, min_dist)
    local_thr = local_otsu_thresholds(row_sums, window=window)
    positions_local = find_peaks_1d_local(row_sums, local_thr)
    positions_local = enforce_min_height(positions_local, row_sums, min_height)
    positions_local = enforce_min_distance(positions_local, min_dist)

    return row_sums, thr, thr_relaxed, positions, local_thr, positions_local


def max_run_length_1d(arr: np.ndarray) -> int:
    """Return the longest consecutive nonzero run in a 1D array."""
    max_run = 0
    cur = 0
    for v in arr:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def merge_close_lines(lines, merge_dist=10):
    """Merge nearby line positions into one representative center."""
    if not lines:
        return []
    lines = sorted(lines)
    merged = []
    cur = [lines[0]]
    for y in lines[1:]:
        if y - cur[-1] <= merge_dist:
            cur.append(y)
        else:
            merged.append(int(round(sum(cur) / len(cur))))
            cur = [y]
    merged.append(int(round(sum(cur) / len(cur))))
    return merged


def unify_positions(positions, merge_dist=10):
    """Merge positions and return both centers and original-to-center mapping."""
    if not positions:
        return [], {}
    positions = sorted({int(p) for p in positions})
    centers = merge_close_lines(positions, merge_dist=merge_dist)
    mapping = {}
    for p in positions:
        nearest = min(centers, key=lambda c: abs(c - p))
        mapping[p] = nearest if abs(nearest - p) <= merge_dist else p
    return centers, mapping


def merge_lines_by_run(rows_with_len, merge_dist=5):
    """Merge nearby rows while keeping the row with strongest run support."""
    if not rows_with_len:
        return []
    rows_with_len = sorted(rows_with_len, key=lambda x: x[0])
    merged = []
    group = [rows_with_len[0]]
    for y, run_len in rows_with_len[1:]:
        if y - group[-1][0] <= merge_dist:
            group.append((y, run_len))
        else:
            merged.append(max(group, key=lambda x: x[1])[0])
            group = [(y, run_len)]
    merged.append(max(group, key=lambda x: x[1])[0])
    return merged


def line_strength_vertical(mask: np.ndarray, x: int, half_window: int = 1) -> int:
    """Measure vertical-line support around an x coordinate."""
    h, w = mask.shape[:2]
    if w == 0:
        return 0
    x0 = max(0, x - half_window)
    x1 = min(w - 1, x + half_window)
    cols = (mask[:, x0:x1 + 1] > 0).astype(np.uint8)
    return int(cols.sum(axis=0).max()) if cols.size else 0


def line_strength_horizontal(mask: np.ndarray, y: int, half_window: int = 1) -> int:
    """Measure horizontal-line support around a y coordinate."""
    h, w = mask.shape[:2]
    if h == 0:
        return 0
    y0 = max(0, y - half_window)
    y1 = min(h - 1, y + half_window)
    rows = (mask[y0:y1 + 1, :] > 0).astype(np.uint8)
    return int(rows.sum(axis=1).max()) if rows.size else 0


def merge_positions_by_length(positions, length_fn, merge_dist=5):
    """Merge close positions and keep the candidate with greatest line strength."""
    if not positions:
        return []
    positions = sorted({int(p) for p in positions})
    merged = []
    group = [positions[0]]
    for p in positions[1:]:
        if p - group[-1] <= merge_dist:
            group.append(p)
        else:
            center = group[len(group) // 2]
            best = max(group, key=lambda v: (length_fn(v), -abs(v - center)))
            merged.append(best)
            group = [p]
    center = group[len(group) // 2]
    best = max(group, key=lambda v: (length_fn(v), -abs(v - center)))
    merged.append(best)
    return sorted(set(merged))


def snap_rows_to_reference(rows, ref_rows, merge_dist=5):
    """Snap row positions to a set of reference rows when close enough."""
    if not rows:
        return []
    if not ref_rows:
        return sorted(set(int(r) for r in rows))
    ref_rows = sorted(ref_rows)
    snapped = []
    for y in rows:
        nearest = min(ref_rows, key=lambda r: abs(r - y))
        if abs(nearest - y) <= merge_dist:
            snapped.append(nearest)
        else:
            snapped.append(int(y))
    return sorted(set(snapped))


def compute_column_rows(mask: np.ndarray, v_positions):
    """Compute horizontal row candidates separately within each vertical column."""
    h, w = mask.shape[:2]
    x_cuts = [0] + [p for p in v_positions if 0 < p < w] + [w]
    x_cuts = sorted(set(x_cuts))
    col_rows = {}
    col_long_rows = {}
    for i in range(len(x_cuts) - 1):
        x0, x1 = x_cuts[i], x_cuts[i + 1]
        col_mask = mask[:, x0:x1]
        row_sums, r_thr, r_thr_relaxed, r_pos, r_local_thr, r_pos_local = find_horizontal_lines_fixed(
            col_mask,
            alpha=0.1,
            window=101,
            min_dist=5,
            min_height=10
        )
        use_r = r_pos_local if r_pos_local else r_pos
        col_width = max(1, x1 - x0)
        run_thresh = int(col_width * 0.8)
        sum_thresh = int(col_width * 0.8)
        long_run_rows = []
        rows_with_len = []
        for y in range(1, h - 1):
            row_slice = (col_mask[y, :] > 0).astype(np.uint8)
            run_len = max_run_length_1d(row_slice)
            if run_len >= run_thresh or row_slice.sum() >= sum_thresh:
                long_run_rows.append(y)
            if y in use_r:
                rows_with_len.append((y, run_len))
        use_r = sorted(set(use_r + long_run_rows))
        # merge close rows within this column, keep the one with longer continuous run
        use_r = merge_lines_by_run(rows_with_len, merge_dist=5)
        col_rows[i] = [int(y) for y in use_r if 0 < y < h]
        col_long_rows[i] = merge_close_lines(
            [int(y) for y in long_run_rows if 0 < y < h], merge_dist=10
        )
    return col_rows, col_long_rows, x_cuts


def visualize(mask: np.ndarray, col_sums, thr, thr_relaxed, positions, local_thr, positions_local):
    """Visualize vertical-line projection diagnostics."""
    fig, (ax_img, ax_plot) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax_img.imshow(mask, cmap="gray")
    for x in positions:
        ax_img.axvline(x, color="red", linewidth=1)
    for x in positions_local:
        ax_img.axvline(x, color="lime", linewidth=1)
    ax_img.set_title("Detected separators (red=global relaxed, green=local)")
    ax_img.axis("off")

    ax_plot.plot(col_sums, color="black", linewidth=1, label="col_sums")
    ax_plot.axhline(thr, color="red", linestyle="--", label=f"Otsu thr={thr}")
    ax_plot.axhline(thr_relaxed, color="orange", linestyle="--", label=f"Relaxed thr={thr_relaxed}")
    ax_plot.plot(local_thr, color="lime", linestyle="--", linewidth=1, label="local Otsu")
    ax_plot.scatter(positions, [col_sums[x] for x in positions], color="red", s=15, label="global peaks")
    ax_plot.scatter(positions_local, [col_sums[x] for x in positions_local], color="lime", s=15, label="local peaks")
    ax_plot.set_title("Column sums")
    ax_plot.set_xlabel("Column index")
    ax_plot.set_ylabel("Sum")
    ax_plot.legend()

    plt.tight_layout()
    plt.show()


def visualize_grid(mask: np.ndarray, v_positions, h_positions, title: str):
    """Visualize a full grid using vertical and horizontal line positions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(mask, cmap="gray")
    for x in v_positions:
        ax.axvline(x, color="red", linewidth=1)
    for y in h_positions:
        ax.axhline(y, color="lime", linewidth=1)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_grid_segments(mask: np.ndarray, v_positions, col_rows, title: str, save_path: Optional[Path] = None):
    """Visualize per-column horizontal row candidates."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(mask, cmap="gray")
    h, w = mask.shape[:2]
    x_cuts = [0] + [p for p in v_positions if 0 < p < w] + [w]
    x_cuts = sorted(set(x_cuts))
    for x in v_positions:
        ax.axvline(x, color="red", linewidth=1)
    for i in range(len(x_cuts) - 1):
        x0, x1 = x_cuts[i], x_cuts[i + 1]
        rows = merge_close_lines(col_rows.get(i, []), merge_dist=5)
        for y in rows:
            ax.hlines(y, x0, x1, color="lime", linewidth=1)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_vertical_only(mask: np.ndarray, v_positions, title: str, save_path: Optional[Path] = None):
    """Save or show a visualization of detected vertical lines."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(mask, cmap="gray")
    for x in v_positions:
        ax.axvline(x, color="red", linewidth=1)
    ax.set_title(f"{title} - vertical lines")
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_horizontal_only(mask: np.ndarray, h_positions, title: str, save_path: Optional[Path] = None):
    """Save or show a visualization of detected horizontal lines."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(mask, cmap="gray")
    for y in h_positions:
        ax.axhline(y, color="lime", linewidth=1)
    ax.set_title(f"{title} - horizontal lines")
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def split_by_vertical_lines(mask: np.ndarray, positions, out_dir: Path, min_width: int = 10):
    """Save column crops split by detected vertical lines."""
    h, w = mask.shape[:2]
    cuts = [0] + [p for p in positions if 0 < p < w] + [w]
    cuts = sorted(set(cuts))
    for i in range(len(cuts) - 1):
        x0, x1 = cuts[i], cuts[i + 1]
        if x1 - x0 < min_width:
            continue
        crop = mask[:, x0:x1]
        out_path = out_dir / f"col_{i:02d}_x{x0:04d}_{x1:04d}.png"
        Image.fromarray(crop).save(out_path)


def build_grid_json(width, height, v_positions, h_positions):
    """Build a rectangular grid JSON from global vertical and horizontal cuts."""
    x_cuts = [0] + [p for p in v_positions if 0 < p < width] + [width]
    y_cuts = [0] + [p for p in h_positions if 0 < p < height] + [height]
    x_cuts = sorted(set(x_cuts))
    y_cuts = sorted(set(y_cuts))

    cells = []
    for r in range(len(y_cuts) - 1):
        for c in range(len(x_cuts) - 1):
            x0, x1 = x_cuts[c], x_cuts[c + 1]
            y0, y1 = y_cuts[r], y_cuts[r + 1]
            cells.append({
                "row": r,
                "col": c,
                "bbox": [int(x0), int(y0), int(x1), int(y1)]
            })

    data = {
        "imageWidth": int(width),
        "imageHeight": int(height),
        "vertical_lines": [int(x) for x in x_cuts],
        "horizontal_lines": [int(y) for y in y_cuts],
        "cells": cells,
    }
    return data


def merge_rows_across_columns(col_rows, merge_dist=5):
    """Merge row candidates across columns and keep per-column assignments."""
    # Build global clusters from all row positions, then reassign per-column.
    all_rows = sorted({y for rows in col_rows.values() for y in rows})
    if not all_rows:
        return [], {k: [] for k in col_rows}

    clusters = []
    cur = [all_rows[0]]
    for y in all_rows[1:]:
        if y - cur[-1] <= merge_dist:
            cur.append(y)
        else:
            clusters.append(int(round(sum(cur) / len(cur))))
            cur = [y]
    clusters.append(int(round(sum(cur) / len(cur))))

    merged_col_rows = {}
    for col_idx, rows in col_rows.items():
        merged = []
        for y in rows:
            nearest = min(clusters, key=lambda c: abs(c - y))
            if abs(nearest - y) <= merge_dist:
                merged.append(nearest)
            else:
                merged.append(y)
        merged_col_rows[col_idx] = sorted(set(merged))

    return clusters, merged_col_rows


def filter_rows_by_support(merged_rows, col_rows, min_support):
    """Keep only row positions supported by enough columns."""
    if not merged_rows:
        return [], {k: [] for k in col_rows}
    support = {r: 0 for r in merged_rows}
    for rows in col_rows.values():
        for r in rows:
            if r in support:
                support[r] += 1
    kept = [r for r in merged_rows if support.get(r, 0) >= min_support]
    kept_set = set(kept)
    filtered_col_rows = {k: [r for r in rows if r in kept_set] for k, rows in col_rows.items()}
    return kept, filtered_col_rows


def main():
    """Generate grid JSON files from predicted line masks."""
    parser = argparse.ArgumentParser(description="Generate grid JSON files from predicted line masks.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing predicted line-mask images.")
    parser.add_argument("--out-dir", required=True, help="Directory for grid JSON output.")
    parser.add_argument(
        "--name-suffix",
        default="_OUT",
        help="Suffix added to each output folder stem so merge code can find <image>_OUT/grid.json. Use '' to disable.",
    )
    args = parser.parse_args()

    folder = Path(args.mask_dir)
    out_root = Path(args.out_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Mask directory not found: {folder}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    files.sort()

    for path in files:
        mask = np.array(Image.open(path).convert("L"))
        col_sums, thr, thr_relaxed, positions, local_thr, positions_local = find_vertical_lines(mask)
        print(f"[{path.name}] Otsu threshold: {thr}, relaxed: {thr_relaxed}, local window: 101, min_dist: 5, min_height: 50")
        print(f"[{path.name}] Global peaks: {positions}")
        print(f"[{path.name}] Local peaks: {positions_local}")
        # visualize(mask, col_sums, thr, thr_relaxed, positions, local_thr, positions_local)

        use_positions = positions_local if positions_local else positions
        # postprocess: merge close vertical lines by keeping the longest one
        use_positions = merge_positions_by_length(
            use_positions,
            lambda x: line_strength_vertical(mask, x),
            merge_dist=5
        )
        out_stem = path.stem
        if args.name_suffix and not out_stem.endswith(args.name_suffix):
            out_stem = out_stem + args.name_suffix
        out_dir = out_root / out_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        split_by_vertical_lines(mask, use_positions, out_dir, min_width=10)

        # Row separators per column (local to each column)
        h, w = mask.shape[:2]
        col_rows, col_long_rows, x_cuts = compute_column_rows(mask, use_positions)

        # merge close horizontal lines per-column (no global clustering)
        merged_col_rows = {}
        for i in range(len(x_cuts) - 1):
            x0, x1 = x_cuts[i], x_cuts[i + 1]
            col_mask = mask[:, x0:x1]
            rows = col_rows.get(i, [])
            rows = merge_close_lines(rows, merge_dist=5)
            rows = merge_positions_by_length(
                rows,
                lambda y, m=col_mask: line_strength_horizontal(m, y),
                merge_dist=5
            )
            long_rows = col_long_rows.get(i, [])
            rows = sorted(set(rows + long_rows))
            rows = merge_close_lines(rows, merge_dist=5)
            merged_col_rows[i] = rows

        # build merged_rows as union of per-column results (for grid_json)
        merged_rows = sorted({y for rows in merged_col_rows.values() for y in rows})

        # unify all horizontal lines across columns (distance <= 5)
        unified_rows, _ = unify_positions(merged_rows, merge_dist=5)
        unified_col_rows = {}
        for k, rows in merged_col_rows.items():
            unified = snap_rows_to_reference(rows, unified_rows, merge_dist=5)
            unified_col_rows[k] = merge_close_lines(unified, merge_dist=5)

        grid_json = build_grid_json(w, h, use_positions, unified_rows)
        grid_json["column_rows"] = {str(k): v for k, v in unified_col_rows.items()}
        json_path = out_dir / "grid.json"
        json_path.write_text(json.dumps(grid_json, ensure_ascii=False, indent=2), encoding="utf-8")

        # visualize grid on mask (no save)
        visualize_vertical_only(mask, use_positions, title=path.name, save_path=out_dir / "vis_vertical.png")
        visualize_horizontal_only(mask, unified_rows, title=path.name, save_path=out_dir / "vis_horizontal.png")
        visualize_grid_segments(mask, use_positions, unified_col_rows, title=path.name, save_path=out_dir / "vis_grid.png")


if __name__ == "__main__":
    main()
