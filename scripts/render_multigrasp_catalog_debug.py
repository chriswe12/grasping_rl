#!/usr/bin/env python3
"""Render a goal-image contact sheet and reset-path diagnostic for the catalog."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_catalog.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/multigrasp_50_debug",
    )
    return parser.parse_args()


def _contact_sheet(arrays: dict[str, np.ndarray], output: Path) -> None:
    rgb = arrays["goal_rgb"]
    orientation_names = arrays["orientation_names"].astype(str).tolist()
    orientation_indices = arrays["orientation_indices"]
    target_ids = arrays["target_ids"].astype(str)
    tile_width = int(rgb.shape[2])
    tile_height = int(rgb.shape[1]) + 24
    columns = max(int(np.sum(orientation_indices == index)) for index in range(len(orientation_names)))
    sheet = Image.new(
        "RGB", (columns * tile_width, len(orientation_names) * tile_height), (15, 15, 15)
    )
    draw = ImageDraw.Draw(sheet)
    for row, orientation_name in enumerate(orientation_names):
        indices = np.flatnonzero(orientation_indices == row)
        for column, target_index in enumerate(indices):
            x = column * tile_width
            y = row * tile_height
            sheet.paste(Image.fromarray(rgb[target_index]), (x, y))
            label = f"{target_index:02d} {target_ids[target_index]}"
            draw.text((x + 4, y + rgb.shape[1] + 5), label, fill=(235, 235, 235))
        draw.text(
            (4, row * tile_height + 4),
            orientation_name,
            fill=(255, 210, 40),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    # Keep diagnostic pixels lossless. JPEG ringing and block boundaries can
    # otherwise be mistaken for artifacts in the stored training observation.
    sheet.save(output, format="PNG")


def _path_plot(arrays: dict[str, np.ndarray], output: Path) -> None:
    trajectories = arrays["reset_joint_trajectories"]
    orientation_names = arrays["orientation_names"].astype(str).tolist()
    orientation_indices = arrays["orientation_indices"]
    grasp_ids = arrays["grasp_ids"].astype(str)
    graph_width = 1200
    graph_height = 250
    margin_left, margin_right, margin_top, margin_bottom = 75, 20, 45, 32
    image = Image.new(
        "RGB", (graph_width, graph_height * len(orientation_names)), (250, 250, 250)
    )
    draw = ImageDraw.Draw(image)
    for orientation_index, orientation_name in enumerate(orientation_names):
        y_offset = orientation_index * graph_height
        indices = np.flatnonzero(orientation_indices == orientation_index)
        curves = []
        for target_index in indices:
            delta = trajectories[target_index] - trajectories[target_index, 0]
            joint_distance = np.linalg.norm(delta, axis=1)
            curves.append((target_index, joint_distance))
        y_max = max(float(curve.max()) for _, curve in curves) * 1.05
        y_max = max(y_max, 1.0e-6)
        x0, x1 = margin_left, graph_width - margin_right
        y0 = y_offset + graph_height - margin_bottom
        y1 = y_offset + margin_top
        draw.rectangle((x0, y1, x1, y0), outline=(70, 70, 70), width=1)
        for fraction in (0.25, 0.5, 0.75):
            y = round(y0 - fraction * (y0 - y1))
            draw.line((x0, y, x1, y), fill=(215, 215, 215), width=1)
        draw.text((5, y_offset + 5), orientation_name, fill=(10, 10, 10))
        draw.text((5, y1), f"{y_max:.2f} rad", fill=(70, 70, 70))
        draw.text((x0, y0 + 8), "p=0", fill=(70, 70, 70))
        draw.text((x1 - 24, y0 + 8), "p=1", fill=(70, 70, 70))
        for curve_index, (target_index, curve) in enumerate(curves):
            rgb_float = colorsys.hsv_to_rgb(curve_index / max(len(curves), 1), 0.75, 0.75)
            color = tuple(round(channel * 255) for channel in rgb_float)
            points = [
                (
                    round(x0 + point_index / (len(curve) - 1) * (x1 - x0)),
                    round(y0 - float(value) / y_max * (y0 - y1)),
                )
                for point_index, value in enumerate(curve)
            ]
            draw.line(points, fill=color, width=2)
            legend_x = x0 + curve_index * ((x1 - x0) // len(curves))
            draw.text((legend_x, y_offset + 22), grasp_ids[target_index], fill=color)
    image.save(output)


def main() -> None:
    args = parse_args()
    with np.load(args.catalog, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = args.output_dir / "goal_rgb_contact_sheet.png"
    legacy_contact_sheet = args.output_dir / "goal_rgb_contact_sheet.jpg"
    path_plot = args.output_dir / "reset_joint_paths.png"
    report_path = args.output_dir / "catalog_report.json"
    _contact_sheet(arrays, contact_sheet)
    # Older catalog runs wrote a JPEG beside the lossless PNG.  Remove it so
    # viewers and users cannot accidentally inspect a stale goal-image sheet.
    legacy_contact_sheet.unlink(missing_ok=True)
    _path_plot(arrays, path_plot)
    report = {
        "catalog": str(args.catalog.resolve()),
        "target_count": int(len(arrays["target_ids"])),
        "orientation_counts": {
            str(name): int(np.sum(arrays["orientation_ids"].astype(str) == str(name)))
            for name in arrays["orientation_names"]
        },
        "reset_path_shape": list(arrays["reset_joint_trajectories"].shape),
        "goal_rgb_shape": list(arrays["goal_rgb"].shape),
        "goal_depth_shape": list(arrays["goal_depth"].shape),
        "worst_reset_path_ik_position_error_mm": float(
            np.max(arrays["reset_path_max_position_error_m"]) * 1000.0
        ),
        "worst_reset_path_ik_rotation_error_deg": float(
            np.degrees(np.max(arrays["reset_path_max_rotation_error_rad"]))
        ),
        "worst_isaac_capture_position_error_mm": float(
            np.max(arrays["goal_tcp_capture_position_error_m"]) * 1000.0
        ),
        "worst_isaac_capture_rotation_error_deg": float(
            np.degrees(np.max(arrays["goal_tcp_capture_rotation_error_rad"]))
        ),
        "contact_sheet": str(contact_sheet.resolve()),
        "path_plot": str(path_plot.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] {contact_sheet}")
    print(f"[DONE] {path_plot}")
    print(f"[DONE] {report_path}")


if __name__ == "__main__":
    main()
