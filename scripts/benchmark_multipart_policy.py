#!/usr/bin/env python3
"""Benchmark the latest multi-part policy and record representative videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAACLAB = Path("/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh")
RUNS_ROOT = REPO_ROOT / "logs/rl_games/grasp_visual_servo_rgbd_multipart"
EULER_RUNS_ROOT = REPO_ROOT / "logs/euler/rl_games/grasp_visual_servo_rgbd_multipart"
CATALOG = REPO_ROOT / "isaac_rl/data/plumbers_block/goal_catalog.npz"
TASK = "Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0"


def _latest_checkpoint() -> Path:
    selection_files = sorted(
        (
            *RUNS_ROOT.glob("*/evaluations/periodic_validation/best_checkpoint.txt"),
            *EULER_RUNS_ROOT.glob("*/evaluations/periodic_validation/best_checkpoint.txt"),
        ),
        key=lambda path: path.stat().st_mtime,
    )
    for selection_file in reversed(selection_files):
        selection_lines = selection_file.read_text(encoding="utf-8").splitlines()
        if not selection_lines:
            continue
        selected = Path(selection_lines[0])
        if selected.is_file():
            return selected.resolve()

    candidates = sorted(
        (
            *RUNS_ROOT.glob("*/nn/grasp_visual_servo_rgbd_multipart.pth"),
            *EULER_RUNS_ROOT.glob("*/nn/grasp_visual_servo_rgbd_multipart.pth"),
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            (*RUNS_ROOT.glob("*/nn/*.pth"), *EULER_RUNS_ROOT.glob("*/nn/*.pth")),
            key=lambda path: path.stat().st_mtime,
        )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found below {RUNS_ROOT}")
    return candidates[-1].resolve()


def _representative_split_indices(split: str, samples_per_part: int) -> tuple[list[int], list[str]]:
    """Return evenly spaced split-local target indices for every part."""
    with np.load(CATALOG, allow_pickle=False) as source:
        split_mask = source["split_ids"].astype(str) == split
        split_parts = source["part_ids"].astype(str)[split_mask]
    indices: list[int] = []
    parts: list[str] = []
    for part in dict.fromkeys(split_parts.tolist()):
        matches = np.flatnonzero(split_parts == part)
        sample_positions = np.linspace(0, len(matches) - 1, num=min(samples_per_part, len(matches)), dtype=int)
        for sample_position in sample_positions:
            indices.append(int(matches[sample_position]))
            parts.append(part)
    if not indices:
        raise ValueError(f"Catalog split {split!r} has no targets")
    return indices, parts


def _run(command: list[str]) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _write_combined_report(output_dir: Path, checkpoint: Path, splits: list[str], video_split: str) -> None:
    payload: dict[str, object] = {"checkpoint": str(checkpoint), "splits": {}}
    lines = ["# Multi-part policy benchmark", "", f"Checkpoint: `{checkpoint}`", ""]
    for split in splits:
        summary_path = output_dir / split / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["splits"][split] = summary
        lines.extend(
            [
                f"## {split.title()}",
                "",
                "| Condition | Attempts | Success | Spawn offset | Initial error | Final error | Best reached |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, metrics in summary["conditions"].items():
            lines.append(
                f"| {condition} | {metrics['attempts']} "
                f"| {100.0 * metrics['success_rate']:.1f}% "
                f"| {metrics['reset_position_offset_mm_mean']:.2f} mm "
                f"| {metrics['initial_position_error_mm_mean']:.2f} mm / "
                f"{metrics['initial_rotation_error_deg_mean']:.2f} deg "
                f"| {metrics['final_position_error_mm_mean']:.2f} mm / "
                f"{metrics['final_rotation_error_deg_mean']:.2f} deg "
                f"| {metrics['best_position_error_mm_mean']:.2f} mm / "
                f"{metrics['best_rotation_error_deg_mean']:.2f} deg |"
            )
        lines.append("")
    videos = output_dir / f"videos_{video_split}" / "debug_video_metrics.json"
    if videos.exists():
        payload["videos"] = json.loads(videos.read_text(encoding="utf-8"))
        lines.extend(["## Debug videos", ""])
        for episode in payload["videos"]["episodes"]:
            lines.append(
                f"- `{episode['part_id']}` / `{episode['condition']}`: "
                f"{episode['initial_position_error_mm']:.2f} -> "
                f"{episode['final_position_error_mm']:.2f} mm; "
                f"[{Path(episode['video']).name}]({episode['video']})"
            )
    (output_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--isaaclab", type=Path, default=DEFAULT_ISAACLAB)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("far", "mid", "close", "nominal", "stress"),
        default=("far", "mid", "close"),
    )
    parser.add_argument("--runs_per_target", type=int, default=1)
    parser.add_argument("--episode_seconds", type=float, default=15.0)
    parser.add_argument("--rotation_deg", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--video_split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--videos_per_part", type=int, default=1)
    parser.add_argument(
        "--video_count",
        type=int,
        default=None,
        help="Maximum policy-rollout videos, excluding the exact-goal reference.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--skip_videos", action="store_true")
    args = parser.parse_args()
    if args.videos_per_part < 1:
        parser.error("--videos_per_part must be at least one")
    if args.video_count is not None and args.video_count < 1:
        parser.error("--video_count must be at least one")

    checkpoint = (args.checkpoint or _latest_checkpoint()).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not args.isaaclab.is_file():
        raise FileNotFoundError(args.isaaclab)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (args.output_dir or checkpoint.parent.parent / "evaluations" / f"multipart_full_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BENCHMARK] checkpoint={checkpoint}", flush=True)
    print(f"[BENCHMARK] output={output_dir}", flush=True)

    if not args.skip_benchmark:
        for split in args.splits:
            _run(
                [
                    str(args.isaaclab),
                    "-p",
                    "isaac_rl/scripts/rl_games/evaluate_multigrasp.py",
                    "--task",
                    TASK,
                    "--checkpoint",
                    str(checkpoint),
                    "--catalog_split",
                    split,
                    "--runs_per_target",
                    str(args.runs_per_target),
                    "--episode_seconds",
                    str(args.episode_seconds),
                    "--rotation_deg",
                    str(args.rotation_deg),
                    "--conditions",
                    *args.conditions,
                    "--seed",
                    str(args.seed),
                    "--output_dir",
                    str(output_dir / split),
                    "--headless",
                ]
            )

    if not args.skip_videos:
        target_indices, parts = _representative_split_indices(args.video_split, args.videos_per_part)
        if args.video_count is not None:
            grouped = {
                part: [
                    index for candidate_part, index in zip(parts, target_indices, strict=True) if candidate_part == part
                ]
                for part in dict.fromkeys(parts)
            }
            selections = [
                (part, grouped[part][rank])
                for rank in range(args.videos_per_part)
                for part in grouped
                if rank < len(grouped[part])
            ][: args.video_count]
            parts = [part for part, _ in selections]
            target_indices = [index for _, index in selections]
        # Far-start rollouts spanning every part, plus a perfect-goal reference.
        video_conditions = ["far"] * len(target_indices) + ["exact"]
        video_targets = target_indices + [target_indices[0]]
        selections = list(zip(parts, target_indices, strict=True))
        print(f"[VIDEOS] representative {args.video_split} targets: {selections}")
        _run(
            [
                str(args.isaaclab),
                "-p",
                "isaac_rl/scripts/rl_games/record_debug_videos.py",
                "--task",
                TASK,
                "--checkpoint",
                str(checkpoint),
                "--catalog_split",
                args.video_split,
                "--episode_seconds",
                str(args.episode_seconds),
                "--seed",
                str(args.seed + 763),
                "--conditions",
                *video_conditions,
                "--target_indices",
                *(str(index) for index in video_targets),
                "--output_dir",
                str(output_dir / f"videos_{args.video_split}"),
                "--headless",
            ]
        )

    _write_combined_report(output_dir, checkpoint, list(args.splits), args.video_split)
    print(f"[DONE] Combined report: {output_dir / 'benchmark_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
