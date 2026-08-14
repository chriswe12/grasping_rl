#!/usr/bin/env python3
"""Build, MoveIt-validate, and Isaac-render the 50-target RL catalog."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.catalog_capture_validation import (  # noqa: E402
    catalog_file_signature,
    validate_fresh_goal_catalog_capture,
)

PATHS_ASSET = REPO_ROOT / "isaac_rl/data/multigrasp_50_paths.npz"
GOAL_CATALOG = REPO_ROOT / "isaac_rl/data/multigrasp_50_catalog.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaaclab",
        type=Path,
        default=Path("/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh"),
    )
    parser.add_argument(
        "--force-replan",
        action="store_true",
        help="Ignore reusable per-target MoveIt plan JSON files.",
    )
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="Reuse the completed planned manifest and per-target plans.",
    )
    parser.add_argument(
        "--skip-isaac-capture",
        action="store_true",
        help="Stop after generating the 50 straight Cartesian reset paths.",
    )
    parser.add_argument(
        "--maximum-quality-replans",
        type=int,
        default=3,
        help="Maximum MoveIt/Isaac replacement rounds for low-information goal views.",
    )
    parser.add_argument(
        "--no-start-mock-moveit",
        action="store_true",
        help="Use an already-running MoveIt server instead of starting the repo mock stack.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"\n[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    _run([python, "isaac_rl/scripts/build_multigrasp_manifest.py"])
    isaaclab = args.isaaclab.expanduser().resolve()
    if not isaaclab.is_file():
        raise FileNotFoundError(isaaclab)
    moveit_process = None
    if not args.skip_planning and not args.no_start_mock_moveit:
        print("[START] Launching the repo-local mock iiwa7 MoveIt stack.", flush=True)
        moveit_process = subprocess.Popen(
            [str(REPO_ROOT / "start_lbr_moveit.sh")], cwd=REPO_ROOT
        )
    try:
        for replacement_round in range(args.maximum_quality_replans + 1):
            if not args.skip_planning or replacement_round > 0:
                plan_command = [python, "isaac_rl/scripts/plan_multigrasp_targets.py"]
                if args.force_replan:
                    plan_command.append("--no-resume")
                _run(plan_command)
            _run([python, "isaac_rl/scripts/build_multigrasp_path_asset.py"])
            _run(
                [python, "isaac_rl/scripts/build_multigrasp_rotation_reset_asset.py"]
            )
            if args.skip_isaac_capture:
                print(
                    "[STOP] Paths are ready, but training remains intentionally disabled until "
                    "the Isaac goal capture completes.",
                    flush=True,
                )
                return
            try:
                previous_catalog_signature = catalog_file_signature(GOAL_CATALOG)
                _run(
                    [
                        str(isaaclab),
                        "-p",
                        "isaac_rl/scripts/capture_multigrasp_goal_catalog.py",
                        "--headless",
                    ]
                )
                target_count = validate_fresh_goal_catalog_capture(
                    GOAL_CATALOG,
                    PATHS_ASSET,
                    previous_signature=previous_catalog_signature,
                )
                print(
                    f"[VALIDATE] Fresh Isaac goal catalog contains {target_count} "
                    "complete targets under the active visual profiles.",
                    flush=True,
                )
                break
            except subprocess.CalledProcessError:
                if args.skip_planning or replacement_round >= args.maximum_quality_replans:
                    raise
                print(
                    f"[RETRY] Isaac rejected goal images; replanning replacements "
                    f"({replacement_round + 1}/{args.maximum_quality_replans}).",
                    flush=True,
                )
    finally:
        if moveit_process is not None and moveit_process.poll() is None:
            print("[STOP] Stopping the repo-local mock MoveIt stack.", flush=True)
            moveit_process.send_signal(signal.SIGINT)
            try:
                moveit_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                moveit_process.terminate()
                moveit_process.wait(timeout=5.0)
    print(
        "\n[DONE] The validated 50-target catalog is active. Start a fresh run with:\n"
        "/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p "
        "isaac_rl/scripts/rl_games/train.py --task Grasp-Visual-Servo-RGBD-Direct-v0 "
        "--num_envs 64 --max_iterations 5000 --headless --enable_cameras",
        flush=True,
    )


if __name__ == "__main__":
    main()
