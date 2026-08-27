#!/usr/bin/env python3
"""Prepare the five-part plumbers-block RL catalog in explicit resumable stages."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.catalog_capture_validation import (  # noqa: E402
    catalog_file_signature,
    validate_fresh_goal_catalog_capture,
)

DATA_ROOT = REPO_ROOT / "isaac_rl/data/plumbers_block"
DEBUG_ROOT = REPO_ROOT / "artifacts/plumbers_block_catalog_debug"
PART_IDS = ("0", "1", "2", "3", "4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "sources",
            "manifest",
            "plan",
            "paths",
            "rotation",
            "mujoco",
            "isaac",
            "finalize",
            "cpu",
        ),
        default="cpu",
        help=(
            "Run one stage, all CPU stages, the deferred MuJoCo goal-render "
            "stage, or finalize the passing subset from a failed capture. "
            "Every stage is resumable."
        ),
    )
    parser.add_argument("--force-sources", action="store_true")
    parser.add_argument("--force-replan", action="store_true")
    parser.add_argument("--targets-per-part-orientation", type=int, default=64)
    parser.add_argument("--alternates-per-part-orientation", type=int, default=256)
    parser.add_argument("--ik-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--isaaclab",
        type=Path,
        default=Path("/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh"),
    )
    parser.add_argument(
        "--no-start-mock-moveit",
        action="store_true",
        help="Use an already-running MoveIt server for the plan stage.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"\n[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as descriptor:
        temporary = Path(descriptor.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _subset_target_aligned_arrays(
    arrays: dict[str, np.ndarray], indices: np.ndarray, target_count: int
) -> dict[str, np.ndarray]:
    """Subset per-target rows while preserving catalog-level arrays."""

    return {
        name: value[indices]
        if value.ndim > 0 and value.shape[0] == target_count
        else value
        for name, value in arrays.items()
    }


def _write_target_subset_paths_asset(
    source_path: Path,
    output_path: Path,
    selected_target_ids: np.ndarray,
) -> int:
    """Write path rows in the exact order required by the rotation-reset asset."""

    selected_ids = np.asarray(selected_target_ids).astype(str)
    if selected_ids.ndim != 1 or selected_ids.size < 1:
        raise ValueError("Selected target_ids must be a non-empty vector.")
    if len(set(selected_ids.tolist())) != len(selected_ids):
        raise ValueError("Selected target_ids must be unique.")
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    source_ids = np.asarray(arrays["target_ids"]).astype(str)
    source_index = {target_id: index for index, target_id in enumerate(source_ids)}
    missing = [target_id for target_id in selected_ids if target_id not in source_index]
    if missing:
        raise ValueError(
            f"Rotation-reset asset contains {len(missing)} targets absent from the "
            f"path asset: {missing[:8]}."
        )
    indices = np.asarray(
        [source_index[target_id] for target_id in selected_ids], dtype=np.int64
    )
    source_target_count = len(source_ids)
    payload = _subset_target_aligned_arrays(arrays, indices, source_target_count)
    np.savez_compressed(output_path, **payload)
    return int(len(selected_ids))


def _finalize_failed_isaac_capture(data_root: Path = DATA_ROOT) -> int:
    """Promote only fully validated diagnostic captures and matching reset rows."""

    diagnostic = data_root / "goal_catalog_failed_validation.npz"
    goal_catalog = data_root / "goal_catalog.npz"
    rotation_reset_asset = data_root / "rotation_resets.npz"
    paths_asset = data_root / "paths.npz"
    for required in (diagnostic, rotation_reset_asset, paths_asset):
        if not required.is_file():
            raise FileNotFoundError(required)

    with np.load(diagnostic, allow_pickle=False) as source:
        diagnostic_arrays = {
            name: source[name].copy() for name in source.files
        }
    diagnostic_target_ids = diagnostic_arrays["target_ids"].astype(str)
    target_count = len(diagnostic_target_ids)
    for name in ("capture_validation_passed", "isaac_goal_rgbd_captured"):
        value = diagnostic_arrays.get(name)
        if value is None or value.shape != (target_count,) or value.dtype != np.bool_:
            raise ValueError(
                f"Diagnostic array '{name}' must be a {target_count}-row boolean vector."
            )
    passing = (
        diagnostic_arrays["capture_validation_passed"]
        & diagnostic_arrays["isaac_goal_rgbd_captured"]
    )
    passing_indices = np.flatnonzero(passing).astype(np.int64)
    if passing_indices.size < 1:
        raise RuntimeError("Isaac diagnostic capture contains no passing targets.")
    passing_target_ids = diagnostic_target_ids[passing_indices]

    with np.load(rotation_reset_asset, allow_pickle=False) as source:
        rotation_arrays = {name: source[name].copy() for name in source.files}
    rotation_target_ids = rotation_arrays["target_ids"].astype(str)
    if np.array_equal(rotation_target_ids, passing_target_ids):
        with np.load(goal_catalog, allow_pickle=False) as source:
            current_goal_ids = source["target_ids"].astype(str)
        if not np.array_equal(current_goal_ids, passing_target_ids):
            raise ValueError(
                "Rotation resets already contain the passing subset, but the goal "
                "catalog target_ids do not match it."
            )
        print(
            f"[REUSE] The finalized Isaac catalog already contains "
            f"{len(passing_target_ids)} targets.",
            flush=True,
        )
        return int(len(passing_target_ids))
    if not np.array_equal(rotation_target_ids, diagnostic_target_ids):
        raise ValueError(
            "Diagnostic capture target_ids do not exactly match rotation_resets.npz; "
            "refusing to create misaligned training assets."
        )

    goal_payload = _subset_target_aligned_arrays(
        diagnostic_arrays, passing_indices, target_count
    )
    rotation_payload = _subset_target_aligned_arrays(
        rotation_arrays, passing_indices, target_count
    )
    goal_payload["capture_validation_passed"] = np.ones(
        len(passing_indices), dtype=np.bool_
    )
    goal_payload["isaac_goal_rgbd_captured"] = np.ones(
        len(passing_indices), dtype=np.bool_
    )
    previous_catalog_signature = catalog_file_signature(goal_catalog)
    _atomic_savez(rotation_reset_asset, rotation_payload)
    _atomic_savez(goal_catalog, goal_payload)

    with tempfile.NamedTemporaryFile(
        prefix=".finalized_goal_paths-",
        suffix=".npz",
        dir=data_root,
        delete=False,
    ) as descriptor:
        filtered_paths = Path(descriptor.name)
    try:
        _write_target_subset_paths_asset(
            paths_asset,
            filtered_paths,
            passing_target_ids,
        )
        finalized_count = validate_fresh_goal_catalog_capture(
            goal_catalog,
            filtered_paths,
            previous_signature=previous_catalog_signature,
        )
    finally:
        filtered_paths.unlink(missing_ok=True)
    with np.load(rotation_reset_asset, allow_pickle=False) as source:
        finalized_rotation_ids = source["target_ids"].astype(str)
    if not np.array_equal(finalized_rotation_ids, passing_target_ids):
        raise RuntimeError("Finalized rotation-reset target_ids changed unexpectedly.")
    print(
        f"[FINALIZE] Promoted {finalized_count}/{target_count} validated goal "
        f"captures and removed {target_count - finalized_count} failing targets "
        "from both the goal catalog and rotation-reset asset.",
        flush=True,
    )
    return finalized_count


def _source_paths(part_id: str) -> tuple[Path, Path]:
    source_dir = DATA_ROOT / "sources"
    return (
        source_dir / f"part_{part_id}_stage1.json",
        source_dir / f"part_{part_id}_stage2.json",
    )


def _generate_sources(*, force: bool) -> None:
    base_path = REPO_ROOT / "configs/grasp_pipeline_sim_isaac.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for part_id in PART_IDS:
        stage1, stage2 = _source_paths(part_id)
        if not force and stage1.is_file() and stage2.is_file():
            print(f"[REUSE] part {part_id} planning sources already exist.", flush=True)
            continue
        payload = yaml.safe_load(yaml.safe_dump(base))
        payload["geometry"]["target_mesh_path"] = (
            f"obj/fabrica/plumbers_block/{part_id}.obj"
        )
        payload["planning"].update(
            {
                "num_surface_samples": 1024,
                "min_jaw_width": 0.012,
                "max_jaw_width": 0.062,
                "detailed_finger_contact_gap_m": 0.005,
                "gripper_collision_model": "pdz_gripper",
            }
        )
        debug_dir = DATA_ROOT / "sources/debug"
        payload["artifacts"] = {
            "stage1_json": str(stage1),
            "stage1_html": str(debug_dir / f"part_{part_id}_stage1.html"),
            "stage2_json": str(stage2),
            "stage2_html": str(debug_dir / f"part_{part_id}_stage2.html"),
            "part_frame_html": str(debug_dir / f"part_{part_id}_frame.html"),
        }
        payload["mujoco_execution"]["enabled"] = False
        payload["isaac_execution"]["enabled"] = False
        payload["isaac_execution"].update(
            {
                "fr3_usd": "assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd",
                "moveit_pose_link": "pdz_gripper_tcp",
                "gripper_width_clearance": 0.01,
                "contact_gap_m": 0.005,
            }
        )
        payload["ros2"]["part_id"] = int(part_id)
        stage1.parent.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_plumbers_part_{part_id}.yaml",
            encoding="utf-8",
            delete=False,
        ) as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            config_path = Path(stream.name)
        try:
            _run(
                [
                    sys.executable,
                    "scripts/run_grasp_pipeline.py",
                    "--mode",
                    "sim",
                    "--config",
                    str(config_path),
                    "--backend",
                    "none",
                    "--headless",
                ]
            )
        finally:
            config_path.unlink(missing_ok=True)


def _build_manifest(
    targets_per_part_orientation: int, alternates_per_part_orientation: int
) -> None:
    _run(
        [
            sys.executable,
            "isaac_rl/scripts/build_assembly_multigrasp_manifest.py",
            "--targets-per-part-orientation",
            str(targets_per_part_orientation),
            "--alternates-per-part-orientation",
            str(alternates_per_part_orientation),
        ]
    )


def _plan(*, force: bool, start_mock_moveit: bool, ik_timeout_s: float) -> None:
    moveit_process: subprocess.Popen | None = None
    if start_mock_moveit:
        print("[START] Launching the repo-local mock iiwa7 MoveIt stack.", flush=True)
        moveit_process = subprocess.Popen(
            [str(REPO_ROOT / "start_lbr_moveit.sh")], cwd=REPO_ROOT
        )
    try:
        command = [
            sys.executable,
            "isaac_rl/scripts/plan_multigrasp_targets.py",
            "--manifest",
            str(DATA_ROOT / "assembly_manifest.json"),
            "--output-manifest",
            str(DATA_ROOT / "planned_manifest.json"),
            "--plans-dir",
            str(DATA_ROOT / "plans"),
            "--exclusions-file",
            str(DATA_ROOT / "goal_exclusions.json"),
            "--targets-per-orientation",
            "0",
            "--allow-variable-counts",
            "--minimum-targets-per-orientation",
            "1",
            "--ik-timeout-s",
            str(ik_timeout_s),
            "--quiet-cached",
        ]
        if force:
            command.append("--no-resume")
        _run(command)
    finally:
        if moveit_process is not None and moveit_process.poll() is None:
            print("[STOP] Stopping the repo-local mock MoveIt stack.", flush=True)
            moveit_process.send_signal(signal.SIGINT)
            try:
                moveit_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                moveit_process.terminate()
                moveit_process.wait(timeout=5.0)


def _build_paths() -> None:
    _run(
        [
            sys.executable,
            "isaac_rl/scripts/build_multigrasp_path_asset.py",
            "--manifest",
            str(DATA_ROOT / "planned_manifest.json"),
            "--output",
            str(DATA_ROOT / "paths.npz"),
            "--quiet",
            "--filtered-manifest-output",
            str(DATA_ROOT / "planned_manifest.json"),
        ]
    )


def _build_rotation_resets() -> None:
    _run(
        [
            sys.executable,
            "isaac_rl/scripts/build_multigrasp_rotation_reset_asset.py",
            "--paths-asset",
            str(DATA_ROOT / "paths.npz"),
            "--output",
            str(DATA_ROOT / "rotation_resets.npz"),
            "--cache-dir",
            str(DATA_ROOT / "rotation_cache"),
            "--variants",
            "8",
            "--minimum-distinct-variants",
            "4",
            "--quiet-cached",
            "--quiet",
        ]
    )


def _render_debug_artifacts() -> None:
    """Refresh diagnostics from the active, finalized goal catalog."""

    _run(
        [
            sys.executable,
            "isaac_rl/scripts/render_multigrasp_catalog_debug.py",
            "--catalog",
            str(DATA_ROOT / "goal_catalog.npz"),
            "--output-dir",
            str(DEBUG_ROOT),
        ]
    )


def _run_mujoco_stage() -> None:
    goal_catalog = DATA_ROOT / "goal_catalog.npz"
    paths_asset = DATA_ROOT / "paths.npz"
    rotation_reset_asset = DATA_ROOT / "rotation_resets.npz"
    capture_paths_asset = paths_asset
    temporary_capture_paths: Path | None = None
    if rotation_reset_asset.is_file():
        with np.load(rotation_reset_asset, allow_pickle=False) as source:
            rotation_target_ids = source["target_ids"].copy()
        with np.load(paths_asset, allow_pickle=False) as source:
            path_target_ids = source["target_ids"].copy()
        if not np.array_equal(
            rotation_target_ids.astype(str), path_target_ids.astype(str)
        ):
            with tempfile.NamedTemporaryFile(
                prefix=".goal_capture_paths-",
                suffix=".npz",
                dir=DATA_ROOT,
                delete=False,
            ) as descriptor:
                temporary_capture_paths = Path(descriptor.name)
            try:
                selected_count = _write_target_subset_paths_asset(
                    paths_asset,
                    temporary_capture_paths,
                    rotation_target_ids,
                )
            except Exception:
                temporary_capture_paths.unlink(missing_ok=True)
                raise
            capture_paths_asset = temporary_capture_paths
            print(
                f"[FILTER] MuJoCo goal rendering will use {selected_count}/"
                f"{len(path_target_ids)} path targets matching the validated "
                "rotation-reset asset.",
                flush=True,
            )
    previous_catalog_signature = catalog_file_signature(goal_catalog)
    failed_capture = DATA_ROOT / "goal_catalog_failed_validation.npz"
    previous_failed_capture_signature = catalog_file_signature(failed_capture)
    try:
        capture_error: subprocess.CalledProcessError | None = None
        try:
            _run(
                [
                    str((REPO_ROOT / "scripts/run_mujoco_filament.sh").resolve()),
                    sys.executable,
                    "isaac_rl/scripts/capture_multigrasp_goal_catalog_mujoco.py",
                    "--paths-asset",
                    str(capture_paths_asset),
                    "--manifest",
                    str(DATA_ROOT / "planned_manifest.json"),
                    "--output",
                    str(goal_catalog),
                    "--contact-sheet",
                    str(DEBUG_ROOT / "goal_rgb_contact_sheet.png"),
                ]
            )
        except subprocess.CalledProcessError as error:
            capture_error = error
        current_failed_capture_signature = catalog_file_signature(failed_capture)
        if (
            current_failed_capture_signature is not None
            and current_failed_capture_signature != previous_failed_capture_signature
        ):
            target_count = _finalize_failed_isaac_capture(DATA_ROOT)
        else:
            if capture_error is not None:
                raise capture_error
            target_count = validate_fresh_goal_catalog_capture(
                goal_catalog,
                capture_paths_asset,
                previous_signature=previous_catalog_signature,
            )
    finally:
        if temporary_capture_paths is not None:
            temporary_capture_paths.unlink(missing_ok=True)
    print(
        f"[VALIDATE] Fresh MuJoCo goal catalog contains {target_count} complete "
        "targets under the active visual profiles.",
        flush=True,
    )
    _render_debug_artifacts()


def main() -> None:
    args = parse_args()
    if args.targets_per_part_orientation <= 0:
        raise ValueError("--targets-per-part-orientation must be positive.")
    if args.alternates_per_part_orientation < 0:
        raise ValueError("--alternates-per-part-orientation must be non-negative.")
    if args.ik_timeout_s <= 0.0:
        raise ValueError("--ik-timeout-s must be positive.")
    stages = (
        ("sources", "manifest", "plan", "paths", "rotation")
        if args.stage == "cpu"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "sources":
            _generate_sources(force=bool(args.force_sources))
        elif stage == "manifest":
            _build_manifest(
                args.targets_per_part_orientation,
                args.alternates_per_part_orientation,
            )
        elif stage == "plan":
            _plan(
                force=bool(args.force_replan),
                start_mock_moveit=not bool(args.no_start_mock_moveit),
                ik_timeout_s=float(args.ik_timeout_s),
            )
        elif stage == "paths":
            _build_paths()
        elif stage == "rotation":
            _build_rotation_resets()
        elif stage in {"mujoco", "isaac"}:
            if stage == "isaac":
                print(
                    "[DEPRECATED] --stage isaac now aliases the MuJoCo Filament "
                    "goal renderer; use --stage mujoco.",
                    flush=True,
                )
            _run_mujoco_stage()
        elif stage == "finalize":
            _finalize_failed_isaac_capture(DATA_ROOT)
            _render_debug_artifacts()
        else:
            raise AssertionError(stage)
    if args.stage == "cpu":
        print(
            "\n[DONE] All CPU stages completed. When the GPU is free, run:\n"
            "python3 isaac_rl/scripts/prepare_plumbers_block_catalog.py --stage mujoco",
            flush=True,
        )


if __name__ == "__main__":
    main()
