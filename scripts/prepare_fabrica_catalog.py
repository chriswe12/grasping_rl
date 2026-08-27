#!/usr/bin/env python3
"""Prepare all-Fabrica RL grasp sources and namespaced manifests in stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/fabrica_all_v1.yaml"
BASELINE_FILES = (
    "assembly_manifest.json",
    "planned_manifest.json",
    "paths.npz",
    "rotation_resets.npz",
    "goal_catalog.npz",
    "goal_catalog_validation_report.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "baseline",
            "inventory",
            "sources",
            "manifest",
            "merge",
            "plan",
            "paths",
            "rotation",
            "mujoco",
            "finalize",
            "cpu",
        ),
        default="cpu",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force-sources", action="store_true")
    parser.add_argument("--force-replan", action="store_true")
    parser.add_argument("--ik-timeout-s", type=float, default=0.5)
    parser.add_argument("--no-start-mock-moveit", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _plan(
    *,
    output_root: Path,
    force: bool,
    start_mock_moveit: bool,
    ik_timeout_s: float,
) -> None:
    moveit_process: subprocess.Popen | None = None
    environment = os.environ.copy()
    environment.setdefault("ROS_LOG_DIR", "/tmp/ros-log")
    Path(environment["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    if start_mock_moveit:
        print("[START] Launching the repo-local mock iiwa7 MoveIt stack.", flush=True)
        moveit_process = subprocess.Popen(
            [str(REPO_ROOT / "start_lbr_moveit.sh")],
            cwd=REPO_ROOT,
            env=environment,
        )
    try:
        command = [
            sys.executable,
            "isaac_rl/scripts/plan_multigrasp_targets.py",
            "--manifest",
            str(output_root / "merged/manifest.json"),
            "--output-manifest",
            str(output_root / "merged/planned_manifest.json"),
            "--plans-dir",
            str(output_root / "merged/plans"),
            "--exclusions-file",
            str(output_root / "reports/goal_exclusions.json"),
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
        print(f"[RUN] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)
    finally:
        if moveit_process is not None and moveit_process.poll() is None:
            print("[STOP] Stopping the repo-local mock MoveIt stack.", flush=True)
            moveit_process.send_signal(signal.SIGINT)
            try:
                moveit_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                moveit_process.terminate()
                moveit_process.wait(timeout=5.0)


def _run_mujoco_capture(output_root: Path) -> None:
    from prepare_plumbers_block_catalog import (  # local script helpers
        _finalize_failed_isaac_capture,
        _write_target_subset_paths_asset,
    )

    data_root = output_root / "merged"
    paths_asset = data_root / "paths.npz"
    rotation_asset = data_root / "rotation_resets.npz"
    capture_paths = paths_asset
    temporary: Path | None = None
    with np.load(paths_asset, allow_pickle=False) as source:
        path_ids = source["target_ids"].astype(str)
    with np.load(rotation_asset, allow_pickle=False) as source:
        rotation_ids = source["target_ids"].astype(str)
    if not np.array_equal(path_ids, rotation_ids):
        with tempfile.NamedTemporaryFile(
            prefix=".fabrica-goal-paths-",
            suffix=".npz",
            dir=data_root,
            delete=False,
        ) as descriptor:
            temporary = Path(descriptor.name)
        _write_target_subset_paths_asset(paths_asset, temporary, rotation_ids)
        os.replace(temporary, paths_asset)
        temporary = None
        capture_paths = paths_asset
        print(
            f"[FILTER] MuJoCo capture uses {len(rotation_ids)}/{len(path_ids)} "
            "paths retained by reset validation.",
            flush=True,
        )
    try:
        try:
            _run(
                [
                    str((REPO_ROOT / "scripts/run_mujoco_filament.sh").resolve()),
                    sys.executable,
                    "isaac_rl/scripts/capture_multigrasp_goal_catalog_mujoco.py",
                    "--paths-asset",
                    str(capture_paths),
                    "--manifest",
                    str(data_root / "planned_manifest.json"),
                    "--output",
                    str(data_root / "goal_catalog.npz"),
                ]
            )
        except subprocess.CalledProcessError:
            diagnostic = data_root / "goal_catalog_failed_validation.npz"
            if not diagnostic.is_file():
                raise
            _finalize_failed_isaac_capture(data_root)
            with np.load(data_root / "goal_catalog.npz", allow_pickle=False) as source:
                finalized_ids = source["target_ids"].astype(str)
            with tempfile.NamedTemporaryFile(
                prefix=".fabrica-final-paths-",
                suffix=".npz",
                dir=data_root,
                delete=False,
            ) as descriptor:
                finalized_paths = Path(descriptor.name)
            try:
                _write_target_subset_paths_asset(
                    paths_asset, finalized_paths, finalized_ids
                )
                os.replace(finalized_paths, paths_asset)
            finally:
                finalized_paths.unlink(missing_ok=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_root(config: dict[str, object]) -> Path:
    path = Path(str(config["dataset"]["output_root"]))
    return path if path.is_absolute() else REPO_ROOT / path


def _record_baseline(output_root: Path) -> None:
    baseline_root = REPO_ROOT / "isaac_rl/data/plumbers_block"
    records: list[dict[str, object]] = []
    for name in BASELINE_FILES:
        path = baseline_root / name
        record: dict[str, object] = {"path": str(path.relative_to(REPO_ROOT))}
        if not path.is_file():
            record["status"] = "missing"
            records.append(record)
            continue
        record.update(
            {
                "status": "present",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                record["schema_version"] = (
                    int(np.asarray(arrays["schema_version"]).item())
                    if "schema_version" in arrays.files
                    else None
                )
                record["target_count"] = (
                    len(arrays["target_ids"]) if "target_ids" in arrays.files else None
                )
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            record["schema_version"] = payload.get("schema_version")
            if isinstance(payload.get("targets"), list):
                record["target_count"] = len(payload["targets"])
        records.append(record)
    output = output_root / "reports/plumbers_block_baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": 1, "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] Recorded plumbers-block baseline signatures: {output}", flush=True)


def _source_results_are_complete(benchmark_root: Path, expected_count: int) -> bool:
    results_path = benchmark_root / "results.json"
    if not results_path.is_file():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("summary", {}).get("part_count", -1)) == expected_count


def _write_reports(
    *, config_path: Path, output_root: Path, benchmark_root: Path
) -> None:
    _run(
        [
            sys.executable,
            "isaac_rl/scripts/build_fabrica_grasp_reports.py",
            "--config",
            str(config_path),
            "--inventory",
            str(output_root / "inventory.json"),
            "--benchmark-results",
            str(benchmark_root / "results.json"),
            "--manifest",
            str(output_root / "merged/manifest.json"),
            "--output-root",
            str(output_root),
        ]
    )


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    if args.ik_timeout_s <= 0.0:
        raise ValueError("--ik-timeout-s must be positive.")
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = _output_root(config).resolve()
    benchmark_root = REPO_ROOT / "artifacts/grasp_generation_benchmark_pdz"
    expected_count = sum(len(values) for values in config["assemblies"].values())
    stages = (
        (
            "baseline",
            "inventory",
            "sources",
            "manifest",
            "merge",
            "plan",
            "paths",
            "rotation",
        )
        if args.stage == "cpu"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "baseline":
            _record_baseline(output_root)
        elif stage == "inventory":
            _run(
                [
                    sys.executable,
                    "isaac_rl/scripts/build_fabrica_rl_inventory.py",
                    "--config",
                    str(config_path),
                ]
            )
        elif stage == "sources":
            if not args.force_sources and _source_results_are_complete(
                benchmark_root, expected_count
            ):
                print(
                    f"[REUSE] Complete {expected_count}-part PDZ benchmark: "
                    f"{benchmark_root / 'results.json'}",
                    flush=True,
                )
            else:
                _run(
                    [
                        sys.executable,
                        "scripts/run_grasp_generation_benchmark.py",
                        "--config",
                        "configs/grasp_generation_benchmark_pdz.yaml",
                        "--jobs",
                        str(args.jobs),
                    ]
                )
        elif stage == "manifest":
            _run(
                [
                    sys.executable,
                    "isaac_rl/scripts/build_fabrica_manifest_from_benchmark.py",
                    "--config",
                    str(config_path),
                    "--benchmark-root",
                    str(benchmark_root),
                    "--output-root",
                    str(output_root),
                ]
            )
        elif stage == "merge":
            command = [
                sys.executable,
                "isaac_rl/scripts/merge_fabrica_multigrasp_manifests.py",
            ]
            for assembly_name in sorted(config["assemblies"]):
                command.extend(
                    [
                        "--manifest",
                        str(output_root / "assemblies" / assembly_name / "manifest.json"),
                    ]
                )
            split = dict(config["splits"])
            command.extend(
                [
                    "--output",
                    str(output_root / "merged/manifest.json"),
                    "--split-seed",
                    str(split["seed"]),
                    "--held-out-part-fraction",
                    str(split["held_out_part_fraction"]),
                ]
            )
            for assembly_name in split.get("held_out_assemblies", []):
                command.extend(["--held-out-assembly", str(assembly_name)])
            _run(command)
            _write_reports(
                config_path=config_path,
                output_root=output_root,
                benchmark_root=benchmark_root,
            )
        elif stage == "plan":
            _plan(
                output_root=output_root,
                force=bool(args.force_replan),
                start_mock_moveit=not bool(args.no_start_mock_moveit),
                ik_timeout_s=float(args.ik_timeout_s),
            )
        elif stage == "paths":
            _run(
                [
                    sys.executable,
                    "isaac_rl/scripts/build_multigrasp_path_asset.py",
                    "--manifest",
                    str(output_root / "merged/planned_manifest.json"),
                    "--output",
                    str(output_root / "merged/paths.npz"),
                    "--quiet",
                    "--filtered-manifest-output",
                    str(output_root / "merged/planned_manifest.json"),
                ]
            )
        elif stage == "rotation":
            reset_config = dict(config["resets"])
            _run(
                [
                    sys.executable,
                    "isaac_rl/scripts/build_multigrasp_rotation_reset_asset.py",
                    "--paths-asset",
                    str(output_root / "merged/paths.npz"),
                    "--output",
                    str(output_root / "merged/rotation_resets.npz"),
                    "--cache-dir",
                    str(output_root / "merged/rotation_cache"),
                    "--variants",
                    str(reset_config["variants_per_target"]),
                    "--minimum-distinct-variants",
                    str(reset_config["minimum_distinct_variants"]),
                    "--quiet-cached",
                    "--quiet",
                ]
            )
        elif stage == "mujoco":
            _run_mujoco_capture(output_root)
            _write_reports(
                config_path=config_path,
                output_root=output_root,
                benchmark_root=benchmark_root,
            )
        elif stage == "finalize":
            from prepare_plumbers_block_catalog import (
                _finalize_failed_isaac_capture,
                _write_target_subset_paths_asset,
            )

            data_root = output_root / "merged"
            _finalize_failed_isaac_capture(data_root)
            with np.load(data_root / "goal_catalog.npz", allow_pickle=False) as source:
                finalized_ids = source["target_ids"].astype(str)
            with tempfile.NamedTemporaryFile(
                prefix=".fabrica-final-paths-",
                suffix=".npz",
                dir=data_root,
                delete=False,
            ) as descriptor:
                finalized_paths = Path(descriptor.name)
            try:
                _write_target_subset_paths_asset(
                    data_root / "paths.npz", finalized_paths, finalized_ids
                )
                os.replace(finalized_paths, data_root / "paths.npz")
            finally:
                finalized_paths.unlink(missing_ok=True)
            _write_reports(
                config_path=config_path,
                output_root=output_root,
                benchmark_root=benchmark_root,
            )
        else:
            raise AssertionError(stage)
    if args.stage == "cpu":
        print(
            f"[DONE] CPU grasp generation and namespaced manifest build completed: "
            f"{output_root / 'merged/manifest.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
