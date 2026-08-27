#!/usr/bin/env python3
"""Convert an all-Fabrica PDZ grasp benchmark into namespaced RL manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_assembly_multigrasp_manifest import (  # noqa: E402
    GLOBAL_ID_SCHEMA_VERSION,
    SPLIT_NAMES,
    _annotate_part_manifest,
    _group_split_assignment,
    _part_key,
)
from build_multigrasp_manifest import (  # noqa: E402
    _pose_payload,
    _target_payload,
    select_diverse_targets,
)
from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.start_poses import PDZ_GRIPPER_CLOSED_WIDTH_M  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs/fabrica_all_v1.yaml"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "artifacts/grasp_generation_benchmark_pdz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def _part_manifest(
    *,
    assembly_name: str,
    local_part_id: str,
    part_index: int,
    part_dir: Path,
    output_root: Path,
    selection: dict[str, object],
    object_xy_world: tuple[float, float],
) -> dict[str, object]:
    selected_targets: list[dict[str, object]] = []
    alternate_targets: list[dict[str, object]] = []
    orientations: list[dict[str, object]] = []
    representative_stage2_bundle: Path | None = None
    source_stage1 = part_dir / "stage1/grasps.json"
    if not source_stage1.is_file():
        raise FileNotFoundError(source_stage1)
    stage1 = load_grasp_bundle(source_stage1)
    target_cap = int(selection["targets_per_part_orientation"])
    alternate_cap = int(selection["alternates_per_part_orientation"])
    min_width = float(selection["min_training_jaw_width_m"])
    max_width = float(selection["max_training_jaw_width_m"])
    if min_width < PDZ_GRIPPER_CLOSED_WIDTH_M:
        raise ValueError(
            "min_training_jaw_width_m cannot be below the PDZ gripper's "
            f"{PDZ_GRIPPER_CLOSED_WIDTH_M:.3f} m closed gap"
        )
    if min_width > max_width:
        raise ValueError("min_training_jaw_width_m cannot exceed max_training_jaw_width_m")
    minimum_pregrasp_height = float(selection["minimum_pregrasp_height_m"])
    target_args = SimpleNamespace(
        pregrasp_offset=float(selection["pregrasp_offset_m"]),
        gripper_width_clearance=float(selection["gripper_width_clearance_m"]),
    )
    for orientation_dir in sorted((part_dir / "orientations").glob("orientation_*")):
        stage2_path = orientation_dir / "stage2.json"
        details_path = orientation_dir / "details.json"
        if not stage2_path.is_file() or not details_path.is_file():
            continue
        details = json.loads(details_path.read_text(encoding="utf-8"))
        if details.get("status") != "direct_success":
            continue
        orientation = dict(details["orientation"])
        local_orientation_id = str(orientation["orientation_id"])
        raw_pose = dict(orientation["object_pose_world"])
        object_pose = ObjectWorldPose(
            position_world=(
                object_xy_world[0],
                object_xy_world[1],
                float(raw_pose["position_world"][2]),
            ),
            orientation_xyzw_world=tuple(
                float(value) for value in raw_pose["orientation_xyzw_world"]
            ),
        )
        stage2 = load_grasp_bundle(stage2_path)
        candidates = [
            _target_payload(
                grasp,
                orientation_id=local_orientation_id,
                object_pose=object_pose,
                args=target_args,
            )
            for grasp in stage2.candidates
            if min_width <= float(grasp.jaw_width) <= max_width
        ]
        candidates = [
            target
            for target in candidates
            if float(target["world_grasp"]["pregrasp_position_w"][2])
            > minimum_pregrasp_height
        ]
        if not candidates:
            continue
        if representative_stage2_bundle is None:
            representative_stage2_bundle = stage2_path.resolve()
        ranked_count = min(len(candidates), target_cap + alternate_cap)
        ranked = select_diverse_targets(candidates, count=ranked_count)
        selected_count = min(target_cap, len(ranked))
        selected = ranked[:selected_count]
        alternates = ranked[selected_count:]
        for rank, target in enumerate(selected):
            target["orientation_selection_rank"] = rank
            target["initial_selection"] = True
        for rank, target in enumerate(alternates, start=selected_count):
            target["orientation_selection_rank"] = rank
            target["initial_selection"] = False
        selected_targets.extend(selected)
        alternate_targets.extend(alternates)
        orientations.append(
            {
                "orientation_id": local_orientation_id,
                "object_pose_world": _pose_payload(object_pose),
                "kind": "robust_stable_orientation",
                "normal_obj": orientation.get("normal_obj"),
                "support_area_m2": orientation.get("area_m2"),
                "stability_margin_m": orientation.get("stability_margin_m"),
                "max_stable_tilt_deg": orientation.get("max_stable_tilt_deg"),
                "ground_feasible_count": len(stage2.candidates),
                "training_width_feasible_count": len(candidates),
                "selected_target_count": len(selected),
                "selected_grasp_ids": [str(target["grasp_id"]) for target in selected],
                "alternate_target_count": len(alternates),
                "alternate_grasp_ids": [str(target["grasp_id"]) for target in alternates],
                "source_stage2_bundle": str(stage2_path.resolve()),
            }
        )
    if not selected_targets:
        raise ValueError("no stable orientation produced a selectable direct grasp")
    distinct_grasp_ids = {
        str(target["grasp_id"])
        for target in selected_targets + alternate_targets
    }
    if len(distinct_grasp_ids) < 3:
        raise ValueError(
            "fewer than three distinct grasp groups passed; leakage-safe "
            "train/validation/test assignment is impossible"
        )
    return _annotate_part_manifest(
        {
            "schema_version": 2,
            "target_mesh_path": stage1.target_mesh_path,
            "mesh_scale": float(stage1.mesh_scale),
            "source_stage1_bundle": str(source_stage1.resolve()),
            "source_current_stage2_bundle": str(representative_stage2_bundle),
            "selection": {
                "target_count": len(selected_targets),
                "orientation_count": len(orientations),
                "targets_per_orientation": target_cap,
                "targets_per_orientation_is_cap": True,
                "min_training_jaw_width_m": min_width,
                "max_training_jaw_width_m": max_width,
                "minimum_pregrasp_height_m": minimum_pregrasp_height,
                "alternates_per_orientation": alternate_cap,
                "method": "benchmark_ground_feasible_farthest_point",
                "moveit_validation_required": True,
                "isaac_goal_capture_required": True,
                "gripper_collision_model": "pdz_gripper",
            },
            "orientations": orientations,
            "targets": selected_targets,
            "alternates": alternate_targets,
        },
        assembly_name=assembly_name,
        part_id=local_part_id,
        part_index=part_index,
        part_usd_path=(
            output_root
            / "assemblies"
            / assembly_name
            / "usd"
            / f"part_{local_part_id}_bundle_local.usd"
        ),
        global_ids=True,
    )


def build_assembly_manifest(
    *,
    assembly_name: str,
    part_ids: list[str],
    benchmark_root: Path,
    output_root: Path,
    selection: dict[str, object],
    split: dict[str, object],
    object_xy_world: tuple[float, float],
) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    orientations: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    alternates: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for part_index, local_part_id in enumerate(part_ids):
        part_dir = benchmark_root / "parts" / assembly_name / local_part_id
        try:
            part_manifest = _part_manifest(
                assembly_name=assembly_name,
                local_part_id=local_part_id,
                part_index=part_index,
                part_dir=part_dir,
                output_root=output_root,
                selection=selection,
                object_xy_world=object_xy_world,
            )
        except Exception as error:
            exclusions.append(
                {
                    "assembly_name": assembly_name,
                    "local_part_id": local_part_id,
                    "part_key": _part_key(assembly_name, local_part_id),
                    "reason": "no_valid_training_targets",
                    "detail": str(error),
                }
            )
            continue
        parts.append(
            {
                "part_id": _part_key(assembly_name, local_part_id),
                "part_key": _part_key(assembly_name, local_part_id),
                "local_part_id": local_part_id,
                "assembly_name": assembly_name,
                "part_index": part_index,
                "target_mesh_path": str(part_manifest["target_mesh_path"]),
                "mesh_scale": float(part_manifest["mesh_scale"]),
                "source_stage1_bundle": str(part_manifest["source_stage1_bundle"]),
                "source_current_stage2_bundle": str(
                    part_manifest["source_current_stage2_bundle"]
                ),
                "part_usd_path": str(
                    output_root
                    / "assemblies"
                    / assembly_name
                    / "usd"
                    / f"part_{local_part_id}_bundle_local.usd"
                ),
                "orientation_count": len(part_manifest["orientations"]),
                "selected_target_count": len(part_manifest["targets"]),
                "alternate_target_count": len(part_manifest["alternates"]),
            }
        )
        orientations.extend(part_manifest["orientations"])
        targets.extend(part_manifest["targets"])
        alternates.extend(part_manifest["alternates"])
    if targets:
        assignment, salt = _group_split_assignment(
            targets + alternates,
            coverage_targets=targets,
            train_fraction=float(split["train_fraction"]),
            validation_fraction=float(split["validation_fraction"]),
            seed=int(split["seed"]),
        )
    else:
        assignment, salt = {}, 0
    for target in targets + alternates:
        target["split"] = assignment[(str(target["part_id"]), str(target["grasp_id"]))]
    split_counts = Counter(str(target["split"]) for target in targets)
    for orientation in orientations:
        orientation["split_target_counts"] = {
            name: sum(
                target["orientation_id"] == orientation["orientation_id"]
                and target["split"] == name
                for target in targets
            )
            for name in SPLIT_NAMES
        }
    return {
        "schema_version": GLOBAL_ID_SCHEMA_VERSION,
        "assembly_name": assembly_name,
        "configured_part_count": len(part_ids),
        "part_count": len(parts),
        "parts": parts,
        "exclusions": exclusions,
        "selection": {
            "selected_target_count": len(targets),
            "alternate_target_count": len(alternates),
            "targets_per_part_orientation_cap": int(
                selection["targets_per_part_orientation"]
            ),
            "alternates_per_part_orientation_cap": int(
                selection["alternates_per_part_orientation"]
            ),
            "method": "benchmark_ground_feasible_farthest_point",
            "moveit_validation_required": True,
            "isaac_goal_capture_required": True,
        },
        "split": {
            "names": list(SPLIT_NAMES),
            "unit": "assembly_name_local_part_id_grasp_id_group",
            "seed": int(split["seed"]),
            "salt": salt,
            "requested_fractions": {
                "train": float(split["train_fraction"]),
                "validation": float(split["validation_fraction"]),
                "test": float(
                    1.0
                    - float(split["train_fraction"])
                    - float(split["validation_fraction"])
                ),
            },
            "target_counts": {name: int(split_counts[name]) for name in SPLIT_NAMES},
        },
        "orientations": orientations,
        "targets": targets,
        "alternates": alternates,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = dict(config["dataset"])
    selection = dict(config["selection"])
    split = dict(config["splits"])
    benchmark_root = args.benchmark_root.expanduser().resolve()
    if not (benchmark_root / "results.json").is_file():
        raise FileNotFoundError(
            f"Benchmark is incomplete; missing '{benchmark_root / 'results.json'}'."
        )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (REPO_ROOT / str(dataset["output_root"])).resolve()
    )
    object_xy_world = (0.4252643585205078, 0.05988234281539917)
    written: list[Path] = []
    for assembly_name in sorted(str(name) for name in config["assemblies"]):
        part_ids = sorted(
            (str(value) for value in config["assemblies"][assembly_name]),
            key=lambda value: (int(value), value),
        )
        manifest = build_assembly_manifest(
            assembly_name=assembly_name,
            part_ids=part_ids,
            benchmark_root=benchmark_root,
            output_root=output_root,
            selection=selection,
            split=split,
            object_xy_world=object_xy_world,
        )
        output = output_root / "assemblies" / assembly_name / "manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written.append(output)
        print(
            f"[ASSEMBLY] {assembly_name}: parts={manifest['part_count']}/"
            f"{manifest['configured_part_count']} targets="
            f"{manifest['selection']['selected_target_count']} exclusions="
            f"{len(manifest['exclusions'])}",
            flush=True,
        )
    print(f"[DONE] Wrote {len(written)} assembly manifests under {output_root}.")


if __name__ == "__main__":
    main()
