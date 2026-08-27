#!/usr/bin/env python3
"""Convert MoveIt-validated grasp targets into Cartesian reset paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_assembly_multigrasp_manifest import (  # noqa: E402
    SPLIT_NAMES,
    _group_split_assignment,
)
from build_reset_trajectory_asset import (  # noqa: E402
    MOVEIT_TO_ISAAC_SIGNS,
    _straight_cartesian_joint_path,
)
from grasp_planning.start_poses import (  # noqa: E402
    PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    PDZ_GRIPPER_APPROACH_PROFILE,
    PDZ_GRIPPER_OPEN_WIDTH_M,
    VISUAL_SERVO_GRIPPER_PROFILE,
    pdz_gripper_approach_width_from_jaw_width,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_planned.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_paths.npz",
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=REPO_ROOT
        / "assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf",
    )
    parser.add_argument("--waypoints", type=int, default=32)
    parser.add_argument("--maximum-position-error-m", type=float, default=5.0e-5)
    parser.add_argument("--maximum-rotation-error-rad", type=float, default=5.0e-4)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--filtered-manifest-output",
        type=Path,
        default=None,
        help=(
            "Drop targets whose straight Cartesian path cannot be validated and "
            "write the resulting re-split manifest here. Without this option, one "
            "invalid target fails the build."
        ),
    )
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_multipart_planned_manifest(
    manifest: dict[str, object], targets: list[dict[str, object]]
) -> bool:
    """Validate and identify the schema-5 multipart manifest contract.

    The legacy schema-3 planner output intentionally contains empty ``parts``
    and ``split`` compatibility placeholders. Key presence therefore cannot be
    used to distinguish it from a real multipart manifest.
    """

    schema_version = int(manifest.get("schema_version", -1))
    if schema_version < 5:
        return False

    parts = manifest.get("parts")
    split = manifest.get("split")
    if not isinstance(parts, list) or not parts:
        raise ValueError(
            f"Multipart planned manifest schema {schema_version} requires a non-empty "
            "parts list."
        )
    if not isinstance(split, dict):
        raise ValueError(
            f"Multipart planned manifest schema {schema_version} requires split metadata."
        )
    split_names = tuple(str(name) for name in split.get("names", ()))
    if split_names != SPLIT_NAMES:
        raise ValueError(
            f"Multipart split names must be {SPLIT_NAMES}, got {split_names or 'missing'}."
        )
    required_target_fields = {
        "part_id",
        "local_orientation_id",
        "split",
    }
    for target in targets:
        missing = sorted(required_target_fields.difference(target))
        if missing:
            raise ValueError(
                f"Multipart target {target.get('target_id', '<unknown>')} is missing {missing}."
            )
        if str(target["split"]) not in split_names:
            raise ValueError(
                f"Multipart target {target.get('target_id', '<unknown>')} has unknown "
                f"split '{target['split']}'."
            )
    return True


def _filter_and_resplit_manifest(
    manifest: dict[str, object],
    targets: list[dict[str, object]],
    failures: list[dict[str, str]],
    *,
    maximum_position_error_m: float,
    maximum_rotation_error_rad: float,
) -> dict[str, object]:
    payload = json.loads(json.dumps(manifest))
    if "split" in payload and all("part_id" in target for target in targets):
        fractions = payload["split"]["requested_fractions"]
        assignment, salt = _group_split_assignment(
            targets,
            coverage_targets=targets,
            train_fraction=float(fractions["train"]),
            validation_fraction=float(fractions["validation"]),
            seed=int(payload["split"]["seed"]),
        )
        for target in targets:
            target["split"] = assignment[
                (str(target["part_id"]), str(target["grasp_id"]))
            ]
        payload["split"]["salt"] = salt

    target_counts_by_orientation = Counter(
        str(target["orientation_id"]) for target in targets
    )
    split_counts = Counter(str(target.get("split", "train")) for target in targets)
    orientations = []
    orientation_split_counts: dict[str, dict[str, int]] = {}
    for orientation in payload["orientations"]:
        orientation_id = str(orientation["orientation_id"])
        count = int(target_counts_by_orientation[orientation_id])
        if count == 0:
            continue
        orientation["selected_target_count"] = count
        counts = {
            split: sum(
                str(target["orientation_id"]) == orientation_id
                and str(target.get("split", "train")) == split
                for target in targets
            )
            for split in SPLIT_NAMES
        }
        orientation["split_target_counts"] = counts
        orientation_split_counts[orientation_id] = counts
        orientations.append(orientation)
    payload["orientations"] = orientations

    part_split_counts: dict[str, dict[str, int]] = {}
    for part in payload.get("parts", []):
        part_id = str(part["part_id"])
        part["selected_target_count"] = sum(
            str(target.get("part_id", "")) == part_id for target in targets
        )
        part["orientation_count"] = sum(
            str(orientation.get("part_id", "")) == part_id
            for orientation in orientations
        )
        part_split_counts[part_id] = {
            split: sum(
                str(target.get("part_id", "")) == part_id
                and str(target.get("split", "train")) == split
                for target in targets
            )
            for split in SPLIT_NAMES
        }
    if "split" in payload:
        payload["split"].update(
            {
                "target_counts": {
                    split: int(split_counts[split]) for split in SPLIT_NAMES
                },
                "part_target_counts": part_split_counts,
                "orientation_target_counts": orientation_split_counts,
            }
        )
    payload["targets"] = targets
    payload["selection"]["selected_target_count"] = len(targets)
    payload["cartesian_path_validation"] = {
        "complete": True,
        "validated_target_count": len(targets),
        "dropped_target_count": len(failures),
        "maximum_position_error_m": maximum_position_error_m,
        "maximum_rotation_error_rad": maximum_rotation_error_rad,
        "failures": failures,
    }
    return payload


def _target_gripper_apertures(
    targets: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return final jaw widths and collision-clear approach widths."""

    jaw_widths: list[float] = []
    approach_widths: list[float] = []
    for target in targets:
        target_id = str(target.get("target_id", "<unknown>"))
        world_grasp = target.get("world_grasp")
        if not isinstance(world_grasp, dict):
            raise ValueError(f"Target {target_id} is missing world_grasp metadata.")
        jaw_width = float(world_grasp["jaw_width_m"])
        approach_width = float(world_grasp["gripper_width_m"])
        expected_width = pdz_gripper_approach_width_from_jaw_width(jaw_width)
        if abs(approach_width - expected_width) > 1.0e-7:
            raise ValueError(
                f"Target {target_id} approach aperture is {approach_width:.6f} m; "
                f"expected jaw width + 10 mm = {expected_width:.6f} m."
            )
        if approach_width > PDZ_GRIPPER_OPEN_WIDTH_M + 1.0e-7:
            raise ValueError(
                f"Target {target_id} approach aperture exceeds the physical gripper opening."
            )
        jaw_widths.append(jaw_width)
        approach_widths.append(approach_width)
    return (
        np.asarray(jaw_widths, dtype=np.float32),
        np.asarray(approach_widths, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    if args.maximum_position_error_m <= 0.0:
        raise ValueError("--maximum-position-error-m must be positive.")
    if args.maximum_rotation_error_rad <= 0.0:
        raise ValueError("--maximum-rotation-error-rad must be positive.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    targets = list(manifest["targets"])
    if not manifest.get("moveit_validation", {}).get("complete", False):
        raise ValueError(f"Planned manifest is incomplete: {args.manifest}")
    expected_count = int(
        manifest.get("cartesian_path_validation", {}).get(
            "validated_target_count",
            manifest["moveit_validation"]["required_target_count"],
        )
    )
    if len(targets) != expected_count:
        raise ValueError(f"Expected {expected_count} targets, got {len(targets)}.")
    multipart_manifest = _is_multipart_planned_manifest(manifest, targets)

    paths = []
    maximum_position_errors = []
    maximum_rotation_errors = []
    path_targets: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, target in enumerate(targets, start=1):
        plan_path = Path(target["moveit_plan_path"])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("target_id") != target["target_id"]:
            raise ValueError(
                f"Plan target {plan.get('target_id')} does not match {target['target_id']}."
            )
        joint_names = tuple(str(name) for name in plan["joint_names"])
        expected_names = tuple(f"lbr_A{joint_index}" for joint_index in range(1, 8))
        if joint_names != expected_names:
            raise ValueError(f"Expected MoveIt joints {expected_names}, got {joint_names}.")
        raw_grasp_trajectory = np.asarray(
            plan["trajectories"]["grasp"], dtype=np.float64
        )
        try:
            path_moveit, maximum_position_error, maximum_rotation_error = (
                _straight_cartesian_joint_path(
                raw_moveit_trajectory=raw_grasp_trajectory,
                plan=plan,
                robot_urdf=args.robot_urdf,
                waypoint_count=args.waypoints,
                maximum_position_error_m=args.maximum_position_error_m,
                maximum_rotation_error_rad=args.maximum_rotation_error_rad,
            )
            )
        except RuntimeError as exc:
            if args.filtered_manifest_output is None:
                raise RuntimeError(f"{target['target_id']}: {exc}") from exc
            failures.append({"target_id": str(target["target_id"]), "error": str(exc)})
            print(f"[DROP PATH] {target['target_id']}: {exc}", flush=True)
            continue
        validated_target = json.loads(json.dumps(target))
        validated_target.setdefault("validation", {})[
            "cartesian_path_validated"
        ] = True
        path_targets.append(validated_target)
        paths.append(
            (path_moveit * MOVEIT_TO_ISAAC_SIGNS[None, :]).astype(np.float32)
        )
        maximum_position_errors.append(maximum_position_error)
        maximum_rotation_errors.append(maximum_rotation_error)
        if not args.quiet or index == len(targets) or index % 100 == 0:
            print(
                f"[PATH] {index:04d}/{len(targets)} {target['target_id']} "
                f"position={maximum_position_error * 1e6:.2f} um "
                f"rotation={np.degrees(maximum_rotation_error):.5f} deg",
                flush=True,
            )

    if not path_targets:
        raise RuntimeError("No target produced a valid straight Cartesian reset path.")
    if failures:
        manifest = _filter_and_resplit_manifest(
            manifest,
            path_targets,
            failures,
            maximum_position_error_m=float(args.maximum_position_error_m),
            maximum_rotation_error_rad=float(args.maximum_rotation_error_rad),
        )
        targets = list(manifest["targets"])
        filtered_manifest_path = args.filtered_manifest_output.expanduser().resolve()
        _atomic_write_json(filtered_manifest_path, manifest)
        print(
            f"[FILTER] Kept {len(targets)}/{expected_count} targets and wrote "
            f"{filtered_manifest_path}.",
            flush=True,
        )
    else:
        targets = path_targets

    orientation_names = [
        str(record["orientation_id"]) for record in manifest["orientations"]
    ]
    orientation_to_index = {
        name: index for index, name in enumerate(orientation_names)
    }

    object_positions = np.asarray(
        [target["object_pose_world"]["position_world"] for target in targets],
        dtype=np.float32,
    )
    object_orientations = np.asarray(
        [
            target["object_pose_world"]["orientation_xyzw_world"]
            for target in targets
        ],
        dtype=np.float32,
    )
    grasp_positions = np.asarray(
        [target["world_grasp"]["position_w"] for target in targets],
        dtype=np.float32,
    )
    grasp_orientations = np.asarray(
        [target["world_grasp"]["orientation_xyzw"] for target in targets],
        dtype=np.float32,
    )
    grasp_jaw_widths, approach_gripper_widths = _target_gripper_apertures(targets)
    # The current KUKA task uses identity grasp-to-TCP rotation and zero
    # center offset, so the target grasp and target TCP poses are identical.
    payload = {
        "schema_version": np.asarray(
            4 if multipart_manifest else 1,
            dtype=np.int64,
        ),
        "robot_profile": np.asarray(VISUAL_SERVO_GRIPPER_PROFILE),
        "approach_gripper_profile": np.asarray(PDZ_GRIPPER_APPROACH_PROFILE),
        "approach_clearance_per_finger_m": np.asarray(
            PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M, dtype=np.float32
        ),
        "source_planned_manifest": np.asarray(str(args.manifest.resolve())),
        "target_ids": np.asarray([str(target["target_id"]) for target in targets]),
        "orientation_names": np.asarray(orientation_names),
        "orientation_ids": np.asarray(
            [str(target["orientation_id"]) for target in targets]
        ),
        "orientation_indices": np.asarray(
            [orientation_to_index[str(target["orientation_id"])] for target in targets],
            dtype=np.int64,
        ),
        "grasp_ids": np.asarray([str(target["grasp_id"]) for target in targets]),
        "object_positions_w": object_positions,
        "object_orientations_xyzw_w": object_orientations,
        "goal_grasp_positions_w": grasp_positions,
        "goal_grasp_orientations_xyzw_w": grasp_orientations,
        "goal_tcp_positions_w": grasp_positions.copy(),
        "goal_tcp_orientations_xyzw_w": grasp_orientations.copy(),
        "grasp_jaw_widths_m": grasp_jaw_widths,
        "approach_gripper_widths_m": approach_gripper_widths,
        "reset_joint_trajectories": np.stack(paths, axis=0),
        "reset_path_progress": np.linspace(
            0.0, 1.0, args.waypoints, dtype=np.float32
        ),
        "reset_path_max_position_error_m": np.asarray(
            maximum_position_errors, dtype=np.float32
        ),
        "reset_path_max_rotation_error_rad": np.asarray(
            maximum_rotation_errors, dtype=np.float32
        ),
        "moveit_plan_validated": np.ones(len(targets), dtype=np.bool_),
        "isaac_goal_rgbd_captured": np.zeros(len(targets), dtype=np.bool_),
    }
    if multipart_manifest:
        part_names = [str(part["part_id"]) for part in manifest["parts"]]
        split_names = [str(name) for name in manifest["split"]["names"]]
        part_to_index = {name: index for index, name in enumerate(part_names)}
        split_to_index = {name: index for index, name in enumerate(split_names)}
        payload.update(
            {
                "assembly_name": np.asarray(str(manifest["assembly_name"])),
                "part_names": np.asarray(part_names),
                "part_usd_paths": np.asarray(
                    [str(part["part_usd_path"]) for part in manifest["parts"]]
                ),
                "part_ids": np.asarray([str(target["part_id"]) for target in targets]),
                "part_indices": np.asarray(
                    [part_to_index[str(target["part_id"])] for target in targets],
                    dtype=np.int64,
                ),
                "local_orientation_ids": np.asarray(
                    [str(target["local_orientation_id"]) for target in targets]
                ),
                "split_names": np.asarray(split_names),
                "split_ids": np.asarray([str(target["split"]) for target in targets]),
                "split_indices": np.asarray(
                    [split_to_index[str(target["split"])] for target in targets],
                    dtype=np.int64,
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"[DONE] Wrote {len(targets)} reset paths to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
