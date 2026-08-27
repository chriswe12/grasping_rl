"""Select a diverse, ground-feasible multi-grasp target manifest for visual RL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    load_asset_mesh,
    load_grasp_bundle,
    quat_to_rotmat_xyzw,
)
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline.fabrica_pipeline import (  # noqa: E402
    PlanningConfig,
    _mesh_in_source_frame,
    _source_frame_pose_from_bundle,
    recheck_stage2_result,
)
from grasp_planning.pipeline.stable_orientations import (  # noqa: E402
    StableOrientationConfig,
    enumerate_stable_orientations,
)

SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-bundle",
        type=Path,
        default=REPO_ROOT / "artifacts/pipeline_stage1_assembly_grasps.json",
    )
    parser.add_argument(
        "--current-stage2-bundle",
        type=Path,
        default=REPO_ROOT / "artifacts/pipeline_stage2_ground_feasible.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_manifest.json",
    )
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument(
        "--targets-per-orientation",
        type=int,
        default=None,
        help=(
            "Optional per-orientation cap. Set --target-count 0 and this value to "
            "build a variable-size catalog when some stable orientations expose "
            "fewer feasible grasps than others."
        ),
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--gripper-width-clearance", type=float, default=0.01)
    parser.add_argument("--max-training-jaw-width", type=float, default=0.075)
    parser.add_argument("--minimum-pregrasp-height", type=float, default=0.05)
    parser.add_argument(
        "--object-xy-world",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Override the source stage-2 XY while preserving each stable support pose.",
    )
    parser.add_argument("--alternates-per-orientation", type=int, default=20)
    return parser.parse_args()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def _angle_distance(a: np.ndarray, b: np.ndarray, *, unsigned: bool = False) -> float:
    dot = float(np.dot(_unit(a), _unit(b)))
    if unsigned:
        dot = abs(dot)
    return math.acos(float(np.clip(dot, -1.0, 1.0)))


def target_diversity_distance(first: dict[str, object], second: dict[str, object]) -> float:
    """Dimensionless distance over contact location, approach, camera roll, and width."""

    position_distance = np.linalg.norm(
        np.asarray(first["grasp_position_obj"], dtype=float)
        - np.asarray(second["grasp_position_obj"], dtype=float)
    ) / 0.015
    approach_distance = _angle_distance(
        np.asarray(first["approach_axis_world"], dtype=float),
        np.asarray(second["approach_axis_world"], dtype=float),
    ) / math.radians(30.0)
    # Finger exchange is physically symmetric, so compare closing axes without sign.
    closing_distance = _angle_distance(
        np.asarray(first["closing_axis_world"], dtype=float),
        np.asarray(second["closing_axis_world"], dtype=float),
        unsigned=True,
    ) / math.radians(45.0)
    jaw_distance = abs(float(first["jaw_width_m"]) - float(second["jaw_width_m"])) / 0.020
    return float(
        math.sqrt(
            position_distance**2
            + approach_distance**2
            + closing_distance**2
            + jaw_distance**2
        )
    )


def select_diverse_targets(
    candidates: list[dict[str, object]],
    *,
    count: int,
    anchor_grasp_ids: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Greedy farthest-point selection with score used as a deterministic prior."""

    if count <= 0:
        return []
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, but only {len(candidates)} passed filters.")
    by_grasp_id = {str(candidate["grasp_id"]): candidate for candidate in candidates}
    selected = [by_grasp_id[grasp_id] for grasp_id in anchor_grasp_ids if grasp_id in by_grasp_id]
    if not selected:
        selected.append(max(candidates, key=lambda item: (float(item["score"]), str(item["target_id"]))))
    remaining = [candidate for candidate in candidates if candidate not in selected]
    minimum_distances = {
        id(candidate): min(
            target_diversity_distance(candidate, prior) for prior in selected
        )
        for candidate in remaining
    }
    while len(selected) < count:
        choice = max(
            remaining,
            key=lambda candidate: (
                minimum_distances[id(candidate)] + 0.15 * float(candidate["score"]),
                float(candidate["score"]),
                str(candidate["target_id"]),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
        for candidate in remaining:
            minimum_distances[id(candidate)] = min(
                minimum_distances[id(candidate)],
                target_diversity_distance(candidate, choice),
            )
    return selected


def _pose_payload(pose: ObjectWorldPose) -> dict[str, list[float]]:
    return {
        "position_world": [float(value) for value in pose.position_world],
        "orientation_xyzw_world": [float(value) for value in pose.orientation_xyzw_world],
    }


def _world_grasp_payload(world_grasp) -> dict[str, object]:
    return {
        "position_w": list(world_grasp.position_w),
        "orientation_xyzw": list(world_grasp.orientation_xyzw),
        "approach_axis_w": list(world_grasp.normal_w),
        "pregrasp_position_w": list(world_grasp.pregrasp_position_w),
        "pregrasp_offset_m": float(world_grasp.pregrasp_offset),
        "jaw_width_m": float(world_grasp.jaw_width),
        "gripper_width_m": float(world_grasp.gripper_width),
    }


def _target_payload(grasp, *, orientation_id: str, object_pose: ObjectWorldPose, args) -> dict[str, object]:
    rotation_world = object_pose.rotation_world_from_object @ quat_to_rotmat_xyzw(
        grasp.grasp_orientation_xyzw_obj
    )
    world_grasp = saved_grasp_to_world_grasp(
        grasp,
        object_pose,
        pregrasp_offset=float(args.pregrasp_offset),
        gripper_width_clearance=float(args.gripper_width_clearance),
    )
    return {
        "target_id": f"{orientation_id}__{grasp.grasp_id}",
        "orientation_id": orientation_id,
        "grasp_id": grasp.grasp_id,
        "score": float(grasp.score or 0.0),
        "jaw_width_m": float(grasp.jaw_width),
        "grasp_position_obj": list(grasp.grasp_position_obj),
        "grasp_orientation_xyzw_obj": list(grasp.grasp_orientation_xyzw_obj),
        "approach_axis_world": [float(value) for value in rotation_world[:, 2]],
        "closing_axis_world": [float(value) for value in rotation_world[:, 1]],
        "object_pose_world": _pose_payload(object_pose),
        "world_grasp": _world_grasp_payload(world_grasp),
        "validation": {
            "ground_and_gripper_feasible": True,
            "moveit_plan_validated": False,
            "isaac_goal_rgbd_captured": False,
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    if args.target_count < 0:
        raise ValueError("--target-count must be non-negative.")
    targets_per_orientation_cap = getattr(args, "targets_per_orientation", None)
    if args.target_count == 0 and (
        targets_per_orientation_cap is None or targets_per_orientation_cap <= 0
    ):
        raise ValueError(
            "Set a positive --target-count, or use --target-count 0 with a positive "
            "--targets-per-orientation cap."
        )
    if args.alternates_per_orientation < 0:
        raise ValueError("--alternates-per-orientation must be non-negative.")
    stage1 = load_grasp_bundle(args.stage1_bundle)
    current_stage2 = load_grasp_bundle(args.current_stage2_bundle)
    current_pose_payload = dict(current_stage2.metadata["execution_world_pose"])
    current_pose = ObjectWorldPose(
        position_world=tuple(float(value) for value in current_pose_payload["position_world"]),
        orientation_xyzw_world=tuple(
            float(value) for value in current_pose_payload["orientation_xyzw_world"]
        ),
    )
    mesh_global = load_asset_mesh(stage1.target_mesh_path, scale=stage1.mesh_scale)
    mesh_local = _mesh_in_source_frame(mesh_global, _source_frame_pose_from_bundle(stage1))
    stable_result = enumerate_stable_orientations(mesh_local, StableOrientationConfig())
    object_xy = (
        tuple(float(value) for value in args.object_xy_world)
        if getattr(args, "object_xy_world", None) is not None
        else current_pose.position_world[:2]
    )
    orientation_poses: list[tuple[str, ObjectWorldPose, dict[str, object]]] = [
        (
            "current",
            current_pose,
            {"kind": "existing_kinematic_pose", "normal_obj": None},
        )
    ]
    for orientation in stable_result.orientations:
        stable_pose = orientation.object_pose_world
        pose = ObjectWorldPose(
            position_world=(object_xy[0], object_xy[1], stable_pose.position_world[2]),
            orientation_xyzw_world=stable_pose.orientation_xyzw_world,
        )
        orientation_poses.append(
            (
                orientation.orientation_id,
                pose,
                {
                    "kind": "robust_stable_orientation",
                    "normal_obj": list(orientation.normal_obj),
                    "support_area_m2": float(orientation.area_m2),
                    "stability_margin_m": float(orientation.stability_margin_m),
                    "max_stable_tilt_deg": float(orientation.max_stable_tilt_deg),
                },
            )
        )
    if args.target_count > 0 and args.target_count % len(orientation_poses) != 0:
        raise ValueError(
            f"--target-count must be divisible by {len(orientation_poses)} orientations for balanced selection."
        )
    targets_per_orientation = (
        args.target_count // len(orientation_poses)
        if args.target_count > 0
        else int(targets_per_orientation_cap)
    )
    gripper_collision_model = str(
        stage1.metadata.get("gripper_model", "kuka_y_gripper")
    )
    detailed_contact_gap_m = (
        0.005 if gripper_collision_model == "pdz_gripper" else 0.002
    )
    planning = PlanningConfig(
        detailed_finger_contact_gap_m=detailed_contact_gap_m,
        gripper_collision_model=gripper_collision_model,
        floor_clearance_margin_m=0.01,
        top_grasp_score_weight=0.35,
        reachability_proxy_score_weight=0.15,
        reachability_proxy_hand_offset_m=0.10,
        contact_lateral_offsets_m=(-0.002916666666666667, 0.0, 0.002916666666666667),
        contact_approach_offsets_m=(-0.0030833333333333333, 0.0, 0.0030833333333333333),
    )
    selected_targets: list[dict[str, object]] = []
    alternate_targets: list[dict[str, object]] = []
    orientation_records: list[dict[str, object]] = []
    for orientation_id, object_pose, orientation_metadata in orientation_poses:
        result = recheck_stage2_result(
            bundle=stage1,
            pickup_spec=None,
            planning=planning,
            object_pose_world=object_pose,
        )
        candidates = [
            _target_payload(
                grasp,
                orientation_id=orientation_id,
                object_pose=object_pose,
                args=args,
            )
            for grasp in result.accepted
            if float(grasp.jaw_width) <= float(args.max_training_jaw_width)
        ]
        # Match the execution runner's hard pregrasp-height gate here. A target
        # that cannot be passed back to the real MoveIt/Isaac path must never
        # enter the training catalog merely because its contact pose clears the
        # ground.
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate["world_grasp"]["pregrasp_position_w"][2])
            > float(args.minimum_pregrasp_height)
        ]
        anchors = ("g1973", "g1875") if orientation_id == "current" else ()
        selected_count = (
            targets_per_orientation
            if args.target_count > 0
            else min(len(candidates), targets_per_orientation)
        )
        if len(candidates) < selected_count:
            raise ValueError(
                f"{orientation_id} has only {len(candidates)} candidates, but "
                f"{selected_count} are required."
            )
        ranked_count = min(
            len(candidates),
            selected_count + int(args.alternates_per_orientation),
        )
        ranked = select_diverse_targets(
            candidates,
            count=ranked_count,
            anchor_grasp_ids=anchors,
        )
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
        orientation_records.append(
            {
                "orientation_id": orientation_id,
                "object_pose_world": _pose_payload(object_pose),
                **orientation_metadata,
                "ground_feasible_count": len(result.accepted),
                "training_width_feasible_count": len(candidates),
                "selected_target_count": len(selected),
                "selected_grasp_ids": [str(target["grasp_id"]) for target in selected],
                "alternate_target_count": len(alternates),
                "alternate_grasp_ids": [str(target["grasp_id"]) for target in alternates],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "target_mesh_path": stage1.target_mesh_path,
        "mesh_scale": float(stage1.mesh_scale),
        "source_stage1_bundle": str(args.stage1_bundle.resolve()),
        "source_current_stage2_bundle": str(args.current_stage2_bundle.resolve()),
        "selection": {
            "target_count": len(selected_targets),
            "orientation_count": len(orientation_records),
            "targets_per_orientation": targets_per_orientation,
            "targets_per_orientation_is_cap": args.target_count == 0,
            "max_training_jaw_width_m": float(args.max_training_jaw_width),
            "minimum_pregrasp_height_m": float(args.minimum_pregrasp_height),
            "alternates_per_orientation": int(args.alternates_per_orientation),
            "method": "balanced_score_seeded_farthest_point",
            "moveit_validation_required": True,
            "isaac_goal_capture_required": True,
            "gripper_collision_model": gripper_collision_model,
        },
        "orientations": orientation_records,
        "targets": selected_targets,
        "alternates": alternate_targets,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(manifest['targets'])} diverse targets across "
        f"{len(manifest['orientations'])} orientations to {args.output}."
    )
    for orientation in manifest["orientations"]:
        print(
            f"  {orientation['orientation_id']}: feasible={orientation['ground_feasible_count']} "
            f"selected={orientation['selected_grasp_ids']}"
        )


if __name__ == "__main__":
    main()
