#!/usr/bin/env python3
"""MoveIt-validate ranked grasps in every catalog part/orientation group."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_assembly_multigrasp_manifest import (  # noqa: E402
    SPLIT_NAMES,
    _group_split_assignment,
)
from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate  # noqa: E402
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    rclpy,
)
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_manifest.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_planned.json",
    )
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_plans",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=REPO_ROOT / "configs/grasp_pipeline_sim_isaac.yaml",
    )
    parser.add_argument(
        "--targets-per-orientation",
        type=int,
        default=10,
        help=(
            "Required successes per orientation. Use 0 to preserve each manifest "
            "orientation's selected_target_count, allowing variable group sizes."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Replan targets even when a matching plan JSON already exists.",
    )
    parser.add_argument(
        "--quiet-cached",
        action="store_true",
        help="Suppress one log line per successfully reused target.",
    )
    parser.add_argument(
        "--allow-variable-counts",
        action="store_true",
        help=(
            "Treat the per-orientation request as a maximum, retain groups that "
            "reach --minimum-targets-per-orientation, and drop unreachable groups."
        ),
    )
    parser.add_argument("--minimum-targets-per-orientation", type=int, default=1)
    parser.add_argument(
        "--ik-timeout-s",
        type=float,
        default=None,
        help="Override the pipeline IK timeout for large offline catalog sweeps.",
    )
    parser.add_argument(
        "--exclusions-file",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_goal_exclusions.json",
        help="Persistent target IDs rejected by Isaac goal-image quality checks.",
    )
    parser.add_argument(
        "--exclude-target-id",
        action="append",
        default=[],
        help="Additional target ID to skip; may be repeated.",
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


def _moveit_settings(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = dict(payload["isaac_execution"])
    return settings


def _commander_config(settings: dict[str, object]) -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=str(settings["moveit_planning_group"]),
        pose_link=str(settings["moveit_pose_link"]),
        joint_names=tuple(str(value) for value in settings["moveit_joint_names"]),
        moveit_namespace=str(settings.get("moveit_namespace", "")),
        pipeline_id=str(settings.get("moveit_pipeline_id", "")),
        planner_id=str(settings.get("moveit_planner_id", "")),
        wait_for_moveit_timeout_s=float(settings["moveit_wait_for_moveit_timeout_s"]),
        ik_timeout_s=float(settings["moveit_ik_timeout_s"]),
        fk_timeout_s=float(settings["moveit_ik_timeout_s"]),
        planning_time_s=float(settings["moveit_planning_time_s"]),
        num_planning_attempts=int(settings["moveit_num_planning_attempts"]),
        velocity_scale=float(settings["moveit_velocity_scale"]),
        acceleration_scale=float(settings["moveit_acceleration_scale"]),
        post_execute_sleep_s=0.0,
        avoid_collisions=not bool(settings.get("moveit_allow_collisions", False)),
    )


def _world_grasp(target: dict[str, object]) -> WorldFrameGraspCandidate:
    payload = target["world_grasp"]
    return WorldFrameGraspCandidate(
        grasp_id=str(target["grasp_id"]),
        position_w=tuple(float(value) for value in payload["position_w"]),
        orientation_xyzw=tuple(float(value) for value in payload["orientation_xyzw"]),
        normal_w=tuple(float(value) for value in payload["approach_axis_w"]),
        pregrasp_offset=float(payload["pregrasp_offset_m"]),
        pregrasp_position_w=tuple(
            float(value) for value in payload["pregrasp_position_w"]
        ),
        gripper_width=float(payload["gripper_width_m"]),
        jaw_width=float(payload["jaw_width_m"]),
        roll_angle_rad=0.0,
        contact_point_a_w=(0.0, 0.0, 0.0),
        contact_point_b_w=(0.0, 0.0, 0.0),
    )


def _trajectory_waypoints(trajectory, joint_names: tuple[str, ...]) -> list[list[float]]:
    source_names = tuple(str(value) for value in trajectory.joint_trajectory.joint_names)
    indices = []
    for name in joint_names:
        if name not in source_names:
            raise RuntimeError(f"MoveIt trajectory is missing joint '{name}'.")
        indices.append(source_names.index(name))
    waypoints = [
        [float(point.positions[index]) for index in indices]
        for point in trajectory.joint_trajectory.points
    ]
    if not waypoints:
        raise RuntimeError("MoveIt returned an empty joint trajectory.")
    return waypoints


def _matching_existing_plan(path: Path, target_id: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("target_id") != target_id:
            return None
        trajectories = payload["trajectories"]
        if not trajectories["pregrasp"] or not trajectories["grasp"]:
            return None
        return payload
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _plan_target(
    commander: MoveItPoseCommander,
    commander_cfg: MoveItPoseCommanderConfig,
    target: dict[str, object],
    settings: dict[str, object],
) -> dict[str, object]:
    world_grasp = _world_grasp(target)
    pose_targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=str(settings["moveit_frame_id"]),
        lift_height_m=float(settings.get("moveit_lift_height_m", 0.08)),
        position_signs=tuple(
            float(value) for value in settings["moveit_target_position_signs"]
        ),
        tcp_to_grasp_offset=(0.0, 0.0, 0.0),
    )
    start = tuple(float(value) for value in settings["moveit_start_joint_positions"])
    trajectories: dict[str, list[list[float]]] = {}
    for label in ("pregrasp", "grasp"):
        trajectory, message = commander.plan_to_pose(
            pose_targets[label],
            label=f"multigrasp_{target['target_id']}_{label}",
            start_joint_positions=start,
        )
        if trajectory is None:
            raise RuntimeError(f"{label}: {message}")
        waypoints = _trajectory_waypoints(trajectory, commander_cfg.joint_names)
        trajectories[label] = waypoints
        start = tuple(waypoints[-1])
    return {
        "schema_version": 1,
        "target_id": str(target["target_id"]),
        "orientation_id": str(target["orientation_id"]),
        "local_orientation_id": str(
            target.get("local_orientation_id", target["orientation_id"])
        ),
        "assembly_name": str(target.get("assembly_name", "")),
        "part_id": str(target.get("part_id", "")),
        "part_index": int(target.get("part_index", 0)),
        "split": str(target.get("split", "train")),
        "selected_grasp_id": str(target["grasp_id"]),
        "selected_world_grasp": target["world_grasp"],
        "object_pose_world": target["object_pose_world"],
        "joint_names": list(commander_cfg.joint_names),
        "start_joint_positions": list(settings["moveit_start_joint_positions"]),
        "trajectories": trajectories,
        "moveit": {
            "frame_id": str(settings["moveit_frame_id"]),
            "planning_group": commander_cfg.planning_group,
            "pose_link": commander_cfg.pose_link,
            "namespace": commander_cfg.moveit_namespace,
            "allow_collisions": not commander_cfg.avoid_collisions,
        },
    }


def main() -> None:  # noqa: C901 - explicit staged catalog validation
    args = parse_args()
    if args.targets_per_orientation < 0:
        raise ValueError("--targets-per-orientation must be non-negative.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.minimum_targets_per_orientation < 1:
        raise ValueError("--minimum-targets-per-orientation must be at least one.")
    cached_failures: dict[str, str] = {}
    if not args.no_resume and args.output_manifest.is_file():
        try:
            prior_output = json.loads(args.output_manifest.read_text(encoding="utf-8"))
            cached_failures = {
                str(item["target_id"]): str(item["error"])
                for item in prior_output.get("moveit_validation", {}).get("failures", [])
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            cached_failures = {}
    excluded_target_ids = set(str(value) for value in args.exclude_target_id)
    if args.exclusions_file.is_file():
        exclusions = json.loads(args.exclusions_file.read_text(encoding="utf-8"))
        for rejection in exclusions.get("rejections", []):
            excluded_target_ids.add(str(rejection["target_id"]))
    if excluded_target_ids:
        print(
            f"[INFO] Excluding {len(excluded_target_ids)} visually invalid targets: "
            f"{sorted(excluded_target_ids)}",
            flush=True,
        )
    orientation_ids = [
        str(item["orientation_id"])
        for item in manifest["orientations"]
        if int(item["selected_target_count"]) > 0
    ]
    required_by_orientation = {
        str(item["orientation_id"]): (
            int(args.targets_per_orientation)
            if args.targets_per_orientation > 0
            else int(item["selected_target_count"])
        )
        for item in manifest["orientations"]
    }
    queues: dict[str, list[dict[str, object]]] = defaultdict(list)
    for target in manifest["targets"] + manifest.get("alternates", []):
        queues[str(target["orientation_id"])].append(target)
    settings = _moveit_settings(args.pipeline_config)
    if args.ik_timeout_s is not None:
        if args.ik_timeout_s <= 0.0:
            raise ValueError("--ik-timeout-s must be positive.")
        settings["moveit_ik_timeout_s"] = float(args.ik_timeout_s)
    commander_cfg = _commander_config(settings)
    if rclpy is None:
        raise RuntimeError(
            "ROS2/MoveIt Python packages are unavailable. Source the LBR MoveIt workspace "
            "before running this command."
        )

    # Keep this standalone catalog tool usable in restricted/containerized
    # sessions where ~/.ros is read-only.
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros-log")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    initialized_here = False
    commander = None
    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        commander = MoveItPoseCommander(
            commander_cfg, node_name="multigrasp_catalog_planner"
        )
        commander.wait_for_moveit(require_execute=False)
        for orientation_id in orientation_ids:
            required_count = required_by_orientation[orientation_id]
            orientation_successes = 0
            print(
                f"[PLAN] orientation={orientation_id} required={required_count}",
                flush=True,
            )
            for target in queues[orientation_id]:
                if orientation_successes >= required_count:
                    break
                target_id = str(target["target_id"])
                if target_id in cached_failures:
                    failures.append(
                        {"target_id": target_id, "error": cached_failures[target_id]}
                    )
                    continue
                if str(target["target_id"]) in excluded_target_ids:
                    failures.append(
                        {
                            "target_id": str(target["target_id"]),
                            "error": "excluded_by_isaac_goal_quality",
                        }
                    )
                    continue
                plan_path = args.plans_dir / f"{target['target_id']}.json"
                existing = None if args.no_resume else _matching_existing_plan(
                    plan_path, str(target["target_id"])
                )
                try:
                    plan = existing or _plan_target(
                        commander, commander_cfg, target, settings
                    )
                except Exception as exc:  # keep trying ranked replacements
                    message = str(exc)
                    failures.append(
                        {"target_id": str(target["target_id"]), "error": message}
                    )
                    print(f"[REJECT] {target['target_id']}: {message}", flush=True)
                    continue
                if existing is None:
                    _atomic_write_json(plan_path, plan)
                validated = json.loads(json.dumps(target))
                validated["validation"]["moveit_plan_validated"] = True
                validated["moveit_plan_path"] = str(plan_path.resolve())
                validated["orientation_selection_rank"] = orientation_successes
                successes.append(validated)
                orientation_successes += 1
                source = "cached" if existing is not None else "planned"
                if not (args.quiet_cached and existing is not None):
                    print(
                        f"[ACCEPT] {target['target_id']} ({orientation_successes}/"
                        f"{required_count}, {source})",
                        flush=True,
                    )
            if orientation_successes < required_count:
                print(
                    f"[ERROR] {orientation_id} produced only {orientation_successes}/"
                    f"{required_count} MoveIt-valid targets.",
                    flush=True,
                )
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()

    successes_by_orientation = Counter(
        str(target["orientation_id"]) for target in successes
    )
    requested_target_count = sum(required_by_orientation.values())
    shortfalls = {
        orientation_id: {
            "requested": required_count,
            "validated": int(successes_by_orientation[orientation_id]),
        }
        for orientation_id, required_count in required_by_orientation.items()
        if successes_by_orientation[orientation_id] < required_count
    }
    surviving_orientation_ids = {
        orientation_id
        for orientation_id in orientation_ids
        if successes_by_orientation[orientation_id]
        >= args.minimum_targets_per_orientation
    }
    expected_part_ids = {
        str(part["part_id"]) for part in manifest.get("parts", [])
    }
    successful_part_ids = {
        str(target.get("part_id", "")) for target in successes
    }
    variable_complete = bool(surviving_orientation_ids) and (
        not expected_part_ids or expected_part_ids <= successful_part_ids
    )
    complete = (
        variable_complete
        if args.allow_variable_counts
        else len(successes) == requested_target_count
    )

    # Reachability filtering changes each stratum's size. Recompute the split
    # over only the survivors while retaining the hard no-leakage rule: an
    # exact (part, local grasp) group belongs to exactly one split even when it
    # appears in several stable orientations.
    if successes and "split" in manifest and all("part_id" in target for target in successes):
        requested_fractions = manifest["split"]["requested_fractions"]
        assignment, split_salt = _group_split_assignment(
            successes,
            coverage_targets=successes,
            train_fraction=float(requested_fractions["train"]),
            validation_fraction=float(requested_fractions["validation"]),
            seed=int(manifest["split"]["seed"]),
        )
        for target in successes:
            target["split"] = assignment[
                (str(target["part_id"]), str(target["grasp_id"]))
            ]
            # Keep the human-readable plan artifact consistent with the
            # authoritative planned manifest used by the asset builders.
            plan_path = Path(str(target["moveit_plan_path"]))
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_payload["split"] = str(target["split"])
            _atomic_write_json(plan_path, plan_payload)
    else:
        split_salt = int(manifest.get("split", {}).get("salt", 0))

    split_counts = Counter(str(target.get("split", "train")) for target in successes)
    part_split_counts: dict[str, dict[str, int]] = {}
    planned_parts = json.loads(json.dumps(manifest.get("parts", [])))
    for part in planned_parts:
        part_id = str(part["part_id"])
        part_split_counts[part_id] = {
            split: sum(
                str(target.get("part_id", "")) == part_id
                and str(target.get("split", "train")) == split
                for target in successes
            )
            for split in SPLIT_NAMES
        }
        part["selected_target_count"] = sum(
            str(target.get("part_id", "")) == part_id for target in successes
        )

    planned_orientations = [
        orientation
        for orientation in json.loads(json.dumps(manifest["orientations"]))
        if not args.allow_variable_counts
        or str(orientation["orientation_id"]) in surviving_orientation_ids
    ]
    orientation_split_counts: dict[str, dict[str, int]] = {}
    for orientation in planned_orientations:
        orientation_id = str(orientation["orientation_id"])
        orientation["selected_target_count"] = int(
            successes_by_orientation[orientation_id]
        )
        counts = {
            split: sum(
                str(target["orientation_id"]) == orientation_id
                and str(target.get("split", "train")) == split
                for target in successes
            )
            for split in SPLIT_NAMES
        }
        orientation["split_target_counts"] = counts
        orientation_split_counts[orientation_id] = counts
    for part in planned_parts:
        part["orientation_count"] = sum(
            str(orientation.get("part_id", "")) == str(part["part_id"])
            for orientation in planned_orientations
        )

    planned_split = json.loads(json.dumps(manifest.get("split", {})))
    if planned_split:
        planned_split.update(
            {
                "salt": split_salt,
                "target_counts": {
                    split: int(split_counts[split]) for split in SPLIT_NAMES
                },
                "part_target_counts": part_split_counts,
                "orientation_target_counts": orientation_split_counts,
            }
        )
    selection = json.loads(json.dumps(manifest.get("selection", {})))
    if selection:
        selection["requested_selected_target_count"] = int(
            selection.get("selected_target_count", requested_target_count)
        )
        selection["selected_target_count"] = len(successes)

    planned_manifest = {
        **manifest,
        "schema_version": 5 if "parts" in manifest else 3,
        "source_geometry_manifest": str(args.manifest.resolve()),
        "parts": planned_parts,
        "selection": selection,
        "split": planned_split,
        "orientations": planned_orientations,
        "targets": successes,
        "alternates": [],
        "moveit_validation": {
            "complete": complete,
            "variable_counts_allowed": bool(args.allow_variable_counts),
            "minimum_targets_per_orientation": int(
                args.minimum_targets_per_orientation
            ),
            "validated_target_count": len(successes),
            "required_target_count": len(successes) if complete else requested_target_count,
            "requested_target_count": requested_target_count,
            "required_by_orientation": required_by_orientation,
            "shortfalls": shortfalls,
            "dropped_orientation_ids": sorted(
                set(orientation_ids) - surviving_orientation_ids
            ),
            "failures": failures,
        },
    }
    _atomic_write_json(args.output_manifest, planned_manifest)
    if not complete:
        raise RuntimeError(
            f"Only {len(successes)} MoveIt-valid targets were found; partial results are in "
            f"{args.output_manifest}."
        )
    print(
        f"[DONE] Validated {len(successes)}/{requested_target_count} requested targets "
        f"across {len(orientation_ids)} orientations; manifest={args.output_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
