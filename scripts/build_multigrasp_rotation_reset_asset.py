#!/usr/bin/env python3
"""Build position-preserving rotational reset paths for all grasp targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_reset_trajectory_asset import (  # noqa: E402
    MOVEIT_TO_ISAAC_SIGNS,
    _robot_tcp_transform_link7,
)
from grasp_planning.grasping.collision import (  # noqa: E402
    BoxCollisionPrimitive,
    GraspCollisionEvaluator,
    MeshCollisionPrimitive,
    PdzGripperCollisionModel,
)
from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402
from grasp_planning.grasping.finger_geometry import finger_box_corners  # noqa: E402
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.mujoco import build_bundle_local_mesh  # noqa: E402
from grasp_planning.start_poses import (  # noqa: E402
    PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    PDZ_GRIPPER_APPROACH_PROFILE,
    VISUAL_SERVO_GRIPPER_PROFILE,
)

ROTATION_RESET_SCHEMA_VERSION = 4
AXIS_SELECTION_METHOD = "fibonacci_farthest_point_v1"
COLLISION_VALIDATION_PROFILE = "pdz_gripper_object_ground_clearance_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-asset",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_paths.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/multigrasp_50_rotation_resets.npz",
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=REPO_ROOT
        / "assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf",
    )
    parser.add_argument("--variants", type=int, default=16)
    parser.add_argument("--far-rotation-deg", type=float, default=15.0)
    parser.add_argument("--near-rotation-deg", type=float, default=5.0)
    parser.add_argument("--rotation-exponent", type=float, default=1.5)
    parser.add_argument("--maximum-iterations", type=int, default=250)
    parser.add_argument(
        "--minimum-collision-clearance-m",
        type=float,
        default=0.001,
        help=(
            "Required separation between the approach-width gripper and both "
            "the target part and ground at every authored reset waypoint."
        ),
    )
    parser.add_argument(
        "--minimum-distinct-variants",
        type=int,
        default=None,
        help=(
            "Minimum feasible axes required before padding the fixed-size pool by "
            "cycling those axes. Defaults to --variants (no padding)."
        ),
    )
    parser.add_argument("--quiet-cached", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional per-target cache directory for resumable large catalogs.",
    )
    return parser.parse_args()


def _primitive_vertices(
    primitive: BoxCollisionPrimitive | MeshCollisionPrimitive,
) -> np.ndarray:
    if isinstance(primitive, BoxCollisionPrimitive):
        return finger_box_corners(
            primitive.center_obj,
            primitive.rotation_obj,
            primitive.half_extents,
        )
    return np.asarray(primitive.vertices_obj, dtype=np.float64)


def _minimum_gripper_clearance(
    *,
    object_scene,
    collision_model: PdzGripperCollisionModel,
    tcp_position_w: np.ndarray,
    tcp_rotation_w: np.ndarray,
    jaw_width_m: float,
) -> float:
    """Return minimum object/ground separation for one approach-width pose."""

    half_jaw = 0.5 * float(jaw_width_m)
    contact_a = np.asarray((0.0, -half_jaw, 0.0), dtype=np.float64)
    contact_b = np.asarray((0.0, half_jaw, 0.0), dtype=np.float64)
    minimum_clearance = np.inf
    for primitive in collision_model.primitives_for_grasp(
        grasp_rotmat=np.asarray(tcp_rotation_w, dtype=np.float64),
        contact_point_a=contact_a,
        contact_point_b=contact_b,
        grasp_center=np.asarray(tcp_position_w, dtype=np.float64),
    ):
        if isinstance(primitive, BoxCollisionPrimitive):
            object_clearance = object_scene.minimum_distance_to_box(primitive)
        else:
            object_clearance = object_scene.minimum_distance_to_mesh(primitive)
        ground_clearance = float(np.min(_primitive_vertices(primitive)[:, 2]))
        minimum_clearance = min(
            minimum_clearance,
            float(object_clearance),
            ground_clearance,
        )
    return float(minimum_clearance)


def _resolve_source_manifest(paths_asset: Path, source_manifest_value: object) -> Path:
    recorded = Path(str(np.asarray(source_manifest_value).item())).expanduser()
    candidates = (recorded, paths_asset.with_name("planned_manifest.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot load the planned manifest needed for reset collision validation. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Return deterministic approximately uniform unit axes on the sphere."""

    if count < 2:
        raise ValueError("--variants must be at least two.")
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = indices * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z), axis=1)


def _space_filling_axis_order(axes: np.ndarray) -> np.ndarray:
    """Order candidate axes so every short prefix covers the whole sphere.

    The raw Fibonacci construction is uniform only as a complete set: its
    first entries all lie near +Z.  The IK builder stops after the requested
    number of feasible entries, so using raw order created a strong +Z bias.
    Greedy farthest-point ordering retains deterministic/reproducible assets
    while spreading even a 16-axis prefix over both hemispheres.
    """

    values = np.asarray(axes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("Candidate axes must have shape (N, 3) with N >= 2.")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1.0e-8):
        raise ValueError("Candidate axes must be normalized.")

    # A non-cardinal seed avoids privileging any one world coordinate axis.
    seed = np.asarray((1.0, 1.0, 1.0), dtype=np.float64)
    seed /= np.linalg.norm(seed)
    current = int(np.argmax(values @ seed))
    selected = np.empty(len(values), dtype=np.int64)
    chosen = np.zeros(len(values), dtype=np.bool_)
    minimum_distance_squared = np.full(len(values), np.inf, dtype=np.float64)
    for output_index in range(len(values)):
        selected[output_index] = current
        chosen[current] = True
        distance_squared = np.sum((values - values[current]) ** 2, axis=1)
        minimum_distance_squared = np.minimum(
            minimum_distance_squared, distance_squared
        )
        minimum_distance_squared[chosen] = -np.inf
        if output_index + 1 < len(values):
            current = int(np.argmax(minimum_distance_squared))
    return values[selected]


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:  # noqa: C901 - asset validation is one deliberate linear pipeline
    args = parse_args()
    if args.far_rotation_deg <= 0.0:
        raise ValueError("--far-rotation-deg must be positive.")
    if not 0.0 <= args.near_rotation_deg <= args.far_rotation_deg:
        raise ValueError("Rotation limits must satisfy 0 <= near <= far.")
    if args.rotation_exponent <= 0.0:
        raise ValueError("--rotation-exponent must be positive.")
    if args.minimum_collision_clearance_m < 0.0:
        raise ValueError("--minimum-collision-clearance-m must be non-negative.")
    minimum_distinct_variants = (
        args.variants
        if args.minimum_distinct_variants is None
        else int(args.minimum_distinct_variants)
    )
    if not 2 <= minimum_distinct_variants <= args.variants:
        raise ValueError(
            "--minimum-distinct-variants must be between two and --variants."
        )

    paths_asset = args.paths_asset.expanduser().resolve()
    with np.load(paths_asset, allow_pickle=False) as source:
        target_ids = source["target_ids"].copy()
        nominal_isaac = source["reset_joint_trajectories"].astype(np.float64)
        progress = source["reset_path_progress"].astype(np.float64)
        jaw_widths = source["grasp_jaw_widths_m"].astype(np.float64)
        approach_widths = source["approach_gripper_widths_m"].astype(np.float64)
        approach_profile = str(
            np.asarray(source.get("approach_gripper_profile", "")).item()
        )
        approach_clearance_per_finger = float(
            np.asarray(source.get("approach_clearance_per_finger_m", np.nan)).item()
        )
        object_positions = source["object_positions_w"].astype(np.float64)
        object_orientations = source["object_orientations_xyzw_w"].astype(np.float64)
        part_indices = source.get(
            "part_indices", np.zeros(len(target_ids), dtype=np.int64)
        ).astype(np.int64)
        source_manifest_value = source.get("source_planned_manifest", "")
    if nominal_isaac.ndim != 3 or nominal_isaac.shape[-1] != 7:
        raise ValueError(
            f"Expected nominal paths shaped (G,N,7), got {nominal_isaac.shape}."
        )
    if progress.shape != (nominal_isaac.shape[1],):
        raise ValueError("reset_path_progress does not match the nominal paths.")
    if jaw_widths.shape != (len(target_ids),) or approach_widths.shape != (
        len(target_ids),
    ):
        raise ValueError("Path asset must contain one jaw and approach width per target.")
    if approach_profile != PDZ_GRIPPER_APPROACH_PROFILE:
        raise ValueError(
            f"Path asset approach profile is '{approach_profile or 'unlabeled'}'; "
            f"expected '{PDZ_GRIPPER_APPROACH_PROFILE}'."
        )
    if abs(
        approach_clearance_per_finger
        - PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
    ) > 1.0e-7:
        raise ValueError("Path asset must encode exactly 5 mm clearance per finger.")
    if not np.allclose(
        approach_widths - jaw_widths,
        2.0 * PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise ValueError("Path asset approach apertures must equal jaw width + 10 mm.")

    source_manifest_path = _resolve_source_manifest(
        paths_asset, source_manifest_value
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    part_records = list(source_manifest.get("parts", []))
    if not part_records:
        raise ValueError(
            "Reset collision validation requires a multipart manifest with part meshes."
        )
    if np.any(part_indices < 0) or np.any(part_indices >= len(part_records)):
        raise ValueError("Path asset part_indices are outside the planned manifest parts.")

    part_meshes: list[TriangleMesh] = []
    for part in part_records:
        bundle_path = Path(str(part["source_current_stage2_bundle"]))
        if not bundle_path.is_file():
            fallback = (
                source_manifest_path.parent
                / "sources"
                / f"part_{part['part_id']}_stage2.json"
            )
            bundle_path = fallback
        bundle = load_grasp_bundle(bundle_path)
        part_meshes.append(build_bundle_local_mesh(bundle))

    collision_model = PdzGripperCollisionModel(
        contact_gap_m=PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
    )
    collision_evaluator = GraspCollisionEvaluator(collision_model)
    collision_scene_cache: dict[tuple[object, ...], object] = {}
    target_collision_scenes: list[object] = []
    for target_index in range(len(target_ids)):
        pose_key = (
            int(part_indices[target_index]),
            *np.round(object_positions[target_index], 9).tolist(),
            *np.round(object_orientations[target_index], 9).tolist(),
        )
        scene = collision_scene_cache.get(pose_key)
        if scene is None:
            object_pose = ObjectWorldPose(
                position_world=tuple(object_positions[target_index].tolist()),
                orientation_xyzw_world=tuple(
                    object_orientations[target_index].tolist()
                ),
            )
            part_mesh = part_meshes[int(part_indices[target_index])]
            world_mesh = TriangleMesh(
                vertices_obj=object_pose.transform_points_to_world(
                    part_mesh.vertices_obj
                ),
                faces=np.asarray(part_mesh.faces, dtype=np.int64),
            )
            scene = collision_evaluator.build_scene(world_mesh)
            collision_scene_cache[pose_key] = scene
        target_collision_scenes.append(scene)

    robot_urdf = args.robot_urdf.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(robot_urdf))
    data = mujoco.MjData(model)
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    if link7_id < 0:
        raise ValueError("Robot URDF did not produce a link7 MuJoCo body.")
    urdf_root = ElementTree.parse(robot_urdf).getroot()
    tcp_offset_link7, tcp_rotation_link7, _tcp_link = _robot_tcp_transform_link7(
        urdf_root
    )
    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]
    # Assembly-scale catalogs include targets close to different joint limits.
    # Search a larger deterministic sphere so each target still gets a full,
    # fixed-size variant pool instead of failing on the first 128 directions.
    candidate_axes = _space_filling_axis_order(
        _fibonacci_sphere(max(512, args.variants * 32))
    )
    angle_profile = np.deg2rad(args.near_rotation_deg) + (
        np.deg2rad(args.far_rotation_deg - args.near_rotation_deg)
        * np.power(1.0 - progress, args.rotation_exponent)
    )

    def pose_and_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        link7_rotation = data.xmat[link7_id].reshape(3, 3).copy()
        position = data.xpos[link7_id].copy() + link7_rotation @ tcp_offset_link7
        rotation = link7_rotation @ tcp_rotation_link7
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            position,
            link7_id,
        )
        return (
            position,
            rotation,
            np.vstack((jacobian_position[:, :7], jacobian_rotation[:, :7])),
        )

    nominal_moveit = nominal_isaac * MOVEIT_TO_ISAAC_SIGNS[None, None, :]
    reset_variants = np.empty(
        (len(target_ids), args.variants, len(progress), 7), dtype=np.float32
    )
    position_residuals = np.empty(
        (len(target_ids), args.variants, len(progress)), dtype=np.float32
    )
    rotation_residuals = np.empty_like(position_residuals)
    collision_clearances = np.empty_like(position_residuals)
    nominal_collision_clearances = np.empty(
        (len(target_ids), len(progress)), dtype=np.float32
    )
    selected_axes = np.empty(
        (len(target_ids), args.variants, 3), dtype=np.float32
    )
    target_valid = np.zeros(len(target_ids), dtype=np.bool_)
    for target_index, target_id in enumerate(target_ids.tolist()):
        cache_path = None
        if args.cache_dir is not None:
            cache_path = args.cache_dir.expanduser().resolve() / f"{target_id}.npz"
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    cache_schema_version = int(
                        np.asarray(cached.get("schema_version", -1)).item()
                    )
                    cache_axis_selection_method = str(
                        np.asarray(cached.get("axis_selection_method", "")).item()
                    )
                    cache_nominal = cached["nominal_joint_trajectory"].astype(
                        np.float64
                    )
                    cache_profile = cached["rotation_angle_profile_rad"].astype(
                        np.float64
                    )
                    cache_axes = cached["rotation_axes_w"].astype(np.float32)
                    cache_joints = cached["rotation_joint_trajectories"].astype(
                        np.float32
                    )
                    cache_position = cached["ik_position_residual_m"].astype(
                        np.float32
                    )
                    cache_rotation = cached["ik_rotation_residual_rad"].astype(
                        np.float32
                    )
                    cache_collision_profile = str(
                        np.asarray(cached.get("collision_validation_profile", "")).item()
                    )
                    cache_collision_threshold = float(
                        np.asarray(cached.get("minimum_collision_clearance_m", np.nan)).item()
                    )
                    cache_approach_width = float(
                        np.asarray(cached.get("approach_gripper_width_m", np.nan)).item()
                    )
                    cache_clearances = np.asarray(
                        cached.get("collision_clearance_m", np.asarray([])),
                        dtype=np.float32,
                    )
                    cache_nominal_clearances = np.asarray(
                        cached.get(
                            "nominal_collision_clearance_m", np.asarray([])
                        ),
                        dtype=np.float32,
                    )
                cached_variant_count = int(cache_axes.shape[0])
                if (
                    cache_schema_version == ROTATION_RESET_SCHEMA_VERSION
                    and cache_axis_selection_method == AXIS_SELECTION_METHOD
                    and cache_collision_profile == COLLISION_VALIDATION_PROFILE
                    and abs(
                        cache_collision_threshold
                        - float(args.minimum_collision_clearance_m)
                    )
                    < 1.0e-9
                    and abs(cache_approach_width - approach_widths[target_index])
                    < 1.0e-7
                    and np.array_equal(cache_nominal, nominal_isaac[target_index])
                    and np.allclose(cache_profile, angle_profile, atol=1.0e-7)
                    and cached_variant_count >= args.variants
                    and cache_axes.shape == (cached_variant_count, 3)
                    and cache_joints.shape == (cached_variant_count, len(progress), 7)
                    and cache_position.shape == (cached_variant_count, len(progress))
                    and cache_rotation.shape == (cached_variant_count, len(progress))
                    and cache_clearances.shape == (
                        cached_variant_count,
                        len(progress),
                    )
                    and cache_nominal_clearances.shape == (len(progress),)
                    and np.all(
                        cache_clearances
                        >= float(args.minimum_collision_clearance_m) - 1.0e-7
                    )
                    and np.all(
                        cache_nominal_clearances
                        >= float(args.minimum_collision_clearance_m) - 1.0e-7
                    )
                ):
                    selected_axes[target_index] = cache_axes[: args.variants]
                    reset_variants[target_index] = cache_joints[: args.variants]
                    position_residuals[target_index] = cache_position[: args.variants]
                    rotation_residuals[target_index] = cache_rotation[: args.variants]
                    collision_clearances[target_index] = cache_clearances[
                        : args.variants
                    ]
                    nominal_collision_clearances[target_index] = (
                        cache_nominal_clearances
                    )
                    target_valid[target_index] = True
                    if not args.quiet_cached and not args.quiet:
                        print(
                            f"[ROTATION RESET] {target_index + 1:04d}/{len(target_ids)} "
                            f"{target_id} cached",
                            flush=True,
                        )
                    continue
                legacy_cache_matches = (
                    cache_schema_version == 2
                    and cache_axis_selection_method == AXIS_SELECTION_METHOD
                    and np.array_equal(cache_nominal, nominal_isaac[target_index])
                    and np.allclose(cache_profile, angle_profile, atol=1.0e-7)
                    and cache_axes.shape == (cached_variant_count, 3)
                    and cache_joints.shape
                    == (cached_variant_count, len(progress), 7)
                    and cache_position.shape
                    == (cached_variant_count, len(progress))
                    and cache_rotation.shape
                    == (cached_variant_count, len(progress))
                )
                if legacy_cache_matches:
                    base_poses = [
                        pose_and_jacobian(q)[:2]
                        for q in nominal_moveit[target_index]
                    ]
                    legacy_nominal_clearances = np.asarray(
                        [
                            _minimum_gripper_clearance(
                                object_scene=target_collision_scenes[target_index],
                                collision_model=collision_model,
                                tcp_position_w=position,
                                tcp_rotation_w=rotation,
                                jaw_width_m=float(jaw_widths[target_index]),
                            )
                            for position, rotation in base_poses
                        ],
                        dtype=np.float32,
                    )
                    safe_indices: list[int] = []
                    safe_clearances: list[np.ndarray] = []
                    if np.all(
                        legacy_nominal_clearances
                        >= float(args.minimum_collision_clearance_m) - 1.0e-7
                    ):
                        for variant_index in range(cached_variant_count):
                            variant_clearances = np.asarray(
                                [
                                    _minimum_gripper_clearance(
                                        object_scene=target_collision_scenes[
                                            target_index
                                        ],
                                        collision_model=collision_model,
                                        tcp_position_w=position,
                                        tcp_rotation_w=rotation,
                                        jaw_width_m=float(jaw_widths[target_index]),
                                    )
                                    for position, rotation in (
                                        pose_and_jacobian(
                                            q * MOVEIT_TO_ISAAC_SIGNS
                                        )[:2]
                                        for q in cache_joints[variant_index]
                                    )
                                ],
                                dtype=np.float32,
                            )
                            if np.all(
                                variant_clearances
                                >= float(args.minimum_collision_clearance_m)
                                - 1.0e-7
                            ):
                                safe_indices.append(variant_index)
                                safe_clearances.append(variant_clearances)
                    if len(safe_indices) >= minimum_distinct_variants:
                        for output_index in range(args.variants):
                            safe_offset = output_index % len(safe_indices)
                            source_index = safe_indices[safe_offset]
                            selected_axes[target_index, output_index] = cache_axes[
                                source_index
                            ]
                            reset_variants[target_index, output_index] = cache_joints[
                                source_index
                            ]
                            position_residuals[target_index, output_index] = (
                                cache_position[source_index]
                            )
                            rotation_residuals[target_index, output_index] = (
                                cache_rotation[source_index]
                            )
                            collision_clearances[target_index, output_index] = (
                                safe_clearances[safe_offset]
                            )
                        nominal_collision_clearances[target_index] = (
                            legacy_nominal_clearances
                        )
                        target_valid[target_index] = True
                        _atomic_savez(
                            cache_path,
                            {
                                "schema_version": np.asarray(
                                    ROTATION_RESET_SCHEMA_VERSION, dtype=np.int64
                                ),
                                "axis_selection_method": np.asarray(
                                    AXIS_SELECTION_METHOD
                                ),
                                "target_id": np.asarray(str(target_id)),
                                "nominal_joint_trajectory": nominal_isaac[
                                    target_index
                                ].astype(np.float32),
                                "rotation_angle_profile_rad": angle_profile.astype(
                                    np.float32
                                ),
                                "rotation_axes_w": selected_axes[target_index],
                                "rotation_joint_trajectories": reset_variants[
                                    target_index
                                ],
                                "ik_position_residual_m": position_residuals[
                                    target_index
                                ],
                                "ik_rotation_residual_rad": rotation_residuals[
                                    target_index
                                ],
                                "collision_validation_profile": np.asarray(
                                    COLLISION_VALIDATION_PROFILE
                                ),
                                "minimum_collision_clearance_m": np.asarray(
                                    args.minimum_collision_clearance_m,
                                    dtype=np.float32,
                                ),
                                "approach_gripper_width_m": np.asarray(
                                    approach_widths[target_index], dtype=np.float32
                                ),
                                "collision_clearance_m": collision_clearances[
                                    target_index
                                ],
                                "nominal_collision_clearance_m": (
                                    nominal_collision_clearances[target_index]
                                ),
                            },
                        )
                        if not args.quiet_cached and not args.quiet:
                            print(
                                f"[ROTATION RESET] {target_index + 1:04d}/"
                                f"{len(target_ids)} {target_id} migrated "
                                f"safe={len(safe_indices)}/{cached_variant_count}",
                                flush=True,
                            )
                        continue
            except (KeyError, OSError, ValueError):
                pass
        base_poses = [pose_and_jacobian(q)[:2] for q in nominal_moveit[target_index]]
        nominal_clearances = np.asarray(
            [
                _minimum_gripper_clearance(
                    object_scene=target_collision_scenes[target_index],
                    collision_model=collision_model,
                    tcp_position_w=position,
                    tcp_rotation_w=rotation,
                    jaw_width_m=float(jaw_widths[target_index]),
                )
                for position, rotation in base_poses
            ],
            dtype=np.float64,
        )
        if np.any(
            nominal_clearances
            < float(args.minimum_collision_clearance_m) - 1.0e-9
        ):
            failing_waypoint = int(np.argmin(nominal_clearances))
            print(
                f"[DROP COLLISION] {target_id} nominal reset is not collision-safe "
                f"at approach "
                f"width {approach_widths[target_index] * 1000.0:.2f} mm: "
                f"waypoint={failing_waypoint}, clearance="
                f"{nominal_clearances[failing_waypoint] * 1000.0:.3f} mm, "
                f"required={args.minimum_collision_clearance_m * 1000.0:.3f} mm.",
                flush=True,
            )
            continue
        nominal_collision_clearances[target_index] = nominal_clearances.astype(
            np.float32
        )
        accepted_count = 0
        rejected_count = 0
        collision_rejected_count = 0
        for axis in candidate_axes:
            variant_joints = np.empty((len(progress), 7), dtype=np.float32)
            variant_position_residuals = np.empty(len(progress), dtype=np.float32)
            variant_rotation_residuals = np.empty(len(progress), dtype=np.float32)
            variant_collision_clearances = np.empty(len(progress), dtype=np.float32)
            previous_delta = np.zeros(7, dtype=np.float64)
            accepted = True
            for waypoint_index, (reference, (target_position, nominal_rotation)) in enumerate(
                zip(nominal_moveit[target_index], base_poses, strict=True)
            ):
                target_rotation = (
                    Rotation.from_rotvec(axis * angle_profile[waypoint_index]).as_matrix()
                    @ nominal_rotation
                )
                q = np.clip(
                    reference + previous_delta,
                    joint_lower + 1.0e-4,
                    joint_upper - 1.0e-4,
                )
                for _ in range(args.maximum_iterations):
                    position, rotation, jacobian = pose_and_jacobian(q)
                    position_error = target_position - position
                    rotation_error = Rotation.from_matrix(
                        target_rotation @ rotation.T
                    ).as_rotvec()
                    if (
                        np.linalg.norm(position_error) < 2.0e-6
                        and np.linalg.norm(rotation_error) < 2.0e-5
                    ):
                        break
                    error = np.concatenate((position_error, rotation_error))
                    damping = 2.0e-3
                    inverse = jacobian.T @ np.linalg.inv(
                        jacobian @ jacobian.T + damping**2 * np.eye(6)
                    )
                    delta = inverse @ error
                    delta += 0.02 * (np.eye(7) - inverse @ jacobian) @ (reference - q)
                    maximum_step = float(np.max(np.abs(delta)))
                    if maximum_step > 0.06:
                        delta *= 0.06 / maximum_step
                    q = np.clip(
                        q + delta,
                        joint_lower + 1.0e-4,
                        joint_upper - 1.0e-4,
                    )
                position, rotation, _ = pose_and_jacobian(q)
                position_residual = float(np.linalg.norm(target_position - position))
                rotation_residual = float(
                    np.linalg.norm(
                        Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
                    )
                )
                if position_residual > 5.0e-5 or rotation_residual > 5.0e-4:
                    accepted = False
                    break
                collision_clearance = _minimum_gripper_clearance(
                    object_scene=target_collision_scenes[target_index],
                    collision_model=collision_model,
                    tcp_position_w=position,
                    tcp_rotation_w=rotation,
                    jaw_width_m=float(jaw_widths[target_index]),
                )
                if (
                    collision_clearance
                    < float(args.minimum_collision_clearance_m) - 1.0e-9
                ):
                    accepted = False
                    collision_rejected_count += 1
                    break
                variant_joints[waypoint_index] = (
                    q * MOVEIT_TO_ISAAC_SIGNS
                ).astype(np.float32)
                variant_position_residuals[waypoint_index] = position_residual
                variant_rotation_residuals[waypoint_index] = rotation_residual
                variant_collision_clearances[waypoint_index] = collision_clearance
                previous_delta = q - reference
            if not accepted:
                rejected_count += 1
                continue
            selected_axes[target_index, accepted_count] = axis.astype(np.float32)
            reset_variants[target_index, accepted_count] = variant_joints
            position_residuals[target_index, accepted_count] = variant_position_residuals
            rotation_residuals[target_index, accepted_count] = variant_rotation_residuals
            collision_clearances[target_index, accepted_count] = (
                variant_collision_clearances
            )
            accepted_count += 1
            if accepted_count == args.variants:
                break
        distinct_count = accepted_count
        if accepted_count < minimum_distinct_variants:
            print(
                f"[DROP ROTATION] {target_id} has only "
                f"{accepted_count}/{minimum_distinct_variants} required collision-safe "
                f"distinct axes; ik_rejected={rejected_count} "
                f"collision_rejected={collision_rejected_count}.",
                flush=True,
            )
            continue
        while accepted_count < args.variants:
            source_index = accepted_count % distinct_count
            selected_axes[target_index, accepted_count] = selected_axes[
                target_index, source_index
            ]
            reset_variants[target_index, accepted_count] = reset_variants[
                target_index, source_index
            ]
            position_residuals[target_index, accepted_count] = position_residuals[
                target_index, source_index
            ]
            rotation_residuals[target_index, accepted_count] = rotation_residuals[
                target_index, source_index
            ]
            collision_clearances[target_index, accepted_count] = collision_clearances[
                target_index, source_index
            ]
            accepted_count += 1
        target_valid[target_index] = True
        if cache_path is not None:
            _atomic_savez(
                cache_path,
                {
                    "schema_version": np.asarray(
                        ROTATION_RESET_SCHEMA_VERSION, dtype=np.int64
                    ),
                    "axis_selection_method": np.asarray(AXIS_SELECTION_METHOD),
                    "target_id": np.asarray(str(target_id)),
                    "nominal_joint_trajectory": nominal_isaac[target_index].astype(
                        np.float32
                    ),
                    "rotation_angle_profile_rad": angle_profile.astype(np.float32),
                    "rotation_axes_w": selected_axes[target_index],
                    "rotation_joint_trajectories": reset_variants[target_index],
                    "ik_position_residual_m": position_residuals[target_index],
                    "ik_rotation_residual_rad": rotation_residuals[target_index],
                    "collision_validation_profile": np.asarray(
                        COLLISION_VALIDATION_PROFILE
                    ),
                    "minimum_collision_clearance_m": np.asarray(
                        args.minimum_collision_clearance_m, dtype=np.float32
                    ),
                    "approach_gripper_width_m": np.asarray(
                        approach_widths[target_index], dtype=np.float32
                    ),
                    "collision_clearance_m": collision_clearances[target_index],
                    "nominal_collision_clearance_m": (
                        nominal_collision_clearances[target_index]
                    ),
                },
            )
        if not args.quiet or (target_index + 1) % 100 == 0 or target_index + 1 == len(target_ids):
            padding = args.variants - distinct_count
            print(
                f"[ROTATION RESET] {target_index + 1:04d}/{len(target_ids)} {target_id} "
                f"distinct={distinct_count} padded={padding} rejected={rejected_count}",
                f"collision_rejected={collision_rejected_count}",
                flush=True,
            )

    valid_indices = np.flatnonzero(target_valid)
    if valid_indices.size < 1:
        raise RuntimeError(
            "No target retained a collision-safe nominal path and rotation pool."
        )
    dropped_count = len(target_ids) - len(valid_indices)
    target_ids = target_ids[valid_indices]
    selected_axes = selected_axes[valid_indices]
    reset_variants = reset_variants[valid_indices]
    position_residuals = position_residuals[valid_indices]
    rotation_residuals = rotation_residuals[valid_indices]
    collision_clearances = collision_clearances[valid_indices]
    nominal_collision_clearances = nominal_collision_clearances[valid_indices]
    approach_widths = approach_widths[valid_indices]

    payload = {
        "schema_version": np.asarray(
            ROTATION_RESET_SCHEMA_VERSION, dtype=np.int64
        ),
        "axis_selection_method": np.asarray(AXIS_SELECTION_METHOD),
        "source_paths_asset": np.asarray(str(paths_asset)),
        "target_ids": target_ids,
        "rotation_axes_w": selected_axes,
        "rotation_angle_profile_rad": angle_profile.astype(np.float32),
        "rotation_joint_trajectories": reset_variants,
        "ik_position_residual_m": position_residuals,
        "ik_rotation_residual_rad": rotation_residuals,
        "collision_validation_profile": np.asarray(COLLISION_VALIDATION_PROFILE),
        "minimum_collision_clearance_m": np.asarray(
            args.minimum_collision_clearance_m, dtype=np.float32
        ),
        "collision_validated": np.ones(
            collision_clearances.shape, dtype=np.bool_
        ),
        "collision_clearance_m": collision_clearances,
        "nominal_collision_validated": np.ones(
            nominal_collision_clearances.shape, dtype=np.bool_
        ),
        "nominal_collision_clearance_m": nominal_collision_clearances,
        "robot_profile": np.asarray(VISUAL_SERVO_GRIPPER_PROFILE),
        "approach_gripper_profile": np.asarray(PDZ_GRIPPER_APPROACH_PROFILE),
        "approach_gripper_widths_m": approach_widths.astype(np.float32),
        "far_rotation_rad": np.asarray(np.deg2rad(args.far_rotation_deg), dtype=np.float32),
        "near_rotation_rad": np.asarray(np.deg2rad(args.near_rotation_deg), dtype=np.float32),
        "rotation_exponent": np.asarray(args.rotation_exponent, dtype=np.float32),
    }
    output = args.output.expanduser().resolve()
    _atomic_savez(output, payload)
    print(
        f"[DONE] Wrote {reset_variants.shape} rotation-reset joints to {output}.\n"
        f"[DONE] Collision-safe targets: {len(target_ids)} retained, "
        f"{dropped_count} dropped; minimum clearance "
        f"{collision_clearances.min() * 1000.0:.3f} mm.\n"
        f"[DONE] Worst IK residual: {position_residuals.max() * 1000.0:.4f} mm / "
        f"{np.degrees(rotation_residuals.max()):.5f} deg.",
        flush=True,
    )


if __name__ == "__main__":
    main()
