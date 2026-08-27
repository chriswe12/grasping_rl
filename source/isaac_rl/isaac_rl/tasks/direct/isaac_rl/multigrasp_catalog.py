"""Validation and loading for goal-conditioned multi-grasp RL catalogs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from grasp_planning.rl.goal_catalog_profiles import MUJOCO_GOAL_RENDERER_PROFILE
from grasp_planning.start_poses import (
    PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    PDZ_GRIPPER_APPROACH_CLEARANCE_TOTAL_M,
    PDZ_GRIPPER_APPROACH_PROFILE,
    PDZ_GRIPPER_OPEN_WIDTH_M,
    VISUAL_SERVO_GRIPPER_PROFILE,
)

CATALOG_SCHEMA_VERSION = 4
SUPPORTED_CATALOG_SCHEMA_VERSIONS = (1, 2, 3, CATALOG_SCHEMA_VERSION)
ROTATION_RESET_SCHEMA_VERSION = 4
ROTATION_AXIS_SELECTION_METHOD = "fibonacci_farthest_point_v1"
ROTATION_COLLISION_VALIDATION_PROFILE = "pdz_gripper_object_ground_clearance_v1"
GOAL_RENDERER_PROFILE = MUJOCO_GOAL_RENDERER_PROFILE


def _require_shape(
    arrays: dict[str, np.ndarray],
    name: str,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"Multi-grasp catalog is missing required array '{name}'.")
    value = arrays[name]
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        expected_text = " x ".join("G" if item is None else str(item) for item in shape)
        raise ValueError(
            f"Catalog array '{name}' must have shape ({expected_text}), got {value.shape}."
        )
    return value


def load_multigrasp_catalog(  # noqa: C901 - strict schema validation is intentionally linear
    path: str | Path,
    *,
    expected_arm_joint_count: int = 7,
    require_complete: bool = True,
) -> dict[str, np.ndarray]:
    """Load a catalog and reject incomplete, malformed, or mislabeled targets."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}

    schema_version = int(np.asarray(arrays.get("schema_version", -1)).item())
    if schema_version not in SUPPORTED_CATALOG_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported multi-grasp catalog schema {schema_version}; "
            f"expected one of {SUPPORTED_CATALOG_SCHEMA_VERSIONS}."
        )

    target_ids = _require_shape(arrays, "target_ids", (None,))
    target_count = int(target_ids.shape[0])
    if target_count <= 0:
        raise ValueError("Multi-grasp catalog contains no targets.")
    target_id_strings = tuple(str(value) for value in target_ids.tolist())
    if len(set(target_id_strings)) != target_count:
        raise ValueError("Multi-grasp catalog target_ids must be unique.")

    for name in ("orientation_ids", "grasp_ids"):
        value = _require_shape(arrays, name, (target_count,))
        if any(not str(item) for item in value.tolist()):
            raise ValueError(f"Catalog array '{name}' contains an empty label.")
    orientation_names = _require_shape(arrays, "orientation_names", (None,))
    orientation_name_strings = tuple(str(value) for value in orientation_names.tolist())
    if not orientation_name_strings or len(set(orientation_name_strings)) != len(
        orientation_name_strings
    ):
        raise ValueError("orientation_names must be non-empty and unique.")
    orientation_indices = _require_shape(
        arrays, "orientation_indices", (target_count,)
    )
    if not np.issubdtype(orientation_indices.dtype, np.integer):
        raise ValueError("orientation_indices must use an integer dtype.")
    if np.any(orientation_indices < 0):
        raise ValueError("orientation_indices must be non-negative.")
    if np.any(orientation_indices >= len(orientation_name_strings)):
        raise ValueError("orientation_indices contains an index outside orientation_names.")
    expected_orientation_ids = np.asarray(orientation_name_strings)[orientation_indices]
    if not np.array_equal(arrays["orientation_ids"].astype(str), expected_orientation_ids):
        raise ValueError("orientation_ids do not match orientation_names[orientation_indices].")

    rgb = _require_shape(arrays, "goal_rgb", (target_count, None, None, 3))
    depth = _require_shape(
        arrays, "goal_depth", (target_count, rgb.shape[1], rgb.shape[2])
    )
    if rgb.dtype != np.uint8:
        raise ValueError(f"goal_rgb must be uint8, got {rgb.dtype}.")
    if not np.issubdtype(depth.dtype, np.floating) or not np.isfinite(depth).all():
        raise ValueError("goal_depth must be a finite floating-point array.")

    for name, width in (
        ("object_positions_w", 3),
        ("object_orientations_xyzw_w", 4),
        ("goal_grasp_positions_w", 3),
        ("goal_grasp_orientations_xyzw_w", 4),
        ("goal_tcp_positions_w", 3),
        ("goal_tcp_orientations_xyzw_w", 4),
    ):
        value = _require_shape(arrays, name, (target_count, width))
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Catalog array '{name}' must contain finite floating-point values.")

    for name in (
        "object_orientations_xyzw_w",
        "goal_grasp_orientations_xyzw_w",
        "goal_tcp_orientations_xyzw_w",
    ):
        norms = np.linalg.norm(arrays[name], axis=1)
        if not np.allclose(norms, 1.0, atol=2.0e-4):
            raise ValueError(f"Catalog quaternions in '{name}' must be normalized.")

    trajectories = _require_shape(
        arrays,
        "reset_joint_trajectories",
        (target_count, None, expected_arm_joint_count),
    )
    if trajectories.shape[1] < 2 or not np.isfinite(trajectories).all():
        raise ValueError("reset_joint_trajectories must have at least two finite waypoints.")
    progress = _require_shape(arrays, "reset_path_progress", (trajectories.shape[1],))
    if (
        not np.issubdtype(progress.dtype, np.floating)
        or not np.isfinite(progress).all()
        or np.any(np.diff(progress) <= 0.0)
        or float(progress[0]) < 0.0
        or float(progress[-1]) > 1.0
    ):
        raise ValueError(
            "reset_path_progress must be a finite, strictly increasing vector inside [0, 1]."
        )

    if schema_version >= 3:
        approach_profile = str(
            np.asarray(arrays.get("approach_gripper_profile", "")).item()
        )
        if approach_profile != PDZ_GRIPPER_APPROACH_PROFILE:
            raise ValueError(
                "Unsupported or missing approach-gripper profile "
                f"'{approach_profile or 'unlabeled'}'; expected "
                f"'{PDZ_GRIPPER_APPROACH_PROFILE}'. Rebuild paths and goal images."
            )
        clearance = float(
            np.asarray(arrays.get("approach_clearance_per_finger_m", np.nan)).item()
        )
        if abs(clearance - PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M) > 1.0e-7:
            raise ValueError(
                "Catalog approach clearance must be exactly 5 mm per finger."
            )
        jaw_widths = _require_shape(arrays, "grasp_jaw_widths_m", (target_count,))
        approach_widths = _require_shape(
            arrays, "approach_gripper_widths_m", (target_count,)
        )
        if not np.isfinite(jaw_widths).all() or np.any(jaw_widths < 0.0):
            raise ValueError("grasp_jaw_widths_m must contain finite non-negative values.")
        if not np.isfinite(approach_widths).all() or np.any(approach_widths < 0.0):
            raise ValueError(
                "approach_gripper_widths_m must contain finite non-negative values."
            )
        if np.any(approach_widths > PDZ_GRIPPER_OPEN_WIDTH_M + 1.0e-7):
            raise ValueError("A catalog approach aperture exceeds the physical gripper opening.")
        if not np.allclose(
            approach_widths - jaw_widths,
            PDZ_GRIPPER_APPROACH_CLEARANCE_TOTAL_M,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError(
                "Every approach aperture must equal its final jaw width plus 10 mm."
            )
    if schema_version >= 4:
        robot_profile = str(np.asarray(arrays.get("robot_profile", "")).item())
        if robot_profile != VISUAL_SERVO_GRIPPER_PROFILE:
            raise ValueError(
                f"Goal catalog robot profile '{robot_profile or 'unlabeled'}' does not "
                f"match '{VISUAL_SERVO_GRIPPER_PROFILE}'."
            )
        renderer_profile = str(
            np.asarray(arrays.get("goal_renderer_profile", "")).item()
        )
        if renderer_profile != GOAL_RENDERER_PROFILE:
            raise ValueError(
                f"Goal renderer profile '{renderer_profile or 'unlabeled'}' does not "
                f"match '{GOAL_RENDERER_PROFILE}'."
            )

    for name in ("moveit_plan_validated", "isaac_goal_rgbd_captured"):
        value = _require_shape(arrays, name, (target_count,))
        if value.dtype != np.bool_:
            raise ValueError(f"Catalog array '{name}' must use boolean dtype.")
        if require_complete and not bool(np.all(value)):
            invalid = [target_id_strings[index] for index in np.flatnonzero(~value)]
            raise ValueError(
                f"Catalog has {len(invalid)} incomplete '{name}' targets: {invalid[:8]}."
            )
    if schema_version >= 2:
        assembly_name = str(np.asarray(arrays.get("assembly_name", "")).item())
        if not assembly_name:
            raise ValueError("Schema-2 catalog assembly_name must be non-empty.")
        part_names = _require_shape(arrays, "part_names", (None,))
        part_name_strings = tuple(str(value) for value in part_names.tolist())
        if not part_name_strings or len(set(part_name_strings)) != len(part_name_strings):
            raise ValueError("part_names must be non-empty and unique.")
        part_usd_paths = _require_shape(
            arrays, "part_usd_paths", (len(part_name_strings),)
        )
        if any(not str(value) for value in part_usd_paths.tolist()):
            raise ValueError("part_usd_paths contains an empty path.")
        part_ids = _require_shape(arrays, "part_ids", (target_count,))
        part_indices = _require_shape(arrays, "part_indices", (target_count,))
        if not np.issubdtype(part_indices.dtype, np.integer):
            raise ValueError("part_indices must use an integer dtype.")
        if np.any(part_indices < 0) or np.any(part_indices >= len(part_name_strings)):
            raise ValueError("part_indices contains an index outside part_names.")
        if not np.array_equal(
            part_ids.astype(str), np.asarray(part_name_strings)[part_indices]
        ):
            raise ValueError("part_ids do not match part_names[part_indices].")
        _require_shape(arrays, "local_orientation_ids", (target_count,))

        split_names = _require_shape(arrays, "split_names", (None,))
        split_name_strings = tuple(str(value) for value in split_names.tolist())
        if set(split_name_strings) != {"train", "validation", "test"}:
            raise ValueError(
                "split_names must contain exactly train, validation, and test."
            )
        split_ids = _require_shape(arrays, "split_ids", (target_count,))
        split_indices = _require_shape(arrays, "split_indices", (target_count,))
        if not np.issubdtype(split_indices.dtype, np.integer):
            raise ValueError("split_indices must use an integer dtype.")
        if np.any(split_indices < 0) or np.any(split_indices >= len(split_name_strings)):
            raise ValueError("split_indices contains an index outside split_names.")
        if not np.array_equal(
            split_ids.astype(str), np.asarray(split_name_strings)[split_indices]
        ):
            raise ValueError("split_ids do not match split_names[split_indices].")
        grasp_split: dict[tuple[str, str], str] = {}
        for part_id, grasp_id, split_id in zip(
            part_ids.astype(str),
            arrays["grasp_ids"].astype(str),
            split_ids.astype(str),
            strict=True,
        ):
            key = (part_id, grasp_id)
            prior = grasp_split.setdefault(key, split_id)
            if prior != split_id:
                raise ValueError(
                    "Exact grasp leakage detected: one (part_id, grasp_id) group appears "
                    "in multiple splits."
                )
    return arrays


def select_catalog_split(
    arrays: dict[str, np.ndarray], split: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return one schema-2 split and its indices in the complete catalog."""

    if "split_ids" not in arrays:
        if split not in ("", "all", "train"):
            raise ValueError(
                f"Legacy catalog has no split metadata; cannot select '{split}'."
            )
        indices = np.arange(len(arrays["target_ids"]), dtype=np.int64)
        return {name: value.copy() for name, value in arrays.items()}, indices
    normalized = str(split).strip().lower()
    if normalized in ("", "all"):
        indices = np.arange(len(arrays["target_ids"]), dtype=np.int64)
        return {name: value.copy() for name, value in arrays.items()}, indices
    available = tuple(str(value) for value in arrays["split_names"].tolist())
    if normalized not in available:
        raise ValueError(f"Unknown catalog split '{split}'; available={available}.")
    indices = np.flatnonzero(arrays["split_ids"].astype(str) == normalized).astype(
        np.int64
    )
    if indices.size == 0:
        raise ValueError(f"Catalog split '{normalized}' contains no targets.")
    target_count = len(arrays["target_ids"])
    target_arrays = {
        "target_ids",
        "orientation_ids",
        "orientation_indices",
        "grasp_ids",
        "goal_rgb",
        "goal_depth",
        "object_positions_w",
        "object_orientations_xyzw_w",
        "goal_grasp_positions_w",
        "goal_grasp_orientations_xyzw_w",
        "goal_tcp_positions_w",
        "goal_tcp_orientations_xyzw_w",
        "grasp_jaw_widths_m",
        "approach_gripper_widths_m",
        "reset_joint_trajectories",
        "reset_path_max_position_error_m",
        "reset_path_max_rotation_error_rad",
        "moveit_plan_validated",
        "isaac_goal_rgbd_captured",
        "goal_tcp_capture_position_error_m",
        "goal_tcp_capture_rotation_error_rad",
        "goal_rgb_std",
        "goal_depth_std_m",
        "part_ids",
        "part_indices",
        "local_orientation_ids",
        "split_ids",
        "split_indices",
    }
    selected: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if name in target_arrays:
            if value.shape[0] != target_count:
                raise ValueError(
                    f"Target array '{name}' has leading size {value.shape[0]}, "
                    f"expected {target_count}."
                )
            selected[name] = value[indices].copy()
        else:
            selected[name] = value.copy()
    return selected, indices


def load_multigrasp_rotation_resets(
    path: str | Path,
    *,
    expected_target_ids: tuple[str, ...],
    expected_waypoint_count: int,
    expected_arm_joint_count: int = 7,
    expected_approach_gripper_widths_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Load validated, position-preserving rotational reset trajectories."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    schema_version = int(np.asarray(arrays.get("schema_version", -1)).item())
    if schema_version != ROTATION_RESET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported rotation reset schema {schema_version}; "
            f"expected {ROTATION_RESET_SCHEMA_VERSION}."
        )
    axis_selection_method = str(
        np.asarray(arrays.get("axis_selection_method", "")).item()
    )
    if axis_selection_method != ROTATION_AXIS_SELECTION_METHOD:
        raise ValueError(
            "Unsupported or missing rotation-axis selection method "
            f"'{axis_selection_method or 'unlabeled'}'; expected "
            f"'{ROTATION_AXIS_SELECTION_METHOD}'. Rebuild the rotation-reset asset."
        )
    target_ids = _require_shape(arrays, "target_ids", (len(expected_target_ids),))
    if tuple(str(value) for value in target_ids.tolist()) != expected_target_ids:
        raise ValueError("Rotation reset target_ids do not exactly match the goal catalog.")
    axes = _require_shape(
        arrays, "rotation_axes_w", (len(expected_target_ids), None, 3)
    )
    variant_count = axes.shape[1]
    if variant_count < 2 or not np.allclose(
        np.linalg.norm(axes, axis=2), 1.0, atol=2.0e-4
    ):
        raise ValueError("rotation_axes_w must contain at least two normalized axes.")
    trajectories = _require_shape(
        arrays,
        "rotation_joint_trajectories",
        (
            len(expected_target_ids),
            variant_count,
            expected_waypoint_count,
            expected_arm_joint_count,
        ),
    )
    if not np.isfinite(trajectories).all():
        raise ValueError("rotation_joint_trajectories must contain finite values.")
    profile = _require_shape(
        arrays, "rotation_angle_profile_rad", (expected_waypoint_count,)
    )
    if not np.isfinite(profile).all() or np.any(profile < 0.0):
        raise ValueError("rotation_angle_profile_rad must contain finite non-negative values.")
    if np.any(np.diff(profile) > 1.0e-6):
        raise ValueError("rotation_angle_profile_rad must decrease toward the grasp.")
    for name in ("ik_position_residual_m", "ik_rotation_residual_rad"):
        residual = _require_shape(
            arrays,
            name,
            (len(expected_target_ids), variant_count, expected_waypoint_count),
        )
        if not np.isfinite(residual).all() or np.any(residual < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values.")
    if float(np.max(arrays["ik_position_residual_m"])) > 5.0e-5:
        raise ValueError("Rotation reset position IK residual exceeds 0.05 mm.")
    if float(np.max(arrays["ik_rotation_residual_rad"])) > 5.0e-4:
        raise ValueError("Rotation reset orientation IK residual exceeds 0.0005 rad.")
    collision_profile = str(
        np.asarray(arrays.get("collision_validation_profile", "")).item()
    )
    if collision_profile != ROTATION_COLLISION_VALIDATION_PROFILE:
        raise ValueError(
            "Unsupported or missing reset-collision validation profile "
            f"'{collision_profile or 'unlabeled'}'; expected "
            f"'{ROTATION_COLLISION_VALIDATION_PROFILE}'. Rebuild rotation resets."
        )
    robot_profile = str(np.asarray(arrays.get("robot_profile", "")).item())
    if robot_profile != VISUAL_SERVO_GRIPPER_PROFILE:
        raise ValueError(
            f"Rotation-reset robot profile '{robot_profile or 'unlabeled'}' does not "
            f"match '{VISUAL_SERVO_GRIPPER_PROFILE}'."
        )
    approach_profile = str(
        np.asarray(arrays.get("approach_gripper_profile", "")).item()
    )
    if approach_profile != PDZ_GRIPPER_APPROACH_PROFILE:
        raise ValueError(
            "Rotation-reset approach-gripper profile "
            f"'{approach_profile or 'unlabeled'}' does not match "
            f"'{PDZ_GRIPPER_APPROACH_PROFILE}'."
        )
    collision_clearance = float(
        np.asarray(arrays.get("minimum_collision_clearance_m", np.nan)).item()
    )
    if not np.isfinite(collision_clearance) or collision_clearance < 0.0:
        raise ValueError("minimum_collision_clearance_m must be finite and non-negative.")
    collision_validated = _require_shape(
        arrays,
        "collision_validated",
        (len(expected_target_ids), variant_count, expected_waypoint_count),
    )
    if collision_validated.dtype != np.bool_ or not bool(np.all(collision_validated)):
        raise ValueError("Every authored rotation-reset state must be collision validated.")
    nominal_collision_validated = _require_shape(
        arrays,
        "nominal_collision_validated",
        (len(expected_target_ids), expected_waypoint_count),
    )
    if nominal_collision_validated.dtype != np.bool_ or not bool(
        np.all(nominal_collision_validated)
    ):
        raise ValueError("Every authored nominal reset state must be collision validated.")
    for name, expected_shape in (
        (
            "collision_clearance_m",
            (len(expected_target_ids), variant_count, expected_waypoint_count),
        ),
        (
            "nominal_collision_clearance_m",
            (len(expected_target_ids), expected_waypoint_count),
        ),
    ):
        clearances = _require_shape(arrays, name, expected_shape)
        if not np.isfinite(clearances).all() or np.any(
            clearances < collision_clearance - 1.0e-7
        ):
            raise ValueError(
                f"{name} contains a state below minimum_collision_clearance_m."
            )
    approach_widths = _require_shape(
        arrays, "approach_gripper_widths_m", (len(expected_target_ids),)
    )
    if expected_approach_gripper_widths_m is not None and not np.allclose(
        approach_widths,
        np.asarray(expected_approach_gripper_widths_m),
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise ValueError(
            "Rotation-reset approach apertures do not match the goal catalog."
        )
    return arrays


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "SUPPORTED_CATALOG_SCHEMA_VERSIONS",
    "ROTATION_RESET_SCHEMA_VERSION",
    "ROTATION_AXIS_SELECTION_METHOD",
    "ROTATION_COLLISION_VALIDATION_PROFILE",
    "load_multigrasp_catalog",
    "load_multigrasp_rotation_resets",
    "select_catalog_split",
]
