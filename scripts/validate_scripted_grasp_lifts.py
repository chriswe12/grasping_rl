"""Validate whether catalog grasps can close and retain dynamically lifted parts.

This is deliberately not a policy benchmark.  Every environment starts at an
exact, collision-validated goal waypoint.  The script settles the dynamic
part, closes the PDZ fingers, commands a vertical TCP lift, and reports setup,
arm-motion, and grasp-retention failures separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0",
    )
    parser.add_argument("--dataset-index", type=Path, default=None)
    parser.add_argument("--dataset-shard", type=int, default=0)
    parser.add_argument("--catalog-split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument(
        "--target-id",
        action="append",
        default=[],
        help="Validate this exact target ID; repeat for multiple targets.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=0,
        help="Balanced cross-part target limit; zero validates the complete selected split.",
    )
    parser.add_argument(
        "--balanced-round",
        type=int,
        default=0,
        help="Skip this many targets per part before selecting the balanced subset.",
    )
    parser.add_argument(
        "--target-offset",
        type=int,
        default=None,
        help="Select a contiguous split batch starting here; requires --max-targets.",
    )
    parser.add_argument("--settle-duration-s", type=float, default=0.40)
    parser.add_argument(
        "--extra-approach-clearance-m",
        type=float,
        default=0.0,
        help=(
            "Additional total jaw opening during settling, beyond the catalog approach width. "
            "The value is clipped at the physical open width and is removed during closure."
        ),
    )
    parser.add_argument("--close-duration-s", type=float, default=1.00)
    parser.add_argument("--postclose-hold-s", type=float, default=0.50)
    parser.add_argument(
        "--disable-part-gravity-until-close",
        action="store_true",
        help=(
            "Hold gravity off through settling and closure, then restore it before lifting. "
            "Use only to isolate grasp retention when a nominal resting pose tips before closure."
        ),
    )
    parser.add_argument(
        "--fixture-part-pose-until-close",
        action="store_true",
        help=(
            "Restore the exact catalog object pose after every settle/close step, then release the "
            "dynamic part before the measured lift. This isolates closed-grasp retention from an "
            "unstable pre-grasp placement and must be reported as fixture-assisted evidence."
        ),
    )
    parser.add_argument(
        "--gravity-release-hold-s",
        type=float,
        default=0.15,
        help="Hold the closed grasp after restoring gravity and before starting the lift.",
    )
    parser.add_argument("--lift-height-m", type=float, default=0.060)
    parser.add_argument("--lift-speed-m-s", type=float, default=0.050)
    parser.add_argument("--postlift-hold-s", type=float, default=0.20)
    parser.add_argument(
        "--squeeze-margin-m",
        type=float,
        default=0.015,
        help="Additional total jaw closure beyond the catalog contact width.",
    )
    parser.add_argument("--minimum-final-lift-m", type=float, default=0.040)
    parser.add_argument("--maximum-settle-translation-m", type=float, default=0.003)
    parser.add_argument("--maximum-settle-rotation-deg", type=float, default=3.0)
    parser.add_argument("--maximum-peak-drop-m", type=float, default=0.015)
    parser.add_argument("--maximum-relative-drift-m", type=float, default=0.030)
    parser.add_argument("--static-friction", type=float, default=10.0)
    parser.add_argument("--dynamic-friction", type=float, default=10.0)
    parser.add_argument("--hand-effort-limit-n", type=float, default=200.0)
    parser.add_argument("--hand-stiffness", type=float, default=15000.0)
    parser.add_argument("--hand-damping", type=float, default=245.0)
    parser.add_argument(
        "--part-mass-kg",
        type=float,
        default=None,
        help="Override every part mass for a permissive geometry/retention screen.",
    )
    parser.add_argument("--dls-damping", type=float, default=0.05)
    parser.add_argument("--maximum-joint-speed-rad-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Record an annotated external/wrist-camera video per target (maximum four targets).",
    )
    parser.add_argument("--video-fps", type=float, default=15.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
for name in (
    "settle_duration_s",
    "close_duration_s",
    "postclose_hold_s",
    "lift_height_m",
    "lift_speed_m_s",
    "postlift_hold_s",
    "minimum_final_lift_m",
    "maximum_settle_translation_m",
    "maximum_settle_rotation_deg",
    "maximum_peak_drop_m",
    "maximum_relative_drift_m",
    "static_friction",
    "dynamic_friction",
    "hand_effort_limit_n",
    "hand_stiffness",
    "hand_damping",
    "dls_damping",
    "maximum_joint_speed_rad_s",
    "video_fps",
):
    if float(getattr(args_cli, name)) <= 0.0:
        raise ValueError(f"--{name.replace('_', '-')} must be positive.")
if float(args_cli.squeeze_margin_m) < 0.0:
    raise ValueError("--squeeze-margin-m must be non-negative.")
if float(args_cli.extra_approach_clearance_m) < 0.0:
    raise ValueError("--extra-approach-clearance-m must be non-negative.")
if float(args_cli.gravity_release_hold_s) < 0.0:
    raise ValueError("--gravity-release-hold-s must be non-negative.")
if int(args_cli.max_targets) < 0:
    raise ValueError("--max-targets must be non-negative.")
if int(args_cli.balanced_round) < 0:
    raise ValueError("--balanced-round must be non-negative.")
if int(args_cli.balanced_round) and not int(args_cli.max_targets):
    raise ValueError("--balanced-round requires a positive --max-targets.")
if args_cli.target_offset is not None and int(args_cli.target_offset) < 0:
    raise ValueError("--target-offset must be non-negative.")
if args_cli.target_offset is not None and not int(args_cli.max_targets):
    raise ValueError("--target-offset requires a positive --max-targets.")
if args_cli.target_id and (
    int(args_cli.max_targets) or int(args_cli.balanced_round) or args_cli.target_offset is not None
):
    raise ValueError("--target-id cannot be combined with bounded subset options.")
if int(args_cli.balanced_round) and args_cli.target_offset is not None:
    raise ValueError("--balanced-round and --target-offset are mutually exclusive.")
if args_cli.part_mass_kg is not None and float(args_cli.part_mass_kg) <= 0.0:
    raise ValueError("--part-mass-kg must be positive when provided.")
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: F401,E402
import torch  # noqa: E402
from grasp_planning.rl.fabrica_dataset import (  # noqa: E402
    DEFAULT_DATASET_INDEX,
    FABRICA_TASK_ID,
    configure_fabrica_env_cfg,
    subset_target_arrays,
)
from grasp_planning.rl.scripted_lift_validation import (  # noqa: E402
    ScriptedLiftThresholds,
    balanced_subset_indices,
    classify_scripted_lift,
    contiguous_subset_indices,
    resolve_explicit_target_indices,
    summarize_scripted_lifts,
)
from grasp_planning.start_poses import (  # noqa: E402
    PDZ_GRIPPER_CLOSED_WIDTH_M,
    PDZ_GRIPPER_OPEN_WIDTH_M,
)
from grasp_planning.video import OpenCvVideoWriter  # noqa: E402
from isaac_rl.tasks.direct.isaac_rl.multigrasp_catalog import (  # noqa: E402
    load_multigrasp_catalog,
    select_catalog_split,
)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import omni.usd  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.utils import bind_physics_material  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_conjugate  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


@dataclass(frozen=True)
class _LiftVideoFrame:
    side_rgb: np.ndarray
    wrist_rgb: np.ndarray
    phase: str
    time_s: float
    object_z_m: float
    jaw_width_m: float


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_lift_video(
    path: Path,
    *,
    frames: list[_LiftVideoFrame],
    target_id: str,
    status: str,
    settled_object_z_m: float,
    fps: float,
) -> None:
    if not frames:
        raise RuntimeError(f"No rendered frames were captured for {target_id!r}.")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    with OpenCvVideoWriter(path, fps=fps, width=960, height=540) as writer:
        for frame in frames:
            canvas = Image.fromarray(frame.side_rgb).resize((960, 540), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 960, 72), fill=(8, 10, 14))
            draw.text((14, 10), target_id, font=font, fill=(245, 247, 250))
            draw.text(
                (14, 34),
                (
                    f"phase={frame.phase}  t={frame.time_s:.2f}s  "
                    f"object_lift={(frame.object_z_m - settled_object_z_m) * 1000.0:+.1f}mm  "
                    f"jaw={frame.jaw_width_m * 1000.0:.1f}mm  result={status}"
                ),
                font=font,
                fill=(126, 231, 255) if status == "success" else (255, 183, 114),
            )
            wrist = Image.fromarray(frame.wrist_rgb).resize((320, 180), Image.Resampling.LANCZOS)
            canvas.paste(wrist, (628, 348))
            draw.rectangle((626, 346, 950, 530), outline=(245, 247, 250), width=2)
            draw.text((638, 356), "wrist camera", font=font, fill=(245, 247, 250))
            writer.append_rgb(np.asarray(canvas))


def _active_part_pose(task_env) -> tuple[torch.Tensor, torch.Tensor]:
    target_indices = task_env.target_index
    selected_parts = task_env.target_part_indices[target_indices]
    positions = torch.empty((task_env.num_envs, 3), dtype=torch.float32, device=task_env.device)
    quaternions = torch.empty((task_env.num_envs, 4), dtype=torch.float32, device=task_env.device)
    for part_index, part in enumerate(task_env.parts):
        mask = selected_parts == part_index
        if mask.any():
            positions[mask] = part.data.root_pos_w[mask]
            quaternions[mask] = part.data.root_quat_w[mask]
    return positions, quaternions


def _quaternion_distance_rad(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _actual_pdz_widths(task_env) -> torch.Tensor:
    names = task_env.context.hand_joint_names
    positions = task_env.robot.data.joint_pos[:, task_env.context.hand_joint_ids]
    indices = [
        index
        for index, name in enumerate(names)
        if name in {"pdz_gripper_left_finger_joint", "pdz_gripper_right_finger_joint"}
    ]
    if len(indices) != 2:
        raise RuntimeError(f"Expected two PDZ finger joints, got {list(names)}")
    return PDZ_GRIPPER_CLOSED_WIDTH_M + positions[:, indices].clamp_min(0.0).sum(dim=-1)


def _pdz_finger_contact_forces(task_env) -> tuple[torch.Tensor, torch.Tensor]:
    body_names = tuple(task_env.gripper_contact_sensor.body_names)
    try:
        left_index = body_names.index("pdz_gripper_left_finger_link")
        right_index = body_names.index("pdz_gripper_right_finger_link")
    except ValueError as exc:
        raise RuntimeError(f"Contact sensor does not expose both PDZ fingers: {body_names}") from exc
    force_norm = torch.linalg.norm(task_env.gripper_contact_sensor.data.net_forces_w, dim=-1)
    return force_norm[:, left_index], force_norm[:, right_index]


def _step_physics(task_env, arm_target: torch.Tensor, *, render: bool = False) -> None:
    task_env.robot.set_joint_position_target(arm_target, joint_ids=task_env.arm_ids)
    task_env.robot.set_joint_velocity_target(torch.zeros_like(arm_target), joint_ids=task_env.arm_ids)
    task_env.context.command_fixed_gripper()
    task_env.scene.write_data_to_sim()
    task_env.sim.step(render=render)
    task_env.scene.update(task_env.sim.get_physics_dt())


def _bind_high_friction(task_env) -> int:
    material_path = "/World/Looks/scripted_lift_high_friction"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(args_cli.static_friction),
        dynamic_friction=float(args_cli.dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)
    stage = omni.usd.get_context().get_stage()
    bound = 0
    for env_index in range(task_env.num_envs):
        roots = [
            (
                f"/World/envs/env_{env_index}/Part"
                if len(task_env.parts) == 1
                else f"/World/envs/env_{env_index}/Part_{part_index}"
            )
            for part_index in range(len(task_env.parts))
        ]
        for root in roots:
            if not stage.GetPrimAtPath(root).IsValid():
                raise RuntimeError(f"Cannot bind lift material; missing prim {root}")
            bind_physics_material(root, material_path, stage=stage, stronger_than_descendants=True)
            bound += 1
    return bound


def _set_part_gravity_disabled(task_env, *, disabled: bool) -> int:
    """Toggle gravity for every part instance without changing collision response."""

    changed = 0
    for part in task_env.parts:
        count = int(part.root_physx_view.count)
        flags = torch.full(
            (count, 1),
            int(disabled),
            dtype=torch.uint8,
            device="cpu",
        )
        indices = torch.arange(count, dtype=torch.int32, device="cpu")
        part.root_physx_view.set_disable_gravities(flags, indices)
        if not disabled:
            part.root_physx_view.wake_up(indices)
        changed += count
    return changed


def _restore_active_part_pose(
    task_env,
    *,
    selected_parts: torch.Tensor,
    positions_w: torch.Tensor,
    quaternions_w: torch.Tensor,
) -> None:
    """Hard-reset selected dynamic parts to a fixture pose with zero velocity."""

    poses = torch.cat((positions_w, quaternions_w), dim=-1)
    zero_velocity = torch.zeros((task_env.num_envs, 6), dtype=torch.float32, device=task_env.device)
    for part_index, part in enumerate(task_env.parts):
        env_ids = torch.nonzero(selected_parts == part_index, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            continue
        part.write_root_pose_to_sim(poses[env_ids], env_ids=env_ids)
        if hasattr(part, "write_root_velocity_to_sim"):
            part.write_root_velocity_to_sim(zero_velocity[env_ids], env_ids=env_ids)


def _configure_environment():
    if args_cli.task != FABRICA_TASK_ID:
        raise ValueError(f"This validator currently requires task {FABRICA_TASK_ID}.")
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    shard = configure_fabrica_env_cfg(
        cfg,
        explicit_shard=int(args_cli.dataset_shard),
        index_path=args_cli.dataset_index or DEFAULT_DATASET_INDEX,
    )
    cfg.catalog_split = str(args_cli.catalog_split)
    temporary_catalog_path: Path | None = None
    target_count = int(shard.split_counts[cfg.catalog_split])
    should_subset = (
        bool(args_cli.target_id)
        or args_cli.target_offset is not None
        or (
            int(args_cli.max_targets) > 0
            and (int(args_cli.max_targets) < target_count or int(args_cli.balanced_round) > 0)
        )
    )
    if should_subset:
        complete_catalog = load_multigrasp_catalog(cfg.goal_catalog_data_path)
        split_catalog, _ = select_catalog_split(complete_catalog, cfg.catalog_split)
        if args_cli.target_id:
            selected_indices = resolve_explicit_target_indices(
                split_catalog["target_ids"],
                args_cli.target_id,
            )
        elif args_cli.target_offset is not None:
            selected_indices = contiguous_subset_indices(
                target_count,
                offset=int(args_cli.target_offset),
                limit=int(args_cli.max_targets),
            )
        else:
            selected_indices = balanced_subset_indices(
                split_catalog["part_ids"],
                int(args_cli.max_targets),
                start_round=int(args_cli.balanced_round),
            )
        if not len(selected_indices):
            raise ValueError(
                f"Balanced round {args_cli.balanced_round} has no targets in split {args_cli.catalog_split!r}."
            )
        source_part_names = tuple(str(value) for value in split_catalog["part_names"].tolist())
        source_part_paths = tuple(str(value) for value in split_catalog["part_usd_paths"].tolist())
        part_path_by_name = dict(zip(source_part_names, source_part_paths, strict=True))
        radius_by_name = dict(zip(cfg.part_names, cfg.part_xy_rotation_radii_m, strict=True))
        selected_part_names = tuple(sorted(set(split_catalog["part_ids"][selected_indices].astype(str).tolist())))
        selected_catalog = subset_target_arrays(
            split_catalog,
            selected_indices,
            part_names=selected_part_names,
            part_usd_paths=tuple(part_path_by_name[name] for name in selected_part_names),
        )
        with tempfile.NamedTemporaryFile(
            prefix="scripted_lift_catalog_",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary_catalog_path = Path(stream.name)
        np.savez_compressed(temporary_catalog_path, **selected_catalog)
        cfg.goal_catalog_data_path = str(temporary_catalog_path)
        cfg.catalog_split = "all"
        cfg.part_names = selected_part_names
        cfg.part_usd_paths = tuple(part_path_by_name[name] for name in selected_part_names)
        cfg.part_xy_rotation_radii_m = tuple(radius_by_name[name] for name in selected_part_names)
        target_count = len(selected_indices)
    if args_cli.video_dir is not None and target_count > 4:
        raise ValueError("Video recording is limited to four targets per run; select targets with --target-id.")
    cfg.scene.num_envs = target_count
    cfg.seed = 7
    cfg.fixed_target_index = -1
    cfg.fixed_target_id = ""
    cfg.random_target_sampling = False
    cfg.sequential_target_sampling = True
    cfg.training_reset_mixture_enabled = False
    cfg.completion_positive_reset_fraction = 1.0
    cfg.reset_ready_exact_fraction = 1.0
    cfg.reset_rotation_randomization_enabled = False
    cfg.reset_position_randomization_enabled = False
    cfg.reset_object_yaw_randomization_enabled = False
    cfg.live_observation_randomization_enabled = False
    cfg.scene_appearance_randomization_enabled = False
    cfg.scene_tslot_surface_enabled = False
    cfg.scene_surface_markings_enabled = False
    cfg.scene_clutter_enabled = False
    cfg.scene_busy_background_enabled = False
    cfg.debug_camera_enabled = args_cli.video_dir is not None
    if cfg.debug_camera_enabled:
        # Close positive-Y inspection view: the target lies between this
        # camera and the robot, so the arm does not hide pad/object contact.
        cfg.debug_camera.offset.pos = (0.50, 0.55, 0.22)
        cfg.debug_camera.offset.rot = (
            0.06133532994118823,
            0.04465388752315237,
            0.5868750686415463,
            0.8061151663608963,
        )
    cfg.part_cfg.spawn.rigid_props.kinematic_enabled = False
    cfg.part_cfg.spawn.rigid_props.disable_gravity = bool(args_cli.disable_part_gravity_until_close)
    cfg.part_cfg.spawn.rigid_props.solver_position_iteration_count = 64
    cfg.part_cfg.spawn.rigid_props.solver_velocity_iteration_count = 4
    if args_cli.part_mass_kg is not None:
        cfg.part_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=float(args_cli.part_mass_kg))
    # Robot collision meshes are instanced and reject runtime subtree material
    # bindings. Using the permissive material as the simulation default and on
    # every part still dominates contact through friction_combine_mode="max".
    cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(args_cli.static_friction),
        dynamic_friction=float(args_cli.dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    cfg.sim.physx.gpu_found_lost_pairs_capacity = max(
        int(cfg.sim.physx.gpu_found_lost_pairs_capacity),
        2**23,
    )
    for actuator_name in ("hand_driver", "hand_follower"):
        actuator = cfg.robot_cfg.actuators.get(actuator_name)
        if actuator is None:
            raise RuntimeError(f"PDZ robot is missing actuator {actuator_name!r}.")
        actuator.effort_limit_sim = float(args_cli.hand_effort_limit_n)
        actuator.stiffness = float(args_cli.hand_stiffness)
        actuator.damping = float(args_cli.hand_damping)
    return cfg, shard, temporary_catalog_path


def _write_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attempts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary, "attempts": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Scripted grasp-lift validation",
        "",
        "This is a permissive simulator feasibility screen, not a learned-policy benchmark.",
        (
            "Every attempt starts at the exact catalog goal pose with a dynamic part, "
            "high friction, and high finger effort."
        ),
    ]
    if metadata["disable_part_gravity_until_close"]:
        lines.extend(
            [
                (
                    "Gravity is disabled only through closure to support a marginal nominal resting pose; "
                    "gravity and dynamic contact are restored before the measured lift."
                ),
            ]
        )
    if metadata["fixture_part_pose_until_close"]:
        lines.append(
            "The part pose is fixture-held through closure and released before the measured lift; "
            "success is fixture-assisted retention evidence, not free-closing success."
        )
    lines.extend(
        [
            "",
            f"- Attempts: {summary['attempts']}",
            f"- Simulator-invalid attempts: {summary['simulator_invalid_attempts']}",
            f"- Valid physical attempts: {summary['valid_attempts']}",
            f"- Retained pickups: {summary['successes']}",
            f"- Retained-pickup rate: {100.0 * float(summary['success_rate']):.1f}%",
            "",
            "## Outcomes",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {status} | {count} |" for status, count in summary["status_counts"].items())
    lines.extend(
        [
            "",
            "## Per part",
            "",
            "| Part | Valid | Success | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for part_id, values in summary["per_part"].items():
        lines.append(
            f"| {part_id} | {values['valid_attempts']} | {values['successes']} "
            f"| {100.0 * float(values['success_rate']):.1f}% |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:  # noqa: C901 - one bounded simulator experiment
    cfg, shard, temporary_catalog_path = _configure_environment()
    thresholds = ScriptedLiftThresholds(
        minimum_final_lift_m=float(args_cli.minimum_final_lift_m),
        maximum_settle_translation_m=float(args_cli.maximum_settle_translation_m),
        maximum_settle_rotation_rad=math.radians(float(args_cli.maximum_settle_rotation_deg)),
        maximum_peak_drop_m=float(args_cli.maximum_peak_drop_m),
        maximum_relative_drift_m=float(args_cli.maximum_relative_drift_m),
    )
    env = None
    try:
        env = gym.make(args_cli.task, cfg=cfg)
        task_env = env.unwrapped
        bound_count = _bind_high_friction(task_env)
        env.reset()
        if args_cli.disable_part_gravity_until_close:
            _set_part_gravity_disabled(task_env, disabled=True)
        target_indices = task_env.target_index.clone()
        if sorted(target_indices.detach().cpu().tolist()) != list(range(task_env.target_count)):
            raise RuntimeError("Exact validation reset did not assign every split target exactly once.")
        with np.load(Path(cfg.goal_catalog_data_path), allow_pickle=False) as catalog:
            all_jaw_widths = torch.as_tensor(catalog["grasp_jaw_widths_m"], dtype=torch.float32, device=task_env.device)
        source_indices = task_env.catalog_target_indices[target_indices]
        jaw_widths = all_jaw_widths[source_indices]
        catalog_approach_widths = task_env.approach_gripper_widths_catalog[target_indices]
        approach_widths = torch.clamp(
            catalog_approach_widths + float(args_cli.extra_approach_clearance_m),
            max=PDZ_GRIPPER_OPEN_WIDTH_M,
        )
        # The environment reset hard-writes the catalog approach aperture.  Apply
        # the diagnostic override before the first physics step so settle-phase
        # contacts are measured at the requested clearance rather than while the
        # fingers slowly drive open.
        task_env.context.set_fixed_gripper_widths(approach_widths)
        task_env.context.write_fixed_gripper_state()
        close_widths = torch.clamp(
            jaw_widths - float(args_cli.squeeze_margin_m),
            min=PDZ_GRIPPER_CLOSED_WIDTH_M,
        )
        selected_parts = task_env.target_part_indices[target_indices]
        nominal_position = task_env.object_positions_catalog[target_indices] + task_env.scene.env_origins
        nominal_quaternion = task_env.object_quaternions_catalog[target_indices]
        arm_target = task_env.robot.data.joint_pos[:, task_env.arm_ids].clone()
        physics_dt = float(task_env.sim.get_physics_dt())
        video_frames: list[list[_LiftVideoFrame]] = [[] for _ in range(task_env.num_envs)]
        video_stride = max(1, round(1.0 / (float(args_cli.video_fps) * physics_dt)))
        physics_step = 0

        def advance(arm_command: torch.Tensor, *, phase: str) -> None:
            nonlocal physics_step
            capture = args_cli.video_dir is not None and physics_step % video_stride == 0
            _step_physics(task_env, arm_command, render=capture)
            if capture:
                if task_env.debug_camera is None:
                    raise RuntimeError("Video recording requested without a debug camera.")
                object_position, _ = _active_part_pose(task_env)
                widths = _actual_pdz_widths(task_env)
                side_rgb = task_env.debug_camera.data.output["rgb"]
                wrist_rgb = task_env.wrist_camera.data.output["rgb"]
                for env_index in range(task_env.num_envs):
                    video_frames[env_index].append(
                        _LiftVideoFrame(
                            side_rgb=_rgb_uint8(side_rgb[env_index]),
                            wrist_rgb=_rgb_uint8(wrist_rgb[env_index]),
                            phase=phase,
                            time_s=physics_step * physics_dt,
                            object_z_m=float(object_position[env_index, 2].item()),
                            jaw_width_m=float(widths[env_index].item()),
                        )
                    )
            physics_step += 1

        print(
            f"[LIFT] shard={shard.shard_index}/{shard.shard_count} split={args_cli.catalog_split} "
            f"round={int(args_cli.balanced_round)} targets={task_env.target_count} "
            f"parts={len(task_env.parts)} material_bindings={bound_count}",
            flush=True,
        )
        settle_steps = max(1, round(float(args_cli.settle_duration_s) / physics_dt))
        settle_peak_contact_force = torch.zeros(task_env.num_envs, device=task_env.device)
        settle_peak_left_finger_force = torch.zeros_like(settle_peak_contact_force)
        settle_peak_right_finger_force = torch.zeros_like(settle_peak_contact_force)
        for _ in range(settle_steps):
            advance(arm_target, phase="settle")
            settle_peak_contact_force = torch.maximum(
                settle_peak_contact_force,
                task_env._gripper_contact_force(),
            )
            left_force, right_force = _pdz_finger_contact_forces(task_env)
            settle_peak_left_finger_force = torch.maximum(settle_peak_left_finger_force, left_force)
            settle_peak_right_finger_force = torch.maximum(settle_peak_right_finger_force, right_force)
            if args_cli.fixture_part_pose_until_close:
                _restore_active_part_pose(
                    task_env,
                    selected_parts=selected_parts,
                    positions_w=nominal_position,
                    quaternions_w=nominal_quaternion,
                )
        settled_position, settled_quaternion = _active_part_pose(task_env)
        settle_translation = torch.linalg.norm(settled_position - nominal_position, dim=-1)
        settle_rotation = _quaternion_distance_rad(settled_quaternion, nominal_quaternion)

        close_steps = max(1, round(float(args_cli.close_duration_s) / physics_dt))
        peak_contact_force = torch.zeros(task_env.num_envs, device=task_env.device)
        peak_left_finger_force = torch.zeros_like(peak_contact_force)
        peak_right_finger_force = torch.zeros_like(peak_contact_force)
        for step in range(close_steps):
            fraction = float(step + 1) / float(close_steps)
            widths = approach_widths + fraction * (close_widths - approach_widths)
            task_env.context.set_fixed_gripper_widths(widths)
            advance(arm_target, phase="close")
            peak_contact_force = torch.maximum(peak_contact_force, task_env._gripper_contact_force())
            left_force, right_force = _pdz_finger_contact_forces(task_env)
            peak_left_finger_force = torch.maximum(peak_left_finger_force, left_force)
            peak_right_finger_force = torch.maximum(peak_right_finger_force, right_force)
            if args_cli.fixture_part_pose_until_close:
                _restore_active_part_pose(
                    task_env,
                    selected_parts=selected_parts,
                    positions_w=nominal_position,
                    quaternions_w=nominal_quaternion,
                )
        for _ in range(max(1, round(float(args_cli.postclose_hold_s) / physics_dt))):
            advance(arm_target, phase="grip_hold")
            peak_contact_force = torch.maximum(peak_contact_force, task_env._gripper_contact_force())
            left_force, right_force = _pdz_finger_contact_forces(task_env)
            peak_left_finger_force = torch.maximum(peak_left_finger_force, left_force)
            peak_right_finger_force = torch.maximum(peak_right_finger_force, right_force)
            if args_cli.fixture_part_pose_until_close:
                _restore_active_part_pose(
                    task_env,
                    selected_parts=selected_parts,
                    positions_w=nominal_position,
                    quaternions_w=nominal_quaternion,
                )
        actual_close_widths = _actual_pdz_widths(task_env)
        if args_cli.disable_part_gravity_until_close:
            _set_part_gravity_disabled(task_env, disabled=False)
            release_steps = round(float(args_cli.gravity_release_hold_s) / physics_dt)
            for _ in range(release_steps):
                advance(arm_target, phase="gravity_release")
                peak_contact_force = torch.maximum(peak_contact_force, task_env._gripper_contact_force())
                left_force, right_force = _pdz_finger_contact_forces(task_env)
                peak_left_finger_force = torch.maximum(peak_left_finger_force, left_force)
                peak_right_finger_force = torch.maximum(peak_right_finger_force, right_force)
        prelift_object_position, _ = _active_part_pose(task_env)
        prelift_tcp_position, _ = task_env.context.get_tcp_pose_w()

        lift_steps = max(1, round(float(args_cli.lift_height_m) / float(args_cli.lift_speed_m_s) / physics_dt))
        q_target = task_env.robot.data.joint_pos[:, task_env.arm_ids].clone()
        peak_object_z = prelift_object_position[:, 2].clone()
        start_tcp_z = prelift_tcp_position[:, 2].clone()
        identity = torch.eye(6, device=task_env.device).expand(task_env.num_envs, -1, -1)
        twist_world = torch.zeros((task_env.num_envs, 6), device=task_env.device)
        twist_world[:, 2] = float(args_cli.lift_speed_m_s)
        for _ in range(lift_steps):
            root_quaternion = task_env.robot.data.root_quat_w
            rotation_base_from_world = matrix_from_quat(quat_conjugate(root_quaternion))
            twist_base = torch.cat(
                (
                    torch.bmm(rotation_base_from_world, twist_world[:, :3, None]).squeeze(-1),
                    torch.bmm(rotation_base_from_world, twist_world[:, 3:, None]).squeeze(-1),
                ),
                dim=-1,
            )
            jacobian = task_env.robot.root_physx_view.get_jacobians()[
                :, task_env.context.ee_jacobi_body_idx, :, task_env.arm_ids
            ]
            transpose = jacobian.transpose(1, 2)
            q_dot = torch.bmm(
                transpose,
                torch.linalg.solve(
                    torch.bmm(jacobian, transpose) + float(args_cli.dls_damping) ** 2 * identity,
                    twist_base.unsqueeze(-1),
                ),
            ).squeeze(-1)
            q_dot.clamp_(
                min=-float(args_cli.maximum_joint_speed_rad_s),
                max=float(args_cli.maximum_joint_speed_rad_s),
            )
            q_target += q_dot * physics_dt
            limits = task_env.robot.data.soft_joint_pos_limits[:, task_env.arm_ids]
            q_target = torch.maximum(torch.minimum(q_target, limits[..., 1]), limits[..., 0])
            advance(q_target, phase="lift")
            object_position, _ = _active_part_pose(task_env)
            peak_object_z = torch.maximum(peak_object_z, object_position[:, 2])
            peak_contact_force = torch.maximum(peak_contact_force, task_env._gripper_contact_force())
            left_force, right_force = _pdz_finger_contact_forces(task_env)
            peak_left_finger_force = torch.maximum(peak_left_finger_force, left_force)
            peak_right_finger_force = torch.maximum(peak_right_finger_force, right_force)
        for _ in range(max(1, round(float(args_cli.postlift_hold_s) / physics_dt))):
            advance(q_target, phase="lift_hold")
            object_position, _ = _active_part_pose(task_env)
            peak_object_z = torch.maximum(peak_object_z, object_position[:, 2])
            peak_contact_force = torch.maximum(peak_contact_force, task_env._gripper_contact_force())
            left_force, right_force = _pdz_finger_contact_forces(task_env)
            peak_left_finger_force = torch.maximum(peak_left_finger_force, left_force)
            peak_right_finger_force = torch.maximum(peak_right_finger_force, right_force)

        final_object_position, _ = _active_part_pose(task_env)
        final_tcp_position, _ = task_env.context.get_tcp_pose_w()
        tcp_final_lift = final_tcp_position[:, 2] - start_tcp_z
        object_final_lift = final_object_position[:, 2] - settled_position[:, 2]
        object_peak_lift = peak_object_z - settled_position[:, 2]
        close_bump = prelift_object_position[:, 2] - settled_position[:, 2]
        initial_relative = prelift_object_position - prelift_tcp_position
        final_relative = final_object_position - final_tcp_position
        relative_drift = torch.linalg.norm(final_relative - initial_relative, dim=-1)

        def cpu(tensor: torch.Tensor) -> list:
            return tensor.detach().cpu().tolist()

        values = {
            "jaw_width": cpu(jaw_widths),
            "catalog_approach_width": cpu(catalog_approach_widths),
            "approach_width": cpu(approach_widths),
            "commanded_close_width": cpu(close_widths),
            "actual_close_width": cpu(actual_close_widths),
            "settle_translation": cpu(settle_translation),
            "settle_rotation": cpu(settle_rotation),
            "settle_peak_contact_force": cpu(settle_peak_contact_force),
            "settle_peak_left_finger_force": cpu(settle_peak_left_finger_force),
            "settle_peak_right_finger_force": cpu(settle_peak_right_finger_force),
            "tcp_final_lift": cpu(tcp_final_lift),
            "object_final_lift": cpu(object_final_lift),
            "object_peak_lift": cpu(object_peak_lift),
            "close_bump": cpu(close_bump),
            "relative_drift": cpu(relative_drift),
            "peak_contact_force": cpu(peak_contact_force),
            "peak_left_finger_force": cpu(peak_left_finger_force),
            "peak_right_finger_force": cpu(peak_right_finger_force),
            "selected_parts": cpu(selected_parts),
            "target_indices": cpu(target_indices),
        }
        rows: list[dict[str, object]] = []
        for env_index in range(task_env.num_envs):
            target_index = int(values["target_indices"][env_index])
            status = classify_scripted_lift(
                commanded_lift_m=float(args_cli.lift_height_m),
                tcp_final_lift_m=float(values["tcp_final_lift"][env_index]),
                object_final_lift_m=float(values["object_final_lift"][env_index]),
                object_peak_lift_m=float(values["object_peak_lift"][env_index]),
                settle_translation_m=float(values["settle_translation"][env_index]),
                settle_rotation_rad=float(values["settle_rotation"][env_index]),
                relative_drift_m=float(values["relative_drift"][env_index]),
                thresholds=thresholds,
            )
            part_index = int(values["selected_parts"][env_index])
            rows.append(
                {
                    "target_index": target_index,
                    "target_id": task_env.target_ids[target_index],
                    "part_id": task_env.part_names[part_index],
                    "status": status,
                    "success": status == "success",
                    "jaw_width_m": float(values["jaw_width"][env_index]),
                    "catalog_approach_width_m": float(values["catalog_approach_width"][env_index]),
                    "approach_width_m": float(values["approach_width"][env_index]),
                    "commanded_close_width_m": float(values["commanded_close_width"][env_index]),
                    "actual_close_width_m": float(values["actual_close_width"][env_index]),
                    "settle_translation_m": float(values["settle_translation"][env_index]),
                    "settle_rotation_deg": math.degrees(float(values["settle_rotation"][env_index])),
                    "settle_peak_gripper_contact_force_n": float(values["settle_peak_contact_force"][env_index]),
                    "settle_peak_left_finger_contact_force_n": float(
                        values["settle_peak_left_finger_force"][env_index]
                    ),
                    "settle_peak_right_finger_contact_force_n": float(
                        values["settle_peak_right_finger_force"][env_index]
                    ),
                    "tcp_final_lift_m": float(values["tcp_final_lift"][env_index]),
                    "object_final_lift_m": float(values["object_final_lift"][env_index]),
                    "object_peak_lift_m": float(values["object_peak_lift"][env_index]),
                    "close_bump_m": float(values["close_bump"][env_index]),
                    "relative_drift_m": float(values["relative_drift"][env_index]),
                    "peak_gripper_contact_force_n": float(values["peak_contact_force"][env_index]),
                    "peak_left_finger_contact_force_n": float(values["peak_left_finger_force"][env_index]),
                    "peak_right_finger_contact_force_n": float(values["peak_right_finger_force"][env_index]),
                    "peak_bilateral_finger_force_n": min(
                        float(values["peak_left_finger_force"][env_index]),
                        float(values["peak_right_finger_force"][env_index]),
                    ),
                    "video_path": "",
                }
            )
        summary = summarize_scripted_lifts(rows)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            args_cli.output_dir.expanduser().resolve()
            if args_cli.output_dir is not None
            else REPO_ROOT
            / "artifacts"
            / "scripted_grasp_lift_validation"
            / (
                f"shard_{shard.shard_index:02d}_{args_cli.catalog_split}"
                f"_round_{int(args_cli.balanced_round):02d}_{stamp}"
            )
        )
        if args_cli.video_dir is not None:
            video_dir = args_cli.video_dir.expanduser().resolve()
            video_dir.mkdir(parents=True, exist_ok=True)
            for env_index, row in enumerate(rows):
                video_path = video_dir / f"scripted_lift_{_safe_filename(str(row['target_id']))}.mp4"
                _write_lift_video(
                    video_path,
                    frames=video_frames[env_index],
                    target_id=str(row["target_id"]),
                    status=str(row["status"]),
                    settled_object_z_m=float(settled_position[env_index, 2].item()),
                    fps=float(args_cli.video_fps),
                )
                row["video_path"] = str(video_path)
        metadata = {
            "dataset_name": shard.dataset_name,
            "dataset_sha256": shard.dataset_sha256,
            "dataset_shard": shard.shard_index,
            "dataset_shard_count": shard.shard_count,
            "catalog_split": args_cli.catalog_split,
            "balanced_round": int(args_cli.balanced_round),
            "target_offset": args_cli.target_offset,
            "dynamic_part": True,
            "static_friction": float(args_cli.static_friction),
            "dynamic_friction": float(args_cli.dynamic_friction),
            "hand_effort_limit_n": float(args_cli.hand_effort_limit_n),
            "hand_stiffness": float(args_cli.hand_stiffness),
            "hand_damping": float(args_cli.hand_damping),
            "part_mass_kg": None if args_cli.part_mass_kg is None else float(args_cli.part_mass_kg),
            "disable_part_gravity_until_close": bool(args_cli.disable_part_gravity_until_close),
            "fixture_part_pose_until_close": bool(args_cli.fixture_part_pose_until_close),
            "gravity_release_hold_s": float(args_cli.gravity_release_hold_s),
            "settle_duration_s": float(args_cli.settle_duration_s),
            "extra_approach_clearance_m": float(args_cli.extra_approach_clearance_m),
            "close_duration_s": float(args_cli.close_duration_s),
            "postclose_hold_s": float(args_cli.postclose_hold_s),
            "squeeze_margin_m": float(args_cli.squeeze_margin_m),
            "lift_height_m": float(args_cli.lift_height_m),
            "lift_speed_m_s": float(args_cli.lift_speed_m_s),
            "postlift_hold_s": float(args_cli.postlift_hold_s),
            "thresholds": thresholds.__dict__,
        }
        _write_outputs(output_dir, rows=rows, summary=summary, metadata=metadata)
        print(
            f"[LIFT] success={summary['successes']}/{summary['valid_attempts']} "
            f"({100.0 * float(summary['success_rate']):.1f}%) "
            f"simulator_invalid={summary['simulator_invalid_attempts']} outcomes={summary['status_counts']}",
            flush=True,
        )
        print(f"[LIFT] report={output_dir / 'report.md'}", flush=True)
    finally:
        if env is not None:
            env.close()
        if temporary_catalog_path is not None:
            temporary_catalog_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
