#!/usr/bin/env python3
"""Render all validated grasp-goal RGB-D observations in one Isaac launch."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--paths-asset",
    type=Path,
    default=Path("isaac_rl/data/multigrasp_50_paths.npz"),
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("isaac_rl/data/multigrasp_50_catalog.npz"),
)
parser.add_argument(
    "--robot-usd",
    type=Path,
    default=Path("assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"),
)
parser.add_argument(
    "--part-usd",
    type=Path,
    default=Path(
        "artifacts/isaac_bundle_assets/pipeline_stage2_ground_feasible_bundle_local.usd"
    ),
)
parser.add_argument("--settle-steps", type=int, default=60)
parser.add_argument(
    "--batch-size",
    type=int,
    default=10,
    help="Parallel goal cameras reused across batches; lower this if Isaac exhausts VRAM.",
)
parser.add_argument("--maximum-position-error-m", type=float, default=0.002)
parser.add_argument("--maximum-rotation-error-deg", type=float, default=1.0)
parser.add_argument(
    "--minimum-depth-std-m",
    type=float,
    default=0.01,
    help="Reject goal views that are nearly a featureless constant-depth plane.",
)
parser.add_argument(
    "--exclusions-file",
    type=Path,
    default=Path("isaac_rl/data/multigrasp_goal_exclusions.json"),
)
parser.add_argument(
    "--target-indices",
    type=int,
    nargs="+",
    default=None,
    help="Capture only these zero-based source target indices (diagnostic use).",
)
parser.add_argument(
    "--contact-sheet",
    type=Path,
    default=None,
    help="Optional PNG contact sheet of the rendered diagnostic targets.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_PROJECT_SOURCE = REPO_ROOT / "isaac_rl/source/isaac_rl"
for import_path in (REPO_ROOT, ISAAC_PROJECT_SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from grasp_planning.d405_wrist_camera import (  # noqa: E402
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
    D405WristCameraConfig,
    camera_pose_in_link7,
)
from grasp_planning.envs.fr3_part_env import make_fr3_part_scene_cfg  # noqa: E402
from grasp_planning.isaac_visual_materials import (  # noqa: E402
    VISUAL_SERVO_MATERIAL_PROFILE,
    apply_visual_servo_materials,
)
from grasp_planning.isaac_visual_scene import (  # noqa: E402
    VISUAL_SERVO_SCENE_PROFILE,
    make_visual_servo_render_cfg,
)
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.start_poses import KUKA_Y_GRIPPER_APPROACH_PROFILE  # noqa: E402
from grasp_planning.visual_servo_workspace import (  # noqa: E402
    VISUAL_SERVO_TSLOT_PROFILE,
    spawn_visual_servo_tslot_surfaces,
)
from isaac_rl.tasks.direct.isaac_rl.multigrasp_catalog import (  # noqa: E402
    load_multigrasp_catalog,
)


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


def _select_targets(payload: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """Return a self-consistent target subset while preserving per-part metadata."""

    source_target_count = len(payload["target_ids"])
    selected: dict[str, np.ndarray] = {}
    for name, value in payload.items():
        if value.ndim >= 1 and value.shape[0] == source_target_count:
            selected[name] = value[indices].copy()
        else:
            selected[name] = value.copy()
    return selected


def _resolve_catalog_asset_path(value: str | Path) -> Path:
    """Resolve a catalogue asset both on the authoring host and in Docker."""

    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    # Multipart assets retain absolute authoring-host paths.  Their suffix
    # starts at isaac_rl/, which is stable inside the project bind mount.
    parts = candidate.parts
    try:
        project_relative = Path(*parts[parts.index("isaac_rl") :])
    except ValueError:
        return candidate.resolve()
    remapped = REPO_ROOT / project_relative
    return remapped.resolve()


def _write_contact_sheet(*, path: Path, rgb: np.ndarray, target_ids: np.ndarray) -> None:
    """Write a small human-reviewable sheet without changing catalogue pixels."""

    from PIL import Image, ImageDraw, ImageFont

    columns = min(4, len(rgb))
    tile_width, tile_height = 384, 216
    label_height = 32
    rows = int(math.ceil(len(rgb) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), (20, 23, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for index, image in enumerate(rgb):
        col, row = index % columns, index // columns
        x, y = col * tile_width, row * (tile_height + label_height)
        tile = Image.fromarray(image).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + tile_height + 7), str(target_ids[index]), fill=(240, 242, 245), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:  # noqa: C901
    if args_cli.settle_steps < 1:
        raise ValueError("--settle-steps must be positive.")
    if args_cli.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    paths_asset = args_cli.paths_asset.resolve()
    robot_usd = args_cli.robot_usd.resolve()
    for required_path in (paths_asset, robot_usd):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    with np.load(paths_asset, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    source_target_count = len(payload["target_ids"])
    if args_cli.target_indices is not None:
        selected_indices = np.asarray(args_cli.target_indices, dtype=np.int64)
        if selected_indices.size == 0 or np.unique(selected_indices).size != selected_indices.size:
            raise ValueError("--target-indices must contain one or more unique indices.")
        if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= source_target_count:
            raise ValueError(
                f"--target-indices must be in [0, {source_target_count - 1}], got {selected_indices.tolist()}."
            )
        payload = _select_targets(payload, selected_indices)
    target_count = len(payload["target_ids"])
    if target_count < 1 or not bool(np.all(payload["moveit_plan_validated"])):
        raise ValueError(
            f"Expected a non-empty MoveIt-validated path catalog, got {target_count}."
        )
    approach_profile = str(
        np.asarray(payload.get("approach_gripper_profile", "")).item()
    )
    if approach_profile != KUKA_Y_GRIPPER_APPROACH_PROFILE:
        raise ValueError(
            "Path asset does not use the required per-grasp approach aperture "
            f"profile '{KUKA_Y_GRIPPER_APPROACH_PROFILE}'. Rebuild the path asset."
        )
    approach_widths = np.asarray(
        payload.get("approach_gripper_widths_m", np.asarray([])), dtype=np.float32
    )
    if approach_widths.shape != (target_count,):
        raise ValueError(
            "Path asset must contain one approach_gripper_widths_m value per target."
        )
    if "part_usd_paths" in payload:
        part_usds = tuple(
            _resolve_catalog_asset_path(str(value))
            for value in payload["part_usd_paths"].tolist()
        )
    else:
        part_usds = (args_cli.part_usd.resolve(),)
    for part_usd in part_usds:
        if not part_usd.is_file():
            raise FileNotFoundError(part_usd)

    # Keep the calibrated 848x480 intrinsic reference exactly as the RL task
    # does.  The renderer may output 256x144, but passing 256x144 into
    # from_intrinsic_matrix without also scaling fx/fy would narrow the FOV by
    # more than 2.5x and produce a zoomed, incompatible goal observation.
    camera_cfg = D405WristCameraConfig(
        enabled=True,
        include_privileged_mask=False,
    )
    first_position = tuple(float(value) for value in payload["object_positions_w"][0])
    first_orientation = tuple(
        float(value) for value in payload["object_orientations_xyzw_w"][0]
    )
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=str(robot_usd),
        part_usd_path=str(part_usds[0]),
        part_position=first_position,
        part_orientation_xyzw=first_orientation,
        part_density_kg_m3=1240.0,
    )
    capture_env_count = min(int(args_cli.batch_size), target_count)
    scene_cfg.num_envs = capture_env_count
    scene_cfg.env_spacing = 1.5
    scene_cfg.part.spawn.rigid_props.kinematic_enabled = True
    scene_cfg.part.prim_path = "{ENV_REGEX_NS}/Part_0"
    for part_index, part_usd in enumerate(part_usds[1:], start=1):
        part_cfg = deepcopy(scene_cfg.part)
        part_cfg.prim_path = f"{{ENV_REGEX_NS}}/Part_{part_index}"
        part_cfg.spawn.usd_path = str(part_usd)
        setattr(scene_cfg, f"part_{part_index}", part_cfg)
    # Keep the deterministic goal reference under the exact same canonical
    # materials, dome, directional key, and RTX settings as the live task.
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=1,
            device=args_cli.device,
            render=make_visual_servo_render_cfg(),
        )
    )
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    scene = InteractiveScene(scene_cfg)
    camera_position, camera_orientation_wxyz = camera_pose_in_link7(camera_cfg)
    camera = TiledCamera(
        TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/link7/D405LeftCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=camera_position,
                rot=camera_orientation_wxyz,
                convention="ros",
            ),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=camera_cfg.intrinsic_matrix_row_major,
                width=camera_cfg.width,
                height=camera_cfg.height,
                clipping_range=camera_cfg.clipping_range_m,
            ),
            width=VISUAL_SERVO_RENDER_WIDTH,
            height=VISUAL_SERVO_RENDER_HEIGHT,
        )
    )
    tslot_bindings = spawn_visual_servo_tslot_surfaces(
        capture_env_count,
        enabled=True,
        geometry_randomization_enabled=False,
    )
    sim.reset()
    scene.reset()
    material_bindings = apply_visual_servo_materials()
    print(
        f"[INFO] Applied goal-capture visual material profile "
        f"{material_bindings['profile']}.",
        flush=True,
    )
    print(
        f"[INFO] Applied canonical render-only workspace profile "
        f"{tslot_bindings['profile']} over the flat collision plane.",
        flush=True,
    )

    robot = scene["robot"]
    parts = [scene["part"]]
    parts.extend(scene[f"part_{index}"] for index in range(1, len(part_usds)))
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=0.084,
    )
    all_position_errors: list[np.ndarray] = []
    all_rotation_errors: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_depth: list[np.ndarray] = []
    for batch_start in range(0, target_count, capture_env_count):
        batch_stop = min(batch_start + capture_env_count, target_count)
        valid_count = batch_stop - batch_start
        batch_indices = np.arange(batch_start, batch_stop, dtype=np.int64)
        if valid_count < capture_env_count:
            batch_indices = np.pad(
                batch_indices,
                (0, capture_env_count - valid_count),
                mode="edge",
            )
        q = torch.as_tensor(
            payload["reset_joint_trajectories"][batch_indices, -1, :],
            dtype=torch.float32,
            device=sim.device,
        )
        qd = torch.zeros_like(q)
        batch_approach_widths = torch.as_tensor(
            approach_widths[batch_indices],
            dtype=torch.float32,
            device=sim.device,
        )
        context.set_fixed_gripper_widths(batch_approach_widths)
        context.write_fixed_gripper_state()
        object_positions = torch.as_tensor(
            payload["object_positions_w"][batch_indices],
            dtype=torch.float32,
            device=sim.device,
        ) + scene.env_origins
        object_quaternions_xyzw = torch.as_tensor(
            payload["object_orientations_xyzw_w"][batch_indices],
            dtype=torch.float32,
            device=sim.device,
        )
        object_pose_wxyz = torch.cat(
            (object_positions, object_quaternions_xyzw[:, (3, 0, 1, 2)]), dim=-1
        )
        zero_part_velocity = torch.zeros(
            (capture_env_count, 6), dtype=torch.float32, device=sim.device
        )
        selected_part_indices = torch.as_tensor(
            payload.get(
                "part_indices", np.zeros(target_count, dtype=np.int64)
            )[batch_indices],
            dtype=torch.long,
            device=sim.device,
        )
        parked_pose = torch.zeros(
            (capture_env_count, 7), dtype=torch.float32, device=sim.device
        )
        parked_pose[:, :3] = scene.env_origins
        parked_pose[:, 2] -= 10.0
        parked_pose[:, 3] = 1.0
        for _ in range(args_cli.settle_steps):
            robot.write_joint_state_to_sim(q, qd, joint_ids=context.arm_joint_ids)
            robot.set_joint_position_target(q, joint_ids=context.arm_joint_ids)
            context.command_fixed_gripper()
            for part_index, part in enumerate(parts):
                part_pose = parked_pose.clone()
                active = selected_part_indices == part_index
                part_pose[active] = object_pose_wxyz[active]
                part.write_root_pose_to_sim(part_pose)
                part.write_root_velocity_to_sim(zero_part_velocity)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt())

        actual_tcp_position, actual_tcp_quaternion_wxyz = context.get_tcp_pose_w()
        actual_tcp_position_local = actual_tcp_position - scene.env_origins
        desired_tcp_position = torch.as_tensor(
            payload["goal_tcp_positions_w"][batch_indices],
            dtype=torch.float32,
            device=sim.device,
        )
        desired_tcp_quaternion_xyzw = torch.as_tensor(
            payload["goal_tcp_orientations_xyzw_w"][batch_indices],
            dtype=torch.float32,
            device=sim.device,
        )
        desired_tcp_quaternion_wxyz = desired_tcp_quaternion_xyzw[:, (3, 0, 1, 2)]
        position_error = torch.linalg.norm(
            actual_tcp_position_local - desired_tcp_position, dim=-1
        )
        quaternion_dot = torch.sum(
            actual_tcp_quaternion_wxyz * desired_tcp_quaternion_wxyz, dim=-1
        ).abs().clamp(max=1.0)
        rotation_error = 2.0 * torch.acos(quaternion_dot)
        rgb_batch = (
            camera.data.output["rgb"][..., :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        depth_batch = (
            camera.data.output["distance_to_image_plane"].detach().cpu().numpy()
        )
        if depth_batch.ndim == 4 and depth_batch.shape[-1] == 1:
            depth_batch = depth_batch[..., 0]
        all_position_errors.append(position_error[:valid_count].detach().cpu().numpy())
        all_rotation_errors.append(rotation_error[:valid_count].detach().cpu().numpy())
        all_rgb.append(rgb_batch[:valid_count])
        all_depth.append(depth_batch[:valid_count])
        print(
            f"[CAPTURE] targets {batch_start + 1}-{batch_stop}/{target_count}",
            flush=True,
        )

    position_error = np.concatenate(all_position_errors).astype(np.float32)
    rotation_error = np.concatenate(all_rotation_errors).astype(np.float32)
    rotation_error_deg = np.degrees(rotation_error)
    worst_position = float(position_error.max())
    worst_rotation_deg = float(rotation_error_deg.max())
    print(
        f"[VALIDATE] worst goal TCP error: {worst_position * 1000.0:.3f} mm / "
        f"{worst_rotation_deg:.4f} deg",
        flush=True,
    )
    rgb = np.concatenate(all_rgb, axis=0)
    depth = np.concatenate(all_depth, axis=0)
    depth = np.nan_to_num(depth, nan=0.50, posinf=0.50, neginf=0.04).astype(
        np.float32
    )
    if rgb.shape != (
        target_count,
        VISUAL_SERVO_RENDER_HEIGHT,
        VISUAL_SERVO_RENDER_WIDTH,
        3,
    ):
        raise RuntimeError(f"Unexpected goal RGB shape: {rgb.shape}.")
    if depth.shape != (
        target_count,
        VISUAL_SERVO_RENDER_HEIGHT,
        VISUAL_SERVO_RENDER_WIDTH,
    ):
        raise RuntimeError(f"Unexpected goal depth shape: {depth.shape}.")
    depth_std = depth.reshape(target_count, -1).std(axis=1)
    rgb_std = rgb.reshape(target_count, -1).std(axis=1)
    position_failure = position_error > float(args_cli.maximum_position_error_m)
    rotation_failure = rotation_error_deg > float(args_cli.maximum_rotation_error_deg)
    quality_failure = depth_std < float(args_cli.minimum_depth_std_m)
    failure = position_failure | rotation_failure | quality_failure
    failed_indices = np.flatnonzero(failure)

    payload["goal_rgb"] = rgb
    payload["goal_depth"] = depth
    payload["goal_tcp_capture_position_error_m"] = position_error
    payload["goal_tcp_capture_rotation_error_rad"] = rotation_error
    payload["goal_rgb_std"] = rgb_std.astype(np.float32)
    payload["goal_depth_std_m"] = depth_std.astype(np.float32)
    payload["capture_validation_passed"] = (~failure).astype(np.bool_)
    payload["isaac_goal_rgbd_captured"] = (~failure).astype(np.bool_)
    payload["visual_material_profile"] = np.asarray(
        VISUAL_SERVO_MATERIAL_PROFILE
    )
    payload["visual_scene_profile"] = np.asarray(VISUAL_SERVO_SCENE_PROFILE)
    payload["visual_tslot_profile"] = np.asarray(VISUAL_SERVO_TSLOT_PROFILE)
    payload["goal_camera_profile"] = np.asarray(
        D405_VISUAL_SERVO_CAMERA_PROFILE
    )
    payload["goal_observation_profile"] = np.asarray(
        D405_VISUAL_SERVO_OBSERVATION_PROFILE
    )
    if args_cli.contact_sheet is not None:
        _write_contact_sheet(
            path=args_cli.contact_sheet.resolve(),
            rgb=rgb,
            target_ids=payload["target_ids"].astype(str),
        )
        print(f"[DONE] Wrote diagnostic contact sheet to {args_cli.contact_sheet.resolve()}.", flush=True)

    target_ids = payload["target_ids"].astype(str)
    part_ids = payload.get("part_ids", np.asarray([""] * target_count)).astype(str)
    orientation_ids = payload["orientation_ids"].astype(str)
    grasp_ids = payload["grasp_ids"].astype(str)
    failures: list[dict[str, object]] = []
    for index in failed_indices.tolist():
        reasons = []
        if position_failure[index]:
            reasons.append("tcp_position_error")
        if rotation_failure[index]:
            reasons.append("tcp_rotation_error")
        if quality_failure[index]:
            reasons.append("goal_depth_std_below_threshold")
        failures.append(
            {
                "target_index": int(index),
                "target_id": str(target_ids[index]),
                "part_id": str(part_ids[index]),
                "orientation_id": str(orientation_ids[index]),
                "grasp_id": str(grasp_ids[index]),
                "reasons": reasons,
                "position_error_mm": float(position_error[index] * 1000.0),
                "rotation_error_deg": float(rotation_error_deg[index]),
                "goal_depth_std_m": float(depth_std[index]),
                "goal_rgb_std": float(rgb_std[index]),
            }
        )
    failures.sort(
        key=lambda item: max(
            float(item["position_error_mm"])
            / (float(args_cli.maximum_position_error_m) * 1000.0),
            float(item["rotation_error_deg"])
            / float(args_cli.maximum_rotation_error_deg),
        ),
        reverse=True,
    )

    output = args_cli.output.resolve()
    validation_report_path = output.with_name(
        f"{output.stem}_validation_report.json"
    )
    validation_report: dict[str, object] = {
        "schema_version": 1,
        "output_catalog": str(output),
        "target_count": int(target_count),
        "thresholds": {
            "maximum_position_error_m": float(args_cli.maximum_position_error_m),
            "maximum_rotation_error_deg": float(args_cli.maximum_rotation_error_deg),
            "minimum_depth_std_m": float(args_cli.minimum_depth_std_m),
        },
        "failure_counts": {
            "any": int(failure.sum()),
            "position": int(position_failure.sum()),
            "rotation": int(rotation_failure.sum()),
            "image_quality": int(quality_failure.sum()),
        },
        "worst_position_error_mm": float(worst_position * 1000.0),
        "worst_rotation_error_deg": float(worst_rotation_deg),
        "failures": failures,
    }
    _atomic_write_json(validation_report_path, validation_report)
    print(
        f"[VALIDATE] failed {len(failures)}/{target_count}: "
        f"position={int(position_failure.sum())}, "
        f"rotation={int(rotation_failure.sum())}, "
        f"image_quality={int(quality_failure.sum())}",
        flush=True,
    )
    for item in failures[:20]:
        print(
            f"[FAILED] index={item['target_index']} target={item['target_id']} "
            f"part={item['part_id']} position={item['position_error_mm']:.3f} mm "
            f"rotation={item['rotation_error_deg']:.4f} deg "
            f"reasons={','.join(item['reasons'])}",
            flush=True,
        )
    if len(failures) > 20:
        print(
            f"[FAILED] ... {len(failures) - 20} more; see {validation_report_path}",
            flush=True,
        )

    quality_rejection_indices = np.flatnonzero(
        quality_failure
    )
    if quality_rejection_indices.size:
        exclusions_path = args_cli.exclusions_file.resolve()
        existing_rejections: dict[str, dict[str, object]] = {}
        if exclusions_path.is_file():
            existing = json.loads(exclusions_path.read_text(encoding="utf-8"))
            existing_rejections = {
                str(item["target_id"]): item for item in existing.get("rejections", [])
            }
        for target_index in quality_rejection_indices:
            target_id = str(payload["target_ids"][target_index])
            existing_rejections[target_id] = {
                "target_id": target_id,
                "reason": "goal_depth_std_below_threshold",
                "depth_std_m": float(depth_std[target_index]),
                "rgb_std": float(rgb_std[target_index]),
            }
        exclusions_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            exclusions_path,
            {
                "schema_version": 1,
                "minimum_depth_std_m": float(args_cli.minimum_depth_std_m),
                "rejections": list(existing_rejections.values()),
            },
        )

    if failed_indices.size:
        diagnostic_output = output.with_name(
            f"{output.stem}_failed_validation.npz"
        )
        _atomic_savez(diagnostic_output, payload)
        raise RuntimeError(
            f"Rejected {len(failures)}/{target_count} captured goals. Preserved all "
            f"rendered RGB-D data at {diagnostic_output} and the exact failure list "
            f"at {validation_report_path}."
        )

    payload["isaac_goal_rgbd_captured"] = np.ones(target_count, dtype=np.bool_)
    _atomic_savez(output, payload)
    load_multigrasp_catalog(output, expected_arm_joint_count=7, require_complete=True)
    print(
        f"[DONE] Wrote and revalidated {target_count} Isaac goal targets to {output}.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit may otherwise close the application before Python prints the
        # exception, making a failed diagnostic capture look successful.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
