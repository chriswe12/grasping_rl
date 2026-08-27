#!/usr/bin/env python3
"""Record composite Isaac policy videos from side and wrist cameras."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0",
    help="Registered Isaac Lab visual-servo play task.",
)
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", default="artifacts/rl_policy_debug_videos")
parser.add_argument("--episode_seconds", type=float, default=15.0)
parser.add_argument("--fps", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=805)
parser.add_argument(
    "--conditions",
    nargs="+",
    choices=("far", "mid", "close", "far_clean", "mid_clean", "close_clean", "exact"),
    default=("far", "mid", "close"),
)
parser.add_argument(
    "--target_indices",
    type=int,
    nargs="+",
    default=None,
    help=("Optional target index per condition. If omitted, every episode draws an independent random catalog target."),
)
parser.add_argument("--catalog_split", choices=("train", "validation", "test", "all"), default="all")
parser.add_argument(
    "--sim2real_profile",
    choices=(
        "nominal",
        "sensor_only",
        "camera_uncertainty",
        "timing_control",
        "appearance",
        "combined_sim2real",
        "combined_clutter",
        "combined_busy_background",
        "combined_depth_robust",
        "stress_test",
    ),
    default="nominal",
    help="Reproducible sensor/camera/timing/appearance profile used for the recorded policy input.",
)
parser.add_argument(
    "--policy-context",
    choices=("action", "action_twist", "action_twist_rotation"),
    default="action",
    help="Actor context contract used by the checkpoint.",
)
parser.add_argument(
    "--force_profile_effects",
    action="store_true",
    help=(
        "Disable the clean-episode mixture and, for clutter profiles, place clutter in the single "
        "debug environment so the requested profile is visible in every diagnostic video."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from grasp_planning.d405_wrist_camera import (  # noqa: E402
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
)
from grasp_planning.rl.policy_context import policy_observation_size, resolve_policy_context  # noqa: E402
from grasp_planning.rl.sim2real_profiles import apply_sim2real_profile  # noqa: E402
from grasp_planning.video import OpenCvVideoWriter  # noqa: E402
from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import (  # noqa: E402
    register_grasp_completion_runner,
)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@dataclass
class DebugFrame:
    side_rgb: np.ndarray
    live_rgb: np.ndarray
    goal_rgb: np.ndarray
    position_error_mm: float
    rotation_error_deg: float
    completion_probability: float
    time_s: float


CONDITIONS = {
    "far": {
        "progress": 0.0,
        "noise_rad": 0.0,
        "rotation_fraction": 1.0,
        "position_fraction": 1.0,
        "object_yaw_fraction": 1.0,
    },
    "mid": {
        "progress": 0.50,
        "noise_rad": 0.0,
        "rotation_fraction": 1.0,
        "position_fraction": 1.0,
        "object_yaw_fraction": 1.0,
    },
    "close": {
        "progress": 0.85,
        "noise_rad": 0.0,
        "rotation_fraction": 1.0,
        "position_fraction": 1.0,
        "object_yaw_fraction": 1.0,
    },
    "far_clean": {
        "progress": 0.0,
        "noise_rad": 0.0,
        "rotation_fraction": 0.0,
        "position_fraction": 0.0,
        "object_yaw_fraction": 0.0,
    },
    "mid_clean": {
        "progress": 0.50,
        "noise_rad": 0.0,
        "rotation_fraction": 0.0,
        "position_fraction": 0.0,
        "object_yaw_fraction": 0.0,
    },
    "close_clean": {
        "progress": 0.85,
        "noise_rad": 0.0,
        "rotation_fraction": 0.0,
        "position_fraction": 0.0,
        "object_yaw_fraction": 0.0,
    },
    # A zero-action exact final-path state for proving that the stored goal is
    # rendered with the same camera, materials, robot, and open fingers.
    "exact": {
        "progress": 1.0,
        "noise_rad": 0.0,
        "rotation_fraction": 0.0,
        "position_fraction": 0.0,
        "object_yaw_fraction": 0.0,
    },
}


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _fit_image(array: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> Image.Image:
    interpolation = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return Image.fromarray(array).resize(size, interpolation)


def _draw_error_plot(
    draw: ImageDraw.ImageDraw,
    *,
    samples: list[DebugFrame],
    current_index: int,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill=(21, 25, 31), outline=(65, 72, 84), width=2)
    draw.text((left + 14, top + 10), "ERROR HISTORY", font=_font(17, bold=True), fill=(235, 238, 244))
    plot_left, plot_top = left + 50, top + 42
    plot_right, plot_bottom = right - 18, bottom - 28
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(95, 102, 115), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(95, 102, 115), width=1)
    visible = samples[: current_index + 1]
    if len(visible) < 2:
        return
    position_max = max(10.0, max(sample.position_error_mm for sample in samples) * 1.05)
    rotation_max = max(5.0, max(sample.rotation_error_deg for sample in samples) * 1.05)

    def points(values: list[float], maximum: float) -> list[tuple[float, float]]:
        denominator = max(len(samples) - 1, 1)
        return [
            (
                plot_left + (plot_right - plot_left) * index / denominator,
                plot_bottom - (plot_bottom - plot_top) * min(value / maximum, 1.0),
            )
            for index, value in enumerate(values)
        ]

    draw.line(
        points([sample.position_error_mm for sample in visible], position_max),
        fill=(66, 196, 255),
        width=3,
    )
    draw.line(
        points([sample.rotation_error_deg for sample in visible], rotation_max),
        fill=(255, 171, 64),
        width=3,
    )
    draw.text(
        (plot_left, bottom - 23),
        f"position 0-{position_max:.1f} mm",
        font=_font(14),
        fill=(66, 196, 255),
    )
    draw.text(
        (plot_left + 230, bottom - 23),
        f"rotation 0-{rotation_max:.1f} deg",
        font=_font(14),
        fill=(255, 171, 64),
    )


def _compose_frame(
    *,
    sample: DebugFrame,
    samples: list[DebugFrame],
    sample_index: int,
    part_id: str,
    target_id: str,
    orientation_id: str,
    condition: str,
    progress: float,
    noise_rad: float,
    rotation_command_deg: float,
    position_offset_mm: float,
    object_yaw_deg: float,
    initial_position_mm: float,
    initial_rotation_deg: float,
    final_position_mm: float,
    final_rotation_deg: float,
    best_position_mm: float,
    best_rotation_deg: float,
    termination: str,
    sim2real_profile: str,
) -> np.ndarray:
    canvas = Image.new("RGB", (1600, 900), (12, 15, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "ISAAC VISUAL-SERVO POLICY DEBUG", font=_font(26, bold=True), fill=(245, 247, 250))
    draw.text(
        (570, 17),
        (
            f"{condition.upper()}  profile={sim2real_profile}  part={part_id}  "
            f"target={target_id}  orientation={orientation_id}"
        ),
        font=_font(18),
        fill=(183, 193, 208),
    )
    canvas.paste(_fit_image(sample.side_rgb, (960, 540)), (20, 55))
    canvas.paste(_fit_image(sample.live_rgb, (512, 288)), (1068, 55))
    canvas.paste(_fit_image(sample.goal_rgb, (512, 288)), (1068, 362))
    draw.rectangle((20, 55, 980, 595), outline=(83, 92, 106), width=2)
    draw.rectangle((1068, 55, 1580, 343), outline=(83, 92, 106), width=2)
    draw.rectangle((1068, 362, 1580, 650), outline=(83, 92, 106), width=2)
    draw.text((38, 70), "EXTERNAL SIDE CAMERA", font=_font(18, bold=True), fill=(255, 255, 255))
    draw.text(
        (1085, 70),
        "LIVE: ISAAC RTX - POLICY RGB 128x72",
        font=_font(17, bold=True),
        fill=(255, 255, 255),
    )
    draw.text(
        (1085, 377),
        "GOAL: MUJOCO FILAMENT - POLICY RGB 128x72",
        font=_font(17, bold=True),
        fill=(255, 255, 255),
    )

    panel = (20, 620, 620, 880)
    draw.rounded_rectangle(panel, radius=10, fill=(21, 25, 31), outline=(65, 72, 84), width=2)
    lines = [
        (f"time {sample.time_s:5.2f} s    result: {termination}", (239, 241, 245)),
        (f"reset progress {progress:.2f}    joint noise +/-{noise_rad:.3f} rad", (181, 191, 205)),
        (
            f"part XY {position_offset_mm:.2f} mm    part yaw {object_yaw_deg:+.2f} deg    "
            f"gripper rot. {rotation_command_deg:.2f} deg",
            (181, 191, 205),
        ),
        (f"initial   {initial_position_mm:8.3f} mm   {initial_rotation_deg:7.3f} deg", (220, 224, 232)),
        (f"current   {sample.position_error_mm:8.3f} mm   {sample.rotation_error_deg:7.3f} deg", (116, 215, 255)),
        (f"policy p(done) {sample.completion_probability:7.4f}", (255, 194, 102)),
        (f"final     {final_position_mm:8.3f} mm   {final_rotation_deg:7.3f} deg", (124, 232, 159)),
        (f"best      {best_position_mm:8.3f} mm   {best_rotation_deg:7.3f} deg", (202, 171, 255)),
        ("strict ready: position <= 4 mm AND rotation <= 3 deg", (255, 194, 102)),
    ]
    for line_index, (text, color) in enumerate(lines):
        draw.text((40, 634 + line_index * 27), text, font=_font(16), fill=color)
    _draw_error_plot(draw, samples=samples, current_index=sample_index, box=(645, 670, 1580, 880))
    return np.asarray(canvas)


def _capture(task_env, time_s: float) -> DebugFrame:
    visual = task_env._camera_observation()[0]
    _, _, position_error, rotation_error = task_env._tcp_error()
    return DebugFrame(
        side_rgb=_rgb_uint8(task_env.debug_camera.data.output["rgb"][0]),
        live_rgb=_rgb_uint8(visual[..., :3]),
        goal_rgb=_rgb_uint8(visual[..., 4:7]),
        position_error_mm=float(torch.linalg.norm(position_error[0]).item() * 1000.0),
        rotation_error_deg=float(torch.rad2deg(torch.linalg.norm(rotation_error[0])).item()),
        completion_probability=float(task_env.completion_probability[0].item()),
        time_s=time_s,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    if args_cli.episode_seconds <= 0.0 or args_cli.fps <= 0.0:
        raise ValueError("Episode duration and FPS must be positive.")
    if args_cli.target_indices is not None and len(args_cli.target_indices) != len(args_cli.conditions):
        raise ValueError("--target_indices must provide exactly one index per condition.")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.catalog_split = args_cli.catalog_split
    env_cfg.fixed_target_index = -1
    env_cfg.fixed_target_id = ""
    env_cfg.random_target_sampling = True
    env_cfg.debug_camera_enabled = True
    sim2real_profile = apply_sim2real_profile(env_cfg, args_cli.sim2real_profile)
    context_spec = resolve_policy_context(args_cli.policy_context)
    env_cfg.policy_context_mode = context_spec.name
    env_cfg.observation_space = policy_observation_size(
        context_spec.name,
        image_value_count=VISUAL_SERVO_OBSERVATION_HEIGHT * VISUAL_SERVO_OBSERVATION_WIDTH * 8,
    )
    agent_cfg["params"]["network"]["policy_context_size"] = context_spec.size
    diagnostic_overrides: dict[str, object] = {}
    if args_cli.force_profile_effects:
        if env_cfg.live_observation_randomization_enabled:
            env_cfg.live_clean_episode_fraction = 0.0
            diagnostic_overrides["live_clean_episode_fraction"] = 0.0
        if env_cfg.scene_clutter_enabled:
            env_cfg.scene_clutter_environment_fraction = 1.0
            diagnostic_overrides["scene_clutter_environment_fraction"] = 1.0
        if env_cfg.scene_busy_background_enabled:
            env_cfg.scene_busy_background_environment_fraction = 1.0
            diagnostic_overrides["scene_busy_background_environment_fraction"] = 1.0
    print(
        f"[INFO] Debug-video sim-to-real profile: {sim2real_profile.identifier} "
        f"({sim2real_profile.description})",
        flush=True,
    )
    if diagnostic_overrides:
        print(f"[INFO] Forced debug-video profile effects: {diagnostic_overrides}", flush=True)
    env_cfg.divergence_position_m = 10.0
    env_cfg.episode_length_s = args_cli.episode_seconds + 2.0
    env_cfg.reset_rotation_fraction_min = 1.0
    env_cfg.reset_rotation_fraction_max = 1.0
    agent_cfg["params"]["seed"] = args_cli.seed

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(checkpoint.parent.parent)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    task_env = env.unwrapped
    if task_env.debug_camera is None:
        raise RuntimeError("Debug camera was not created.")
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concatenate)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = 1
    runner = Runner()
    register_grasp_completion_runner(runner)
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    control_dt = float(task_env.step_dt)
    capture_stride = max(1, int(round(1.0 / (args_cli.fps * control_dt))))
    max_steps = int(math.ceil(args_cli.episode_seconds / control_dt))
    summaries: list[dict[str, object]] = []
    for episode_index, condition in enumerate(args_cli.conditions):
        condition_cfg = CONDITIONS[condition]
        progress = float(condition_cfg["progress"])
        noise_rad = float(condition_cfg["noise_rad"])
        rotation_fraction = float(condition_cfg["rotation_fraction"])
        position_fraction = float(condition_cfg["position_fraction"])
        object_yaw_fraction = float(condition_cfg["object_yaw_fraction"])
        task_env.cfg.reset_progress_min = progress
        task_env.cfg.reset_progress_max = progress
        task_env.cfg.reset_joint_noise_far_rad = noise_rad
        task_env.cfg.reset_joint_noise_near_rad = noise_rad
        task_env.cfg.reset_rotation_fraction_min = rotation_fraction
        task_env.cfg.reset_rotation_fraction_max = rotation_fraction
        task_env.cfg.reset_position_fraction_min = position_fraction
        task_env.cfg.reset_position_fraction_max = position_fraction
        task_env.cfg.reset_object_yaw_fraction_min = object_yaw_fraction
        task_env.cfg.reset_object_yaw_fraction_max = object_yaw_fraction
        task_env.fixed_target_index = (
            int(args_cli.target_indices[episode_index]) if args_cli.target_indices is not None else -1
        )
        obs = env.reset()
        # Explicitly refresh RTX products after changing the fixed target.
        # IsaacLab normally rerenders on reset, but consecutive manual resets
        # through the RL-Games wrapper can otherwise expose the previous
        # episode's final camera product in the first debug frame.
        task_env.scene.write_data_to_sim()
        task_env.sim.forward()
        task_env.sim.render()
        task_env.sim.render()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
        target_index = int(task_env.target_index[0].item())
        target_id = task_env.target_ids[target_index]
        part_index = int(task_env.target_part_indices[target_index].item())
        part_id = task_env.part_names[part_index]
        orientation_index = int(task_env.target_orientation_indices[target_index].item())
        orientation_id = task_env.orientation_names[orientation_index]
        rotation_command_deg = float(torch.rad2deg(task_env.reset_rotation_command[0]).item())
        position_offset_mm = float(torch.linalg.norm(task_env.reset_position_offset[0]).item() * 1000.0)
        object_yaw_deg = float(torch.rad2deg(task_env.reset_object_yaw_offset[0]).item())
        initial_position_mm = float(task_env.initial_position_error[0].item() * 1000.0)
        initial_rotation_deg = float(torch.rad2deg(task_env.initial_rotation_error[0]).item())

        frames: list[DebugFrame] = []
        termination = "timeout"
        for step in range(max_steps + 1):
            should_capture = step % capture_stride == 0 or step == max_steps
            current = _capture(task_env, step * control_dt)
            if should_capture:
                frames.append(current)
            if condition == "exact" and step == 0:
                termination = "exact_goal_reference"
                break
            if step == max_steps:
                break
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=True)
            current.completion_probability = float(torch.as_tensor(actions)[0, -1].item())
            obs, _, dones, _ = env.step(actions)
            if bool(torch.as_tensor(dones).any().item()):
                evaluation = task_env.extras.get("evaluation", {})
                if bool(torch.as_tensor(evaluation.get("success", False)).any().item()):
                    termination = "policy_declared_success"
                elif bool(torch.as_tensor(evaluation.get("premature_completion", False)).any().item()):
                    termination = "policy_declared_prematurely"
                elif bool(torch.as_tensor(evaluation.get("collision", False)).any().item()):
                    termination = "unsafe_collision"
                elif bool(torch.as_tensor(evaluation.get("diverged", False)).any().item()):
                    termination = "diverged"
                else:
                    termination = "timeout"
                if not should_capture:
                    frames.append(current)
                break

        final_position_mm = frames[-1].position_error_mm
        final_rotation_deg = frames[-1].rotation_error_deg
        best_position_mm = min(frame.position_error_mm for frame in frames)
        best_rotation_deg = min(frame.rotation_error_deg for frame in frames)
        maximum_completion_probability = max(frame.completion_probability for frame in frames)
        initial_rgb_mae = float(
            np.mean(np.abs(frames[0].live_rgb.astype(np.float32) - frames[0].goal_rgb.astype(np.float32)))
        )
        safe_target = "".join(character if character.isalnum() or character in "-_" else "_" for character in target_id)
        video_path = output_dir / f"debug_{episode_index + 1:02d}_{condition}_{safe_target}.mp4"
        with OpenCvVideoWriter(video_path, fps=args_cli.fps, width=1600, height=900) as writer:
            for frame_index, frame in enumerate(frames):
                writer.append_rgb(
                    _compose_frame(
                        sample=frame,
                        samples=frames,
                        sample_index=frame_index,
                        part_id=part_id,
                        target_id=target_id,
                        orientation_id=orientation_id,
                        condition=condition,
                        progress=progress,
                        noise_rad=noise_rad,
                        rotation_command_deg=rotation_command_deg,
                        position_offset_mm=position_offset_mm,
                        object_yaw_deg=object_yaw_deg,
                        initial_position_mm=initial_position_mm,
                        initial_rotation_deg=initial_rotation_deg,
                        final_position_mm=final_position_mm,
                        final_rotation_deg=final_rotation_deg,
                        best_position_mm=best_position_mm,
                        best_rotation_deg=best_rotation_deg,
                        termination=termination,
                        sim2real_profile=sim2real_profile.name,
                    )
                )
            for _ in range(int(round(args_cli.fps))):
                writer.append_rgb(
                    _compose_frame(
                        sample=frames[-1],
                        samples=frames,
                        sample_index=len(frames) - 1,
                        part_id=part_id,
                        target_id=target_id,
                        orientation_id=orientation_id,
                        condition=condition,
                        progress=progress,
                        noise_rad=noise_rad,
                        rotation_command_deg=rotation_command_deg,
                        position_offset_mm=position_offset_mm,
                        object_yaw_deg=object_yaw_deg,
                        initial_position_mm=initial_position_mm,
                        initial_rotation_deg=initial_rotation_deg,
                        final_position_mm=final_position_mm,
                        final_rotation_deg=final_rotation_deg,
                        best_position_mm=best_position_mm,
                        best_rotation_deg=best_rotation_deg,
                        termination=termination,
                        sim2real_profile=sim2real_profile.name,
                    )
                )
        summary = {
            "episode": episode_index,
            "condition": condition,
            "target_index": target_index,
            "part_id": part_id,
            "target_id": target_id,
            "orientation_id": orientation_id,
            "reset_progress": progress,
            "reset_noise_rad": noise_rad,
            "reset_rotation_command_deg": rotation_command_deg,
            "reset_position_offset_mm": position_offset_mm,
            "reset_object_yaw_deg": object_yaw_deg,
            "initial_position_error_mm": initial_position_mm,
            "initial_rotation_error_deg": initial_rotation_deg,
            "final_position_error_mm": final_position_mm,
            "final_rotation_error_deg": final_rotation_deg,
            "best_position_error_mm": best_position_mm,
            "best_rotation_error_deg": best_rotation_deg,
            "maximum_completion_probability": maximum_completion_probability,
            "initial_live_goal_rgb_mae_0_255": initial_rgb_mae,
            "termination": termination,
            "video": str(video_path),
        }
        summaries.append(summary)
        print(
            f"[VIDEO] {condition} part={part_id} target={target_id} {termination} "
            f"part_pose={position_offset_mm:.2f} mm/{object_yaw_deg:+.2f} deg "
            f"initial={initial_position_mm:.2f} mm/{initial_rotation_deg:.2f} deg "
            f"final={final_position_mm:.2f} mm/{final_rotation_deg:.2f} deg -> {video_path}",
            flush=True,
        )

    payload = {
        "checkpoint": str(checkpoint),
        "seed": args_cli.seed,
        "sim2real_profile": sim2real_profile.name,
        "sim2real_profile_id": sim2real_profile.identifier,
        "sim2real_profile_description": sim2real_profile.description,
        "sim2real_profile_overrides": dict(sim2real_profile.overrides),
        "diagnostic_profile_overrides": diagnostic_overrides,
        "episodes": summaries,
    }
    (output_dir / "debug_video_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
