"""Run a bounded reset/step smoke test for the grasp visual-servo task."""

import argparse
import sys
from pathlib import Path

# Keep the external project importable when this file is launched directly by
# Isaac Sim. Python otherwise puts only isaac_rl/scripts on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--task",
    default=None,
    help="Registered task to smoke-test; overrides --training-resets selection.",
)
parser.add_argument(
    "--training-resets",
    action="store_true",
    help="Use the balanced training reset distribution instead of the far-play config.",
)
parser.add_argument(
    "--positive-completion-resets",
    action="store_true",
    help="Force exact nominal goal resets for autonomous-completion checks.",
)
parser.add_argument(
    "--declare-completion",
    action="store_true",
    help="After the zero-action hold, send one explicit completion action.",
)
parser.add_argument(
    "--sim2real-profile",
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
    default=None,
    help="Apply one full-strength named profile for runtime diagnostics.",
)
parser.add_argument(
    "--policy-context",
    choices=("action", "action_twist", "action_twist_rotation"),
    default="action",
    help="Deployment-measurable actor context to validate.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import isaac_rl.tasks  # noqa: F401,E402
import torch
from grasp_planning.d405_wrist_camera import (
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
)
from grasp_planning.rl.policy_context import policy_observation_size, resolve_policy_context
from grasp_planning.rl.sim2real_profiles import apply_sim2real_profile

from isaaclab_tasks.utils import parse_env_cfg

task_id = args.task or (
    "Grasp-Visual-Servo-RGBD-Direct-v0" if args.training_resets else "Grasp-Visual-Servo-RGBD-Direct-Play-v0"
)
cfg = parse_env_cfg(
    task_id,
    device=args.device,
    num_envs=args.num_envs,
)
cfg.seed = 7
context_spec = resolve_policy_context(args.policy_context)
cfg.policy_context_mode = context_spec.name
cfg.observation_space = policy_observation_size(
    context_spec.name,
    image_value_count=VISUAL_SERVO_OBSERVATION_HEIGHT * VISUAL_SERVO_OBSERVATION_WIDTH * 8,
)
print(
    f"[SMOKE] policy_context={context_spec.name} context_size={context_spec.size} "
    f"observation_size={cfg.observation_space}",
    flush=True,
)
if args.sim2real_profile is not None:
    profile = apply_sim2real_profile(cfg, args.sim2real_profile)
    # A smoke test must exercise the selected profile immediately instead of
    # waiting through the normal training curriculum warmup.
    cfg.training_curriculum_enabled = False
    print(f"[SMOKE] sim2real_profile={profile.identifier}", flush=True)
if args.positive_completion_resets:
    # Bypass the training mixture so this diagnostic remains deterministic.
    cfg.training_reset_mixture_enabled = False
    cfg.completion_positive_reset_fraction = 1.0
    cfg.reset_ready_exact_fraction = 1.0
env = gym.make(task_id, cfg=cfg)
observation, _ = env.reset()
print(
    f"[SMOKE] policy={tuple(observation['policy'].shape)} "
    f"critic={tuple(observation['critic'].shape)} action={env.action_space.shape} "
    f"policy_rate_hz={1.0 / float(env.unwrapped.step_dt):.1f}",
    flush=True,
)
print(
    f"[SMOKE] observation_delay_steps={env.unwrapped.live_observation_delay_steps.detach().cpu().tolist()} "
    f"action_delay_steps={env.unwrapped.motion_action_delay_steps.detach().cpu().tolist()} "
    f"response_scale={env.unwrapped.motion_response_scale.flatten().detach().cpu().tolist()} "
    f"stiffness_scale={env.unwrapped.physics_joint_stiffness_scale.flatten().detach().cpu().tolist()} "
    f"damping_scale={env.unwrapped.physics_joint_damping_scale.flatten().detach().cpu().tolist()}",
    flush=True,
)
if env.unwrapped.live_workspace_appearance_randomizer is not None:
    workspace = env.unwrapped.live_workspace_appearance_randomizer
    layout_names = [variant.name for variant in env.unwrapped.tslot_visual_bindings["variants"]]
    print(
        f"[SMOKE] part_palette_indices={workspace.part_palette_index.detach().cpu().tolist()} "
        f"tslot_background_indices={workspace.background_index.detach().cpu().tolist()} "
        f"tslot_layouts={layout_names} "
        f"collision_surface={env.unwrapped.tslot_visual_bindings['collision_surface']}",
        flush=True,
    )
clutter = env.unwrapped.clutter_visual_bindings
print(
    f"[SMOKE] clutter_profile={clutter['profile']} "
    f"clutter_active_environments={clutter['active_environment_count']}/{env.unwrapped.num_envs} "
    f"clutter_objects={len(clutter['prim_paths'])}",
    flush=True,
)
background = env.unwrapped.busy_background_visual_bindings
print(
    f"[SMOKE] busy_background_profile={background['profile']} "
    f"background_active_environments={background['active_environment_count']}/"
    f"{env.unwrapped.num_envs} standing_people={background['people_count']} "
    f"table_edge_coworkers={background['worker_reach_count']} "
    f"styles={background['style_counts']}",
    flush=True,
)
target_indices = env.unwrapped.target_index.detach().cpu().tolist()
target_ids = [env.unwrapped.target_ids[index] for index in target_indices]
print(
    f"[SMOKE] targets={target_ids} progress={env.unwrapped.reset_progress.detach().cpu().tolist()}",
    flush=True,
)
print(
    f"[SMOKE] reset_modes={env.unwrapped.reset_mode.detach().cpu().tolist()} "
    f"timeouts_s={env.unwrapped.reset_timeout_s.detach().cpu().tolist()} "
    f"positive={env.unwrapped.completion_positive_reset.detach().cpu().tolist()} "
    f"exact={env.unwrapped.completion_exact_reset.detach().cpu().tolist()}",
    flush=True,
)
initial_rotation_deg = torch.rad2deg(env.unwrapped.initial_rotation_error.detach())
reset_rotation_deg = torch.rad2deg(env.unwrapped.reset_rotation_command.detach())
reset_position_mm = torch.linalg.norm(env.unwrapped.reset_position_offset.detach(), dim=-1) * 1000.0
reset_object_yaw_deg = torch.rad2deg(env.unwrapped.reset_object_yaw_offset.detach())
print(
    f"[SMOKE] authored_position_offset_mm="
    f"mean={float(reset_position_mm.mean()):.2f} "
    f"range=[{float(reset_position_mm.min()):.2f}, {float(reset_position_mm.max()):.2f}] "
    f"object_yaw_deg="
    f"mean_abs={float(reset_object_yaw_deg.abs().mean()):.2f} "
    f"range=[{float(reset_object_yaw_deg.min()):.2f}, {float(reset_object_yaw_deg.max()):.2f}] "
    f"authored_rotation_deg="
    f"mean={float(reset_rotation_deg.mean()):.2f} "
    f"range=[{float(reset_rotation_deg.min()):.2f}, {float(reset_rotation_deg.max()):.2f}] "
    f"realized_initial_rotation_deg="
    f"mean={float(initial_rotation_deg.mean()):.2f} "
    f"range=[{float(initial_rotation_deg.min()):.2f}, {float(initial_rotation_deg.max()):.2f}]",
    flush=True,
)
with torch.inference_mode():
    ever_done = False
    for _ in range(args.steps):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observation, reward, terminated, truncated, info = env.step(actions)
        ever_done |= bool(torch.any(terminated | truncated))
log = info.get("log", {})
print(
    f"[SMOKE] steps={args.steps} reward={float(reward.mean()):.4f} "
    f"ever_done={ever_done} position_error_mm={float(log.get('position_error_mm', -1)):.2f} "
    f"rotation_error_deg={float(log.get('rotation_error_deg', -1)):.2f} "
    f"done_signal={float(log.get('completion/stop_signal_mean', -1)):.3f} "
    f"geometric_ready={float(log.get('completion/geometric_ready_rate', -1)):.3f} "
    f"collision_rate={float(log.get('collision_rate', -1)):.3f}",
    flush=True,
)
if args.declare_completion:
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    actions[:, -1] = 1.0
    completion_reward = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    completion_terminated = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    completion_truncated = completion_terminated.clone()
    completion_info = {}
    for _ in range(int(cfg.completion_required_consecutive_steps)):
        (
            _,
            completion_reward,
            completion_terminated,
            completion_truncated,
            completion_info,
        ) = env.step(actions)
    completion_evaluation = completion_info.get("evaluation", {})
    print(
        "[SMOKE] explicit_completion "
        f"reward={float(completion_reward.mean()):.4f} "
        f"done={bool(torch.any(completion_terminated | completion_truncated))} "
        f"correct={float(completion_evaluation.get('success', torch.zeros(1)).float().mean()):.3f} "
        f"premature={float(completion_evaluation.get('premature_completion', torch.zeros(1)).float().mean()):.3f}",
        flush=True,
    )
env.close()
app.close()
