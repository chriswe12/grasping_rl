"""Run a bounded reset/step smoke test for the grasp visual-servo task."""

import argparse

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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import isaac_rl.tasks  # noqa: F401,E402
import torch

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
if args.positive_completion_resets:
    # Bypass the training mixture so this diagnostic remains deterministic.
    cfg.training_reset_mixture_enabled = False
    cfg.completion_positive_reset_fraction = 1.0
    cfg.reset_ready_exact_fraction = 1.0
env = gym.make(task_id, cfg=cfg)
observation, _ = env.reset()
print(
    f"[SMOKE] policy={tuple(observation['policy'].shape)} "
    f"critic={tuple(observation['critic'].shape)} action={env.action_space.shape}",
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
print(
    f"[SMOKE] authored_position_offset_mm="
    f"mean={float(reset_position_mm.mean()):.2f} "
    f"range=[{float(reset_position_mm.min()):.2f}, {float(reset_position_mm.max()):.2f}] "
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
