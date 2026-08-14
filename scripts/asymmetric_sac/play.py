#!/usr/bin/env python3
"""Run a deterministic asymmetric-SAC visual-servo checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ISAAC_RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ISAAC_RL_ROOT / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_rl.tasks  # noqa: F401
import torch
import yaml

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_tasks.utils.hydra import hydra_task_config

SAC_MODULE_PATH = ISAAC_RL_ROOT / "source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/asymmetric_sac.py"
SAC_CONFIG_PATH = ISAAC_RL_ROOT / "source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/asymmetric_sac_cfg.yaml"
SAC_MODULE_NAME = "isaac_rl_worktree_asymmetric_sac_play"
SAC_SPEC = importlib.util.spec_from_file_location(SAC_MODULE_NAME, SAC_MODULE_PATH)
if SAC_SPEC is None or SAC_SPEC.loader is None:
    raise ImportError(f"Could not load asymmetric SAC module from {SAC_MODULE_PATH}.")
SAC_MODULE = importlib.util.module_from_spec(SAC_SPEC)
sys.modules[SAC_MODULE_NAME] = SAC_MODULE
SAC_SPEC.loader.exec_module(SAC_MODULE)
AsymmetricSacActor = SAC_MODULE.AsymmetricSacActor
VisualObservationSpec = SAC_MODULE.VisualObservationSpec


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    _ppo_agent_cfg: dict,
) -> None:
    agent_cfg = yaml.safe_load(SAC_CONFIG_PATH.read_text(encoding="utf-8"))
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    device = torch.device(str(agent_cfg.get("device", env_cfg.sim.device)))
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    actor_cfg = agent_cfg["actor"]
    actor = AsymmetricSacActor(
        observation_spec=VisualObservationSpec(),
        action_size=int(agent_cfg["critic"]["action_size"]),
        geometry_feature_size=int(actor_cfg["geometry_feature_size"]),
        pretrained=False,
        log_std_min=float(actor_cfg["log_std_min"]),
        log_std_max=float(actor_cfg["log_std_max"]),
    ).to(device)
    payload = torch.load(Path(args_cli.checkpoint), map_location=device, weights_only=False)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    print(
        f"[INFO] Loaded {args_cli.checkpoint} at transition {int(payload.get('transition_count', 0)):,}.",
        flush=True,
    )

    observation, _ = env.reset()
    completed_episodes = 0
    successes = 0
    try:
        while simulation_app.is_running() and completed_episodes < int(args_cli.episodes):
            policy_observation = observation["policy"]
            with torch.inference_mode():
                output = actor.from_observation(
                    policy_observation,
                    deterministic=True,
                    with_log_probability=False,
                )
                action = torch.cat((output.action, torch.sigmoid(output.completion_logits)), dim=-1)
            observation, _, terminated, truncated, extras = env.step(action)
            done = terminated | truncated
            completed_episodes += int(done.sum())
            if isinstance(extras, dict):
                evaluation = extras.get("evaluation")
                if isinstance(evaluation, dict) and isinstance(evaluation.get("success"), torch.Tensor):
                    successes += int(evaluation["success"][done].sum())
        print(
            f"[RESULT] episodes={completed_episodes}, successes={successes}, "
            f"success_rate={successes / max(completed_episodes, 1):.3f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
