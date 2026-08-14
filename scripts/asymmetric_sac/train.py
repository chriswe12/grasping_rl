#!/usr/bin/env python3
"""Train the visual-servo policy with asymmetric Soft Actor-Critic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ISAAC_RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ISAAC_RL_ROOT / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0",
    help="Registered Isaac Lab task.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel Isaac environments.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--total_transitions", type=int, default=None)
parser.add_argument("--replay_capacity", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--minimum_replay_size", type=int, default=None)
parser.add_argument("--random_steps", type=int, default=None)
parser.add_argument("--update_to_data_ratio", type=float, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume actor, critics, temperature, and optimizers.")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Tiny no-pretraining run that exercises collection, replay, one update, and checkpointing.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util
import random
import time
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import isaac_rl.tasks  # noqa: F401
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml

from isaaclab_tasks.utils.hydra import hydra_task_config

SAC_MODULE_PATH = ISAAC_RL_ROOT / "source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/asymmetric_sac.py"
SAC_CONFIG_PATH = ISAAC_RL_ROOT / "source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/asymmetric_sac_cfg.yaml"
SAC_MODULE_NAME = "isaac_rl_worktree_asymmetric_sac"
SAC_SPEC = importlib.util.spec_from_file_location(SAC_MODULE_NAME, SAC_MODULE_PATH)
if SAC_SPEC is None or SAC_SPEC.loader is None:
    raise ImportError(f"Could not load asymmetric SAC module from {SAC_MODULE_PATH}.")
SAC_MODULE = importlib.util.module_from_spec(SAC_SPEC)
sys.modules[SAC_MODULE_NAME] = SAC_MODULE
SAC_SPEC.loader.exec_module(SAC_MODULE)
AsymmetricSacActor = SAC_MODULE.AsymmetricSacActor
AsymmetricSacAgent = SAC_MODULE.AsymmetricSacAgent
CompressedVisualReplayBuffer = SAC_MODULE.CompressedVisualReplayBuffer
TwinQCritic = SAC_MODULE.TwinQCritic
VisualObservationSpec = SAC_MODULE.VisualObservationSpec


def _observation_groups(observation: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(observation, dict) or "policy" not in observation or "critic" not in observation:
        raise TypeError("Asymmetric SAC requires observation groups named 'policy' and 'critic'.")
    return observation["policy"], observation["critic"]


def _scalar(value: object) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach())
    if isinstance(value, (float, int)):
        return float(value)
    return None


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    _ppo_agent_cfg: dict,
) -> None:
    agent_cfg = yaml.safe_load(SAC_CONFIG_PATH.read_text(encoding="utf-8"))
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    seed = int(agent_cfg["seed"] if args_cli.seed is None else args_cli.seed)
    env_cfg.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    replay_cfg = agent_cfg["replay"]
    training_cfg = agent_cfg["training"]
    if args_cli.replay_capacity is not None:
        replay_cfg["capacity"] = args_cli.replay_capacity
    if args_cli.batch_size is not None:
        replay_cfg["batch_size"] = args_cli.batch_size
    if args_cli.minimum_replay_size is not None:
        replay_cfg["minimum_size"] = args_cli.minimum_replay_size
    if args_cli.random_steps is not None:
        replay_cfg["random_steps"] = args_cli.random_steps
    if args_cli.update_to_data_ratio is not None:
        replay_cfg["update_to_data_ratio"] = args_cli.update_to_data_ratio
    if args_cli.total_transitions is not None:
        training_cfg["total_transitions"] = args_cli.total_transitions
    if args_cli.smoke:
        env_cfg.scene.num_envs = min(int(env_cfg.scene.num_envs), 2)
        agent_cfg["actor"]["pretrained"] = False
        replay_cfg.update(
            capacity=64,
            batch_size=2,
            minimum_size=2,
            random_steps=0,
            update_to_data_ratio=1.0,
        )
        training_cfg.update(
            total_transitions=4,
            log_interval_transitions=2,
            checkpoint_interval_transitions=4,
        )

    capacity = int(replay_cfg["capacity"])
    batch_size = int(replay_cfg["batch_size"])
    minimum_size = int(replay_cfg["minimum_size"])
    if not 0 < batch_size <= minimum_size <= capacity:
        raise ValueError(
            "Replay sizes must satisfy 0 < batch_size <= minimum_size <= capacity, "
            f"received {batch_size}, {minimum_size}, {capacity}."
        )

    device = torch.device(str(agent_cfg.get("device", env_cfg.sim.device)))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs") / "asymmetric_sac" / str(agent_cfg["name"]) / run_name
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    writer = SummaryWriter(str(log_dir))
    print(f"[INFO] Asymmetric SAC logs: {log_dir.resolve()}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    unwrapped = env.unwrapped
    num_envs = int(unwrapped.num_envs)
    goal_catalog = unwrapped.goal_rgbd_catalog
    spec = VisualObservationSpec()
    actor_cfg = agent_cfg["actor"]
    actor = AsymmetricSacActor(
        observation_spec=spec,
        action_size=int(agent_cfg["critic"]["action_size"]),
        geometry_feature_size=int(actor_cfg["geometry_feature_size"]),
        pretrained=bool(actor_cfg["pretrained"]),
        log_std_min=float(actor_cfg["log_std_min"]),
        log_std_max=float(actor_cfg["log_std_max"]),
    )
    critic = TwinQCritic(
        state_size=int(agent_cfg["critic"]["state_size"]),
        action_size=int(agent_cfg["critic"]["action_size"]),
    )
    algorithm_cfg = dict(agent_cfg["algorithm"])
    agent = AsymmetricSacAgent(actor, critic, device=device, **algorithm_cfg)
    replay = CompressedVisualReplayBuffer(
        capacity,
        observation_spec=spec,
        critic_state_size=int(agent_cfg["critic"]["state_size"]),
        action_size=int(agent_cfg["critic"]["action_size"]),
        pin_memory=bool(replay_cfg["pin_memory"]),
    )
    print(
        f"[INFO] Replay capacity={capacity:,}, allocated={replay.allocated_bytes / 2**30:.2f} GiB, "
        "canonical goals shared by catalog index.",
        flush=True,
    )

    transition_count = 0
    if args_cli.checkpoint:
        payload = agent.load(args_cli.checkpoint)
        transition_count = int(payload.get("transition_count", 0))
        print(f"[INFO] Resumed {args_cli.checkpoint} at transition {transition_count:,}.", flush=True)

    total_transitions = int(training_cfg["total_transitions"])
    random_steps = int(replay_cfg["random_steps"])
    update_to_data_ratio = float(replay_cfg["update_to_data_ratio"])
    if update_to_data_ratio <= 0.0:
        raise ValueError("update_to_data_ratio must be positive.")
    log_interval = int(training_cfg["log_interval_transitions"])
    checkpoint_interval = int(training_cfg["checkpoint_interval_transitions"])
    next_log = ((transition_count // log_interval) + 1) * log_interval
    next_checkpoint = ((transition_count // checkpoint_interval) + 1) * checkpoint_interval
    metrics: dict[str, list[float]] = defaultdict(list)
    update_budget = 0.0
    start_time = time.time()

    observation, _ = env.reset()
    policy_observation, critic_state = _observation_groups(observation)
    try:
        while transition_count < total_transitions:
            current_goal_index = unwrapped.target_index.detach().clone()
            if transition_count < random_steps:
                motion_action = torch.empty((num_envs, 6), device=device).uniform_(-1.0, 1.0)
                completion_probability = torch.zeros((num_envs, 1), device=device)
            else:
                motion_action, completion_probability = agent.act(
                    policy_observation,
                    deterministic=bool(training_cfg["deterministic_collection"]),
                )
            environment_action = torch.cat((motion_action, completion_probability), dim=-1)
            next_observation, reward, terminated, truncated, extras = env.step(environment_action)
            next_policy_observation, next_critic_state = _observation_groups(next_observation)
            next_goal_index = unwrapped.target_index.detach().clone()
            # Isaac auto-resets vector environments. Treat both true terminals
            # and time limits as episode boundaries so targets never bootstrap
            # from the next episode's reset observation.
            done = terminated | truncated
            replay.add(
                observation=policy_observation,
                critic_state=critic_state,
                goal_index=current_goal_index,
                action=motion_action,
                reward=reward,
                done=done,
                next_observation=next_policy_observation,
                next_critic_state=next_critic_state,
                next_goal_index=next_goal_index,
            )
            transition_count += num_envs
            policy_observation = next_policy_observation
            critic_state = next_critic_state

            if len(replay) >= minimum_size:
                update_budget += num_envs * update_to_data_ratio / batch_size
                while update_budget >= 1.0:
                    batch = replay.sample(batch_size, device=device, goal_rgbd_catalog=goal_catalog)
                    for name, value in agent.update(batch).items():
                        metrics[name].append(value)
                    update_budget -= 1.0

            if isinstance(extras, dict) and isinstance(extras.get("log"), dict):
                for name, value in extras["log"].items():
                    scalar = _scalar(value)
                    if scalar is not None:
                        writer.add_scalar(f"environment/{name}", scalar, transition_count)

            if transition_count >= next_log:
                elapsed = max(time.time() - start_time, 1.0e-6)
                writer.add_scalar("performance/transitions_per_second", transition_count / elapsed, transition_count)
                writer.add_scalar("replay/size", len(replay), transition_count)
                writer.add_scalar("replay/update_to_data_ratio", update_to_data_ratio, transition_count)
                for name, values in metrics.items():
                    if values:
                        writer.add_scalar(f"sac/{name}", sum(values) / len(values), transition_count)
                metrics.clear()
                print(
                    f"[SAC] transitions={transition_count:,}/{total_transitions:,} "
                    f"replay={len(replay):,} updates={agent.update_count:,}",
                    flush=True,
                )
                next_log += log_interval

            if transition_count >= next_checkpoint:
                checkpoint = checkpoint_dir / f"step_{transition_count:012d}.pt"
                agent.save(checkpoint, transition_count=transition_count, agent_cfg=agent_cfg)
                print(f"[INFO] Saved {checkpoint}.", flush=True)
                next_checkpoint += checkpoint_interval

        final_checkpoint = checkpoint_dir / "final.pt"
        agent.save(final_checkpoint, transition_count=transition_count, agent_cfg=agent_cfg)
        print(f"[INFO] Training complete. Final checkpoint: {final_checkpoint}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
