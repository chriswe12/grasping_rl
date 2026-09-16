# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=450, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--dataset-index", type=Path, default=None)
parser.add_argument("--dataset-shard", type=int, default=None)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--reset_progress",
    type=float,
    default=None,
    help="Fixed normalized approach progress: 0=pregrasp/far, 0.5=mid, 1=grasp.",
)
parser.add_argument(
    "--reset_noise_rad",
    type=float,
    default=None,
    help="Fixed independent uniform joint perturbation magnitude in radians.",
)
parser.add_argument(
    "--reset_rotation_deg",
    type=float,
    default=None,
    help="Fixed authored TCP-orientation reset magnitude in degrees (0 to 15).",
)
parser.add_argument(
    "--reset_rotation_range_deg",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Uniform authored TCP-orientation reset range in degrees (0 to 15).",
)
parser.add_argument(
    "--random_targets",
    action="store_true",
    default=False,
    help="Independently sample a random catalog target after every reset.",
)
parser.add_argument(
    "--target_index",
    type=int,
    default=None,
    help="Fixed zero-based target index from the multi-grasp catalog.",
)
parser.add_argument(
    "--target_id",
    type=str,
    default=None,
    help="Fixed catalog target id, for example orientation_002__g1875.",
)
parser.add_argument(
    "--catalog_split",
    choices=("train", "validation", "test", "all"),
    default=None,
    help="Catalog split used for target selection; defaults to the task configuration.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import json  # noqa: I001 - register Isaac Lab tasks before the external task below
import math
import os
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_rl.tasks  # noqa: F401
from grasp_planning.rl.fabrica_dataset import (
    DEFAULT_DATASET_INDEX,
    FABRICA_PLAY_TASK_ID,
    FABRICA_TASK_ID,
    configure_fabrica_env_cfg,
)
from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import (
    register_grasp_completion_runner,
)


def _scalar(value) -> float:
    """Convert an RL-Games scalar or one-element tensor to a Python float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu().item())
    return float(value)


def _initial_pose_error(task_env) -> tuple[float, float]:
    """Read the realized post-reset TCP error for a single environment."""

    position_mm = _scalar(task_env.initial_position_error) * 1000.0
    rotation_deg = math.degrees(_scalar(task_env.initial_rotation_error))
    return position_mm, rotation_deg


def _finish_episode(
    episode: dict[str, object],
    *,
    termination: str,
) -> dict[str, object]:
    """Add terminal and best-error summaries to a completed playback episode."""

    samples = episode["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("Cannot finish a playback episode without error samples.")
    final_sample = samples[-1]
    best_position_sample = min(samples, key=lambda sample: sample["position_error_mm"])
    best_rotation_sample = min(samples, key=lambda sample: sample["rotation_error_deg"])
    episode.update(
        {
            "termination": termination,
            "success": termination == "success",
            "final_position_error_mm": final_sample["position_error_mm"],
            "final_rotation_error_deg": final_sample["rotation_error_deg"],
            "minimum_position_error_mm": best_position_sample["position_error_mm"],
            "minimum_position_error_step": best_position_sample["step"],
            "minimum_rotation_error_deg": best_rotation_sample["rotation_error_deg"],
            "minimum_rotation_error_step": best_rotation_sample["step"],
            "steps": len(samples),
        }
    )
    return episode


class PlaybackMetricsRecorder:
    """Collect exact single-environment pose errors without changing policy inputs."""

    def __init__(self, task_env):
        self.task_env = task_env
        self.episodes: list[dict[str, object]] = []
        self.current_episode: dict[str, object] | None = None
        self.enabled = (
            task_env.num_envs == 1
            and hasattr(task_env, "initial_position_error")
            and hasattr(task_env, "initial_rotation_error")
        )
        if self.enabled:
            self._start_episode(0)
        else:
            print("[WARNING] Detailed playback metrics require a single visual-servo environment.")

    def _start_episode(self, start_step: int) -> None:
        initial_position_mm, initial_rotation_deg = _initial_pose_error(self.task_env)
        target_index = int(self.task_env.target_index[0].detach().cpu().item())
        self.current_episode = {
            "episode": len(self.episodes),
            "start_step": start_step,
            "target_index": target_index,
            "target_id": self.task_env.target_ids[target_index],
            "initial_position_error_mm": initial_position_mm,
            "initial_rotation_error_deg": initial_rotation_deg,
            "reset_position_offset_mm": _scalar(torch.linalg.norm(self.task_env.reset_position_offset, dim=-1))
            * 1000.0,
            "samples": [],
        }

    def record_step(
        self,
        *,
        step: int,
        dones,
        infos: dict[str, object],
        continue_after_done: bool,
    ) -> None:
        """Record the pre-auto-reset error exported by the direct environment."""

        if self.current_episode is None:
            return
        episode_info = infos.get("episode", infos.get("log", {}))
        if "position_error_mm" not in episode_info or "rotation_error_deg" not in episode_info:
            return
        success = _scalar(episode_info.get("success_rate", 0.0)) >= 0.5
        timeout = _scalar(episode_info.get("timeout_rate", 0.0)) >= 0.5
        evaluation = infos.get("evaluation", {})
        completion_probability = _scalar(evaluation.get("completion_probability", 0.0))
        self.current_episode["samples"].append(
            {
                "step": step,
                "position_error_mm": _scalar(episode_info["position_error_mm"]),
                "rotation_error_deg": _scalar(episode_info["rotation_error_deg"]),
                "success": success,
                "completion_probability": completion_probability,
            }
        )
        if not bool(torch.as_tensor(dones).any().item()):
            return
        premature = _scalar(evaluation.get("premature_completion", 0.0)) >= 0.5
        collision = _scalar(evaluation.get("collision", 0.0)) >= 0.5
        termination = (
            "success"
            if success
            else "premature_completion"
            if premature
            else "unsafe_collision"
            if collision
            else "timeout"
            if timeout
            else "diverged"
        )
        self.episodes.append(_finish_episode(self.current_episode, termination=termination))
        self.current_episode = None
        if continue_after_done:
            self._start_episode(step)

    def finish_video(self) -> None:
        """Close a non-terminal episode when recording stops at its requested length."""

        if self.current_episode is not None and self.current_episode["samples"]:
            self.episodes.append(_finish_episode(self.current_episode, termination="video_end"))
            self.current_episode = None

    def report_and_write(
        self,
        *,
        video_folder: Path | None,
        resume_path: str,
        env_cfg,
        reset_progress: float | None,
        reset_noise_rad: float | None,
        reset_rotation_deg: float | None,
        reset_rotation_range_deg: tuple[float, float] | None,
        video_length_steps: int,
        control_dt: float,
    ) -> None:
        """Print concise results and write the full per-step JSON sidecar."""

        if not self.episodes:
            return
        for episode in self.episodes:
            print(
                "[RESULT] "
                f"episode={episode['episode']} target={episode['target_id']} "
                f"termination={episode['termination']} "
                f"spawn_offset={episode['reset_position_offset_mm']:.3f} mm, "
                f"initial={episode['initial_position_error_mm']:.3f} mm / "
                f"{episode['initial_rotation_error_deg']:.3f} deg, "
                f"final={episode['final_position_error_mm']:.3f} mm / "
                f"{episode['final_rotation_error_deg']:.3f} deg, "
                f"best_position={episode['minimum_position_error_mm']:.3f} mm, "
                f"best_rotation={episode['minimum_rotation_error_deg']:.3f} deg"
            )
        if video_folder is None:
            return
        video_folder.mkdir(parents=True, exist_ok=True)
        metrics_path = video_folder / "play_metrics.json"
        metrics_payload = {
            "checkpoint": str(Path(resume_path).resolve()),
            "seed": int(env_cfg.seed),
            "reset_progress": reset_progress,
            "reset_noise_rad": reset_noise_rad,
            "reset_rotation_deg": reset_rotation_deg,
            "reset_rotation_range_deg": reset_rotation_range_deg,
            "control_hz": 1.0 / control_dt,
            "episode_length_s": env_cfg.episode_length_s,
            "video_length_steps": video_length_steps,
            "completion_ready_position_mm": (env_cfg.completion_ready_position_m * 1000.0),
            "completion_ready_rotation_deg": math.degrees(env_cfg.completion_ready_rotation_rad),
            "completion_probability_threshold": (env_cfg.completion_probability_threshold),
            "completion_required_consecutive_steps": (env_cfg.completion_required_consecutive_steps),
            "episodes": self.episodes,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        print(f"[RESULT] Playback metrics written to: {metrics_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(  # noqa: C901 - Isaac Lab playback setup and lifecycle
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    if args_cli.task in (FABRICA_TASK_ID, FABRICA_PLAY_TASK_ID):
        shard = configure_fabrica_env_cfg(
            env_cfg,
            explicit_shard=args_cli.dataset_shard,
            index_path=args_cli.dataset_index or DEFAULT_DATASET_INDEX,
        )
        print(
            f"[INFO] Fabrica dataset shard={shard.shard_index}/{shard.shard_count} "
            f"targets={shard.target_count} parts={len(shard.part_names)}",
            flush=True,
        )

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "live_observation_randomization_enabled"):
        env_cfg.live_observation_randomization_enabled = False
    if hasattr(env_cfg, "scene_appearance_randomization_enabled"):
        env_cfg.scene_appearance_randomization_enabled = False
    if args_cli.catalog_split is not None:
        env_cfg.catalog_split = args_cli.catalog_split
    if args_cli.reset_progress is not None:
        if not hasattr(env_cfg, "reset_progress_min"):
            raise ValueError("--reset_progress is only supported by reset-path tasks.")
        if not 0.0 <= args_cli.reset_progress <= 1.0:
            raise ValueError("--reset_progress must be between 0 and 1.")
        env_cfg.reset_progress_min = args_cli.reset_progress
        env_cfg.reset_progress_max = args_cli.reset_progress
    if args_cli.reset_noise_rad is not None:
        if not hasattr(env_cfg, "reset_joint_noise_far_rad"):
            raise ValueError("--reset_noise_rad is only supported by reset-path tasks.")
        if args_cli.reset_noise_rad < 0.0:
            raise ValueError("--reset_noise_rad must be non-negative.")
        # Equal endpoints make noise independent of approach progress, which
        # lets playback vary distance and perturbation as separate variables.
        env_cfg.reset_joint_noise_far_rad = args_cli.reset_noise_rad
        env_cfg.reset_joint_noise_near_rad = args_cli.reset_noise_rad
    if args_cli.reset_rotation_deg is not None and args_cli.reset_rotation_range_deg is not None:
        raise ValueError("Set only one of --reset_rotation_deg and --reset_rotation_range_deg.")
    if args_cli.reset_rotation_deg is not None:
        if not hasattr(env_cfg, "reset_rotation_far_rad"):
            raise ValueError("--reset_rotation_deg is only supported by rotation-reset tasks.")
        maximum_rotation_deg = math.degrees(env_cfg.reset_rotation_far_rad)
        if not 0.0 <= args_cli.reset_rotation_deg <= maximum_rotation_deg + 1.0e-6:
            raise ValueError(f"--reset_rotation_deg must be between 0 and {maximum_rotation_deg:.1f}.")
        fraction = min(1.0, max(0.0, args_cli.reset_rotation_deg / maximum_rotation_deg))
        env_cfg.reset_rotation_fraction_min = fraction
        env_cfg.reset_rotation_fraction_max = fraction
    if args_cli.reset_rotation_range_deg is not None:
        if not hasattr(env_cfg, "reset_rotation_far_rad"):
            raise ValueError("--reset_rotation_range_deg is only supported by rotation-reset tasks.")
        minimum_deg, maximum_deg = args_cli.reset_rotation_range_deg
        authored_maximum_deg = math.degrees(env_cfg.reset_rotation_far_rad)
        if not 0.0 <= minimum_deg <= maximum_deg <= authored_maximum_deg + 1.0e-6:
            raise ValueError(f"--reset_rotation_range_deg must satisfy 0 <= MIN <= MAX <= {authored_maximum_deg:.1f}.")
        env_cfg.reset_rotation_fraction_min = minimum_deg / authored_maximum_deg
        env_cfg.reset_rotation_fraction_max = min(1.0, maximum_deg / authored_maximum_deg)
    if args_cli.random_targets:
        if not hasattr(env_cfg, "random_target_sampling"):
            raise ValueError("--random_targets is only supported by catalog tasks.")
        env_cfg.random_target_sampling = True
    if args_cli.target_index is not None and args_cli.target_id is not None:
        raise ValueError("Set only one of --target_index and --target_id.")
    if args_cli.target_index is not None:
        if args_cli.target_index < 0:
            raise ValueError("--target_index must be non-negative.")
        env_cfg.fixed_target_index = args_cli.target_index
    if args_cli.target_id is not None:
        env_cfg.fixed_target_id = args_cli.target_id

    if hasattr(env_cfg, "reset_progress_min"):
        print(
            "[INFO] Playback reset: "
            f"progress=[{env_cfg.reset_progress_min:.3f}, {env_cfg.reset_progress_max:.3f}], "
            f"joint_noise=[{env_cfg.reset_joint_noise_near_rad:.4f}, "
            f"{env_cfg.reset_joint_noise_far_rad:.4f}] rad, "
            f"rotation_fraction=[{env_cfg.reset_rotation_fraction_min:.3f}, "
            f"{env_cfg.reset_rotation_fraction_max:.3f}], "
            f"target_sampling={'random' if env_cfg.random_target_sampling else 'balanced'}"
        )

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    video_folder: Path | None = None
    if args_cli.video:
        video_run_name = ["play"]
        if args_cli.reset_progress is not None:
            video_run_name.append(f"progress_{args_cli.reset_progress:.3f}".replace(".", "p"))
        if args_cli.reset_noise_rad is not None:
            video_run_name.append(f"noise_{args_cli.reset_noise_rad:.4f}".replace(".", "p"))
        if args_cli.reset_rotation_deg is not None:
            video_run_name.append(f"rotation_{args_cli.reset_rotation_deg:.1f}deg".replace(".", "p"))
        if args_cli.reset_rotation_range_deg is not None:
            minimum_deg, maximum_deg = args_cli.reset_rotation_range_deg
            video_run_name.append(f"rotation_{minimum_deg:.1f}-{maximum_deg:.1f}deg".replace(".", "p"))
        if args_cli.random_targets:
            video_run_name.append("random_targets")
        if args_cli.target_index is not None:
            video_run_name.append(f"target_{args_cli.target_index:02d}")
        if args_cli.target_id is not None:
            video_run_name.append(f"target_{args_cli.target_id}")
        video_run_name.append(f"seed_{env_cfg.seed}")
        video_folder = Path(log_dir) / "videos" / "_".join(video_run_name)
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    register_grasp_completion_runner(runner)
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    metrics = PlaybackMetricsRecorder(env.unwrapped)
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, infos = env.step(actions)
            timestep += 1
            metrics.record_step(
                step=timestep,
                dones=dones,
                infos=infos,
                continue_after_done=not args_cli.video,
            )

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            # A video is one evaluation attempt. Stop before the environment's
            # automatic reset can add the first frame of another episode.
            if timestep >= args_cli.video_length or bool(torch.as_tensor(dones).any().item()):
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    metrics.finish_video()
    metrics.report_and_write(
        video_folder=video_folder,
        resume_path=resume_path,
        env_cfg=env_cfg,
        reset_progress=args_cli.reset_progress,
        reset_noise_rad=args_cli.reset_noise_rad,
        reset_rotation_deg=args_cli.reset_rotation_deg,
        reset_rotation_range_deg=(
            tuple(args_cli.reset_rotation_range_deg) if args_cli.reset_rotation_range_deg is not None else None
        ),
        video_length_steps=args_cli.video_length,
        control_dt=dt,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
