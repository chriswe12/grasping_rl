# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import fcntl
import math
import os
import sys
from distutils.util import strtobool
from pathlib import Path

# Keep the external project importable when this file is launched directly by
# Isaac Sim. Python otherwise puts only isaac_rl/scripts/rl_games on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--dataset-index",
    type=Path,
    default=None,
    help="Override the portable Fabrica-all dataset index (advanced/debug use).",
)
parser.add_argument(
    "--dataset-shard",
    type=int,
    default=None,
    help="Use one explicit Fabrica-all shard for a single-GPU probe.",
)
parser.add_argument(
    "--experiment-name",
    type=str,
    default=None,
    help="Stable, human-readable run-directory name for controlled ablations.",
)
parser.add_argument(
    "--global_minibatch_size",
    type=int,
    default=None,
    help=(
        "Target effective PPO minibatch across all distributed ranks. "
        "Defaults to the agent configuration's single-GPU minibatch."
    ),
)
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
    default="combined_sim2real",
    help="Named randomization profile recorded with this training run.",
)
parser.add_argument(
    "--policy-context",
    choices=("action", "action_twist", "action_twist_rotation"),
    default="action",
    help=(
        "Deployment-measurable actor context: previous action only; add normalized camera-frame TCP twist; "
        "or additionally add the continuous 6D base-from-camera orientation."
    ),
)
parser.add_argument(
    "--training-profile",
    choices=(
        "baseline",
        "long_run_improved",
        "robust_no_reward_change",
        "robust_reward_change",
        "lift_conservative",
        "lift_primary",
    ),
    default="baseline",
    help="Named PPO/critic/reset profile recorded with this training run.",
)
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


def _acquire_distributed_startup_lock():
    """Serialize Isaac application and environment startup across local ranks."""
    lock_path_value = os.getenv("ISAAC_RL_STARTUP_LOCK")
    if not lock_path_value:
        return None
    lock_path = Path(lock_path_value)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    print(f"[EULER_DISTRIBUTED] waiting for Isaac startup lock: {lock_path}", flush=True)
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    print(f"[EULER_DISTRIBUTED] acquired Isaac startup lock: {lock_path}", flush=True)
    return lock_handle


def _release_distributed_startup_lock(lock_handle) -> None:
    if lock_handle is None:
        return
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    print("[EULER_DISTRIBUTED] released Isaac startup lock after environment setup", flush=True)


def _wait_for_distributed_environment_barrier(global_rank: int, world_size: int) -> None:
    """Wait until every rank has constructed its Isaac environment.

    Isaac startup is serialized on Euler because concurrent Kit/PhysX startup
    is not reliable.  Without this pre-NCCL barrier, the first rank enters
    RL-Games' distributed rendezvous while the last rank is still constructing
    its scene and can hit PyTorch's ten-minute TCPStore timeout.
    """
    if world_size <= 1:
        return
    barrier_path_value = os.getenv("ISAAC_RL_DISTRIBUTED_READY_DIR")
    if not barrier_path_value:
        raise RuntimeError(
            "Distributed Isaac startup requires ISAAC_RL_DISTRIBUTED_READY_DIR."
        )
    barrier_path = Path(barrier_path_value)
    barrier_path.mkdir(parents=True, exist_ok=True)
    (barrier_path / f"rank_{global_rank}.ready").write_text("ready\n", encoding="utf-8")
    print(
        f"[EULER_DISTRIBUTED] rank {global_rank}/{world_size} waiting at environment barrier: "
        f"{barrier_path}",
        flush=True,
    )
    deadline = time.monotonic() + 30.0 * 60.0
    while True:
        ready_count = sum((barrier_path / f"rank_{rank}.ready").is_file() for rank in range(world_size))
        if ready_count == world_size:
            print(
                f"[EULER_DISTRIBUTED] environment barrier complete ({ready_count}/{world_size})",
                flush=True,
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for distributed environments: {ready_count}/{world_size} ready at "
                f"{barrier_path}."
            )
        time.sleep(1.0)


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse and construct each distributed environment under one shared
# lock. Multiple cold Isaac/PhysX startups on the same node have aborted in
# native allocation code even when GPUs and writable caches were isolated.
startup_lock_handle = _acquire_distributed_startup_lock()
try:
    app_launcher = AppLauncher(args_cli)
except BaseException:
    _release_distributed_startup_lock(startup_lock_handle)
    raise
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import logging
import random
import time
from datetime import datetime

import gymnasium as gym
from rl_games.common import a2c_common, env_configurations, vecenv
from rl_games.torch_runner import Runner
from tensorboardX import SummaryWriter

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import isaac_rl.tasks  # noqa: F401
from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
)
from grasp_planning.rl.distributed_observer import DistributedSafeIsaacAlgoObserver
from grasp_planning.rl.fabrica_dataset import (
    DEFAULT_DATASET_INDEX,
    FABRICA_TASK_ID,
    configure_fabrica_env_cfg,
)
from grasp_planning.rl.policy_context import policy_observation_size, resolve_policy_context
from grasp_planning.rl.policy_timing import PHYSICS_RATE_HZ, POLICY_RATE_HZ
from grasp_planning.rl.ppo_batching import resolve_local_minibatch_size
from grasp_planning.rl.sim2real_profiles import apply_sim2real_profile
from grasp_planning.rl.training_profiles import apply_training_profile
from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import (
    register_grasp_completion_runner,
)


class EssentialSummaryWriter(SummaryWriter):
    """Keep TensorBoard focused on diagnostics needed for this task."""

    _allowed_scalars = {
        "performance/step_inference_rl_update_fps",
        "losses/a_loss",
        "losses/c_loss",
        "losses/cval_loss",
        "losses/entropy",
        "losses/bounds_loss",
        "losses/pose_aux_loss",
        "losses/completion_aux_loss",
        "info/last_lr",
        "info/kl",
        "rewards/iter",
        "episode_lengths/iter",
    }
    _run_metadata: dict[str, object] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._run_metadata:
            self.add_text(
                "configuration/sim2real_profile",
                "```json\n" + json.dumps(self._run_metadata, indent=2) + "\n```",
                0,
            )

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, **kwargs):
        if tag == "losses/completion_probability_mean":
            return super().add_scalar(
                "diagnostics/completion_probability_mean",
                scalar_value,
                global_step,
                walltime,
                **kwargs,
            )
        if tag in self._allowed_scalars or tag.startswith("Episode/"):
            return super().add_scalar(tag, scalar_value, global_step, walltime, **kwargs)
        return None


# RL-Games otherwise writes duplicate step/iteration/time variants for rewards
# and episode length, plus several redundant timing and scheduler plots.
a2c_common.SummaryWriter = EssentialSummaryWriter


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    global_rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("ISAAC_RL_ORIGINAL_LOCAL_RANK", os.getenv("LOCAL_RANK", "0")))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    fabrica_shard = None
    if args_cli.task == FABRICA_TASK_ID:
        fabrica_shard = configure_fabrica_env_cfg(
            env_cfg,
            rank=global_rank,
            world_size=world_size,
            explicit_shard=args_cli.dataset_shard,
            index_path=args_cli.dataset_index or DEFAULT_DATASET_INDEX,
        )
        print(
            f"[INFO] Fabrica dataset={fabrica_shard.dataset_name} "
            f"shard={fabrica_shard.shard_index}/{fabrica_shard.shard_count} "
            f"targets={fabrica_shard.target_count} parts={len(fabrica_shard.part_names)}",
            flush=True,
        )
    sim2real_profile = apply_sim2real_profile(env_cfg, args_cli.sim2real_profile)
    training_profile = apply_training_profile(env_cfg, agent_cfg, args_cli.training_profile)
    context_spec = resolve_policy_context(args_cli.policy_context)
    env_cfg.policy_context_mode = context_spec.name
    env_cfg.observation_space = policy_observation_size(
        context_spec.name,
        image_value_count=VISUAL_SERVO_OBSERVATION_HEIGHT * VISUAL_SERVO_OBSERVATION_WIDTH * 8,
    )
    agent_cfg["params"]["network"]["policy_context_size"] = context_spec.size
    run_profile_metadata = {
        "profile": sim2real_profile.name,
        "profile_id": sim2real_profile.identifier,
        "description": sim2real_profile.description,
        "camera_profile": D405_VISUAL_SERVO_CAMERA_PROFILE,
        "observation_profile": D405_VISUAL_SERVO_OBSERVATION_PROFILE,
        "policy_rate_hz": POLICY_RATE_HZ,
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "overrides": dict(sim2real_profile.overrides),
        "training_profile": training_profile.metadata(),
        "policy_context": {
            "mode": context_spec.name,
            "size": context_spec.size,
            "uses_tcp_twist": context_spec.uses_tcp_twist,
            "uses_camera_rotation": context_spec.uses_camera_rotation,
            "network_input_size": env_cfg.observation_space,
        },
        "distributed": {
            "enabled": bool(args_cli.distributed),
            "world_size": world_size,
            "environments_per_rank": None,
            "total_environments": None,
            "rollout_batch_size_per_rank": None,
            "global_rollout_batch_size": None,
            "target_global_minibatch_size": None,
            "minibatch_size_per_rank": None,
            "effective_global_minibatch_size": None,
            "optimizer_updates_per_epoch": None,
        },
        "dataset": None if fabrica_shard is None else fabrica_shard.metadata(),
    }
    EssentialSummaryWriter._run_metadata = run_profile_metadata
    print(f"[INFO] Sim-to-real profile: {sim2real_profile.identifier} ({sim2real_profile.description})")
    print(f"[INFO] Training profile: {training_profile.identifier} ({training_profile.description})")
    print(
        f"[INFO] Policy context: {context_spec.name} ({context_spec.size} values); "
        f"full observation={env_cfg.observation_space}"
    )
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    run_profile_metadata["distributed"]["environments_per_rank"] = env_cfg.scene.num_envs
    run_profile_metadata["distributed"]["total_environments"] = env_cfg.scene.num_envs * world_size
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
    else:
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = args_cli.experiment_name or agent_cfg["params"]["config"].get("full_experiment_name")
    if not log_dir:
        log_dir = os.getenv("ISAAC_RL_EXPERIMENT_NAME") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    if global_rank == 0:
        dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
        dump_yaml(
            os.path.join(log_root_path, log_dir, "params", "sim2real_profile.yaml"),
            run_profile_metadata,
        )
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")
    print(
        f"[INFO] Distributed rank={global_rank}/{world_size} local_rank={local_rank} "
        f"environments_per_rank={env_cfg.scene.num_envs} "
        f"total_environments={env_cfg.scene.num_envs * world_size}"
    )

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    finally:
        _release_distributed_startup_lock(startup_lock_handle)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # RL-Games applies minibatch_size independently on every rank and then
    # averages gradients. Treat the configured single-GPU value as the target
    # *global* minibatch so adding GPUs does not silently multiply the
    # effective gradient batch and reduce the updates made per rollout.
    train_cfg = agent_cfg["params"]["config"]
    rollout_batch_size = env.unwrapped.num_envs * train_cfg["horizon_length"]
    global_rollout_batch_size = rollout_batch_size * world_size
    target_global_minibatch_size = (
        args_cli.global_minibatch_size
        if args_cli.global_minibatch_size is not None
        else train_cfg["minibatch_size"]
    )
    minibatch_size = resolve_local_minibatch_size(
        rollout_batch_size_per_rank=rollout_batch_size,
        target_global_minibatch_size=target_global_minibatch_size,
        world_size=world_size,
    )
    effective_global_minibatch_size = minibatch_size * world_size
    optimizer_updates_per_epoch = (
        rollout_batch_size // minibatch_size * int(train_cfg["mini_epochs"])
    )
    train_cfg["minibatch_size"] = minibatch_size
    if "central_value_config" in train_cfg:
        train_cfg["central_value_config"]["minibatch_size"] = minibatch_size
    run_profile_metadata["distributed"].update(
        {
            "rollout_batch_size_per_rank": rollout_batch_size,
            "global_rollout_batch_size": global_rollout_batch_size,
            "target_global_minibatch_size": target_global_minibatch_size,
            "minibatch_size_per_rank": minibatch_size,
            "effective_global_minibatch_size": effective_global_minibatch_size,
            "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        }
    )
    if global_rank == 0:
        # Rewrite the initially captured configuration with the resolved
        # runtime batch sizes so pulled artifacts describe what actually ran.
        dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
        dump_yaml(
            os.path.join(log_root_path, log_dir, "params", "sim2real_profile.yaml"),
            run_profile_metadata,
        )
    print(
        f"[INFO] RL-Games rollout batch={rollout_batch_size}, "
        f"global rollout batch={global_rollout_batch_size}, "
        f"target global minibatch={target_global_minibatch_size}, "
        f"minibatch/rank={minibatch_size}, "
        f"effective global minibatch={effective_global_minibatch_size}, "
        f"optimizer updates/epoch={optimizer_updates_per_epoch} "
        f"({env.unwrapped.num_envs} environments/rank)."
    )
    # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([DistributedSafeIsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(DistributedSafeIsaacAlgoObserver())

    register_grasp_completion_runner(runner)
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    _wait_for_distributed_environment_barrier(global_rank, world_size)

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
