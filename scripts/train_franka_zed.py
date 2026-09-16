#!/usr/bin/env python3
"""Train the Panda/ZED pilot with the existing RGB-D actor and completion PPO."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--catalog", type=Path, default=ROOT / "isaac_rl/data/franka_zed_cube/catalog.npz")
parser.add_argument("--camera-profile", type=Path, default=ROOT / "configs/franka_zed_mini.json")
parser.add_argument("--object-usd", default="")
parser.add_argument("--robot-asset-manifest", default="", help="Verified offline mirror of the exact Panda USD")
parser.add_argument("--gripper-open-width", type=float, default=0.06)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument(
    "--distributed", action="store_true", help="One Isaac/PPO worker per GPU under torchrun; --num-envs is per rank"
)
parser.add_argument("--experiment-name", help="Shared run name; required for multiple distributed ranks")
parser.add_argument("--global-minibatch-size", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-dir", type=Path, default=ROOT / "logs/franka_zed")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument(
    "--resume-job", type=int, help="Resume the verified final checkpoint of an earlier job in this run directory"
)
parser.add_argument("--save-frequency", type=int, default=50)
parser.add_argument(
    "--evaluate-after", action="store_true", help="Euler batch wrapper evaluates validation after this segment"
)
parser.add_argument(
    "--evaluate-test-after", action="store_true", help="Euler wrapper evaluates test only after the final segment"
)
parser.add_argument(
    "--evaluate", action="store_true", help="Evaluate a checkpoint on held-out targets, without learning"
)
parser.add_argument("--catalog-split", choices=("train", "validation", "test", "all"), default="train")
parser.add_argument("--evaluation-steps", type=int, default=600)
parser.add_argument(
    "--dry-run", action="store_true", help="Build the player and check one inference; no optimizer or training"
)
parser.add_argument(
    "--no-pretrained",
    action="store_true",
    help="Offline network smoke only; first layers remain frozen in the shared architecture",
)
AppLauncher.add_app_launcher_args(parser)
# Older queued Slurm scripts pass Kit's --/setting as a separate token.
# argparse treats that value as another option unless joined with '='.
# Accept those already-submitted jobs without cancelling their queue position.
for index in range(len(sys.argv) - 2, 0, -1):
    if sys.argv[index] == "--kit_args" and sys.argv[index + 1].startswith("--/"):
        sys.argv[index : index + 2] = ["--kit_args=" + sys.argv[index + 1]]
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1 or not str(args.device).startswith("cuda"):
    parser.error("Use positive environments/iterations and a CUDA device")
if args.evaluate and (not args.checkpoint or args.catalog_split == "train"):
    parser.error("Evaluation needs --checkpoint and a held-out --catalog-split")
world_size = int(os.environ.get("WORLD_SIZE", "1")) if args.distributed else 1
global_rank = int(os.environ.get("RANK", "0")) if args.distributed else 0
local_rank = int(os.environ.get("LOCAL_RANK", "0")) if args.distributed else 0
if world_size > 1 and (not args.experiment_name or args.evaluate):
    parser.error("Distributed runs need --experiment-name; evaluate on one GPU separately")
if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not args.distributed:
    parser.error("Multiple torchrun workers require --distributed")
if args.global_minibatch_size < 1:
    parser.error("Positive global minibatch size required")
if args.save_frequency < 1:
    parser.error("Positive checkpoint frequency required")
if args.resume_job:
    if args.checkpoint:
        parser.error("Use either --checkpoint or --resume-job")
    candidates = list(args.run_dir.glob(f"*_job_{args.resume_job}_*gpu"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one completed run for job {args.resume_job}: {candidates}")
    previous = candidates[0]
    verified = json.loads((previous / "verification.json").read_text())
    completed = json.loads((previous / "training_completed.json").read_text())
    if not verified["passed"] or not completed["completed"]:
        raise ValueError("Continuation requires a verified completed predecessor")
    if verified["catalog_sha256"] != hashlib.sha256(args.catalog.read_bytes()).hexdigest():
        raise ValueError("Continuation requires the exact verified predecessor catalog")
    if not (args.evaluate or args.dry_run) and args.iterations <= verified["epochs"]:
        raise ValueError("Continuation epoch limit must exceed its predecessor")
    args.checkpoint = previous / "nn" / Path(verified["checkpoint"]).name
if args.distributed:
    args.device = f"cuda:{local_rank}"
args.enable_cameras = True
from grasp_planning.rl.franka_distributed_startup import (
    acquire_startup_lock,
    environment_barrier,
    release_startup_lock,
)

startup_lock = acquire_startup_lock()
try:
    app = AppLauncher(args).app
except BaseException:
    release_startup_lock(startup_lock)
    raise

import gymnasium as gym
import yaml
from grasp_planning.rl.distributed_observer import DistributedSafeIsaacAlgoObserver
from grasp_planning.rl.ppo_batching import resolve_local_minibatch_size
from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import register_grasp_completion_runner
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import TASK_ID, FrankaZedEnvCfg
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


class FrankaTrainingObserver(DistributedSafeIsaacAlgoObserver):
    """Retain the writer so a short run flushes before Isaac shuts down."""

    writer = None

    def after_init(self, algo):
        super().after_init(algo)
        self.writer = algo.writer
        self.algorithm = algo
        import torch

        backend = dict(
            rank=global_rank,
            cuda=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            cudnn_enabled=torch.backends.cudnn.enabled,
            mode=os.environ.get("FRANKA_CUDNN_MODE", "unspecified"),
        )
        properties = torch.cuda.get_device_properties(args.device)
        backend.update(gpu_name=properties.name, gpu_uuid=str(getattr(properties, "uuid", "unavailable")))
        (self.run_directory / f"backend_rank_{global_rank}.json").write_text(json.dumps(backend, indent=2) + "\n")
        print(f"[FRANKA BACKEND] {backend}", flush=True)

    def after_print_stats(self, *args, **kwargs):
        super().after_print_stats(*args, **kwargs)
        # Preserve short-run logs and usable sidecars while a long run is alive.
        if global_rank != 0:
            return
        if self.writer is not None:
            self.writer.flush()
        preview_path = self.run_directory / "first_training_rgb.png"
        if not preview_path.exists():
            save_wrist_preview(self.environment, preview_path)
        for checkpoint in self.run_directory.rglob("*.pth"):
            sidecar = checkpoint.with_suffix(".contract.json")
            if not sidecar.exists():
                sidecar.write_text(json.dumps(self.contract, indent=2) + "\n")


def save_wrist_preview(environment, path):
    from PIL import Image

    rgb = environment.wrist_camera.data.output["rgb"][..., :3].detach().cpu()
    image_std = rgb.float().flatten(1).std(dim=1)
    height, width = rgb.shape[1:3]
    preview = Image.new("RGB", (width * 2, height * 2))
    for i in range(min(4, rgb.shape[0])):
        preview.paste(Image.fromarray(rgb[i].numpy()), ((i % 2) * width, (i // 2) * height))
    preview.save(path)
    assert (image_std > 2.0).all(), f"Blank or nearly uniform wrist render; saved {path}"
    return float(image_std.min())


def main():
    cfg = FrankaZedEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed + global_rank
    cfg.catalog_path = str(args.catalog)
    cfg.catalog_split = args.catalog_split
    if args.evaluate:
        import numpy as np

        with np.load(args.catalog, allow_pickle=False) as source:
            count = (
                len(source["target_ids"])
                if args.catalog_split == "all"
                else int((source["split"] == args.catalog_split).sum())
            )
            if "target_part_indices" in source:
                selected = (
                    np.ones(len(source["target_ids"]), dtype=bool)
                    if args.catalog_split == "all"
                    else source["split"] == args.catalog_split
                )
                count = len(np.unique(source["target_part_indices"][selected]))
        args.num_envs = max(args.num_envs, count)
        cfg.scene.num_envs = args.num_envs
        cfg.rgb_gain_randomization = 0.0
        cfg.reset_ready_fraction = 0.0
        cfg.fixed_waypoint_index = 0
        cfg.sequential_target_assignment = True
    cfg.camera_profile_path = str(args.camera_profile)
    cfg.object_usd_path = args.object_usd
    cfg.robot_asset_manifest = args.robot_asset_manifest
    cfg.gripper_open_width_m = args.gripper_open_width
    try:
        env = gym.make(TASK_ID, cfg=cfg)
    finally:
        release_startup_lock(startup_lock)
    environment_barrier(global_rank, world_size)
    contract = env.unwrapped.contract()
    if args.checkpoint:
        # A checkpoint's sidecar is copied beside it when sharing/resuming.
        sidecar = args.checkpoint.with_suffix(".contract.json")
        if not sidecar.is_file() or json.loads(sidecar.read_text()) != contract:
            raise ValueError("Checkpoint needs a matching .contract.json sidecar; refusing cross-camera resume")
    config_path = ROOT / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/rl_games_ppo_cfg.yaml"
    config = yaml.safe_load(config_path.read_text())
    # RL-Games adds global_rank itself; keep environment and learner seeds aligned.
    config["params"]["seed"] = args.seed
    config["params"]["network"]["pretrained"] = not args.no_pretrained
    run = args.run_dir / (args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=args.distributed)
    train = config["params"]["config"]
    train.update(
        name="franka_zed",
        full_experiment_name=run.name,
        train_dir=str(args.run_dir),
        device=args.device,
        device_name=args.device,
        num_actors=args.num_envs,
        multi_gpu=args.distributed and not args.dry_run,
        max_epochs=args.iterations,
        save_best_after=0,
        save_frequency=min(args.save_frequency, args.iterations),
    )
    batch = resolve_local_minibatch_size(
        rollout_batch_size_per_rank=args.num_envs * train["horizon_length"],
        target_global_minibatch_size=args.global_minibatch_size,
        world_size=world_size,
    )
    train["minibatch_size"] = batch
    train["central_value_config"]["minibatch_size"] = batch
    if global_rank == 0:
        (run / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (run / "camera.json").write_text(json.dumps(env.unwrapped.camera_profile, indent=2) + "\n")
        (run / "agent.yaml").write_text(yaml.safe_dump(config))
        (run / "run.json").write_text(
            json.dumps(
                {
                    "catalog": str(args.catalog),
                    "seed": args.seed,
                    "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
                    "catalog_split": args.catalog_split,
                    "multipart_geometry_assignment_version": 1 if contract.get("object_assets") else None,
                    "iterations": args.iterations,
                    "num_envs": args.num_envs,
                    "resume_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "world_size": world_size,
                    "total_envs": args.num_envs * world_size,
                    "local_minibatch_size": batch,
                    "effective_global_minibatch_size": batch * world_size,
                    "sim2real": (
                        "provisional_zed_episode_appearance"
                        if contract.get("appearance_randomization")
                        else "provisional_zed_rgb_gain_only"
                    ),
                    "physics_hz": 120,
                    "policy_hz": 15,
                },
                indent=2,
            )
            + "\n"
        )
    (run / f"rank_{global_rank}.json").write_text(
        json.dumps(
            dict(
                rank=global_rank,
                local_rank=local_rank,
                world_size=world_size,
                device=args.device,
                seed=cfg.seed,
                num_envs=args.num_envs,
                original_local_rank=os.environ.get("ISAAC_RL_ORIGINAL_LOCAL_RANK"),
                selected_gpu=os.environ.get("ISAAC_RL_SELECTED_GPU"),
                robot_asset_manifest=args.robot_asset_manifest,
                assigned_part_indices=env.unwrapped.cfg.env_part_indices,
                assigned_parts=(
                    [env.unwrapped.cfg.object_assets[i]["part_key"] for i in env.unwrapped.cfg.env_part_indices]
                    if env.unwrapped.cfg.object_assets
                    else None
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    if args.checkpoint and global_rank == 0 and not (args.evaluate or args.dry_run):
        # Keep the supplied policy available for post-training evaluation even
        # if the resumed run never improves its recorded best reward.
        (run / "nn").mkdir(exist_ok=True)
        shutil.copyfile(args.checkpoint, run / "nn/franka_zed.pth")
        shutil.copyfile(args.checkpoint.with_suffix(".contract.json"), run / "nn/franka_zed.contract.json")
        shutil.copyfile(args.checkpoint, run / "resume_source.pth")
        shutil.copyfile(args.checkpoint.with_suffix(".contract.json"), run / "resume_source.contract.json")
    wrapped = RlGamesVecEnvWrapper(env, args.device, 5.0, 1.0)
    vecenv.register("IsaacRlgWrapper", lambda name, actors, **kw: RlGamesGpuEnv(name, actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    observer = FrankaTrainingObserver()
    observer.run_directory = run
    observer.contract = contract
    observer.environment = env.unwrapped
    runner = Runner(observer)
    register_grasp_completion_runner(runner)
    runner.load(config)
    runner.reset()
    if args.dry_run or args.evaluate:
        import torch

        player = runner.create_player()
        if args.checkpoint:
            player.restore(str(args.checkpoint))
        obs = wrapped.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        player.get_batch_size(obs, 1)
        if player.is_rnn:
            player.init_rnn()
        with torch.inference_mode():
            actions = player.get_action(obs, is_deterministic=True)
        assert actions.shape == (args.num_envs, 7) and torch.isfinite(actions).all()
        if args.evaluate:
            episodes = []
            seen = set()
            for _ in range(args.evaluation_steps):
                previous_targets = env.unwrapped.target_index.clone()
                with torch.inference_mode():
                    action = player.get_action(obs, is_deterministic=True)
                    result, _, done, _ = wrapped.step(action)
                obs = result["obs"] if isinstance(result, dict) else result
                transition = env.unwrapped.last_transition
                for i in torch.nonzero(done).flatten().tolist():
                    target_id = env.unwrapped.target_ids[int(previous_targets[i])]
                    if target_id in seen:
                        continue
                    record = {
                        key: (bool(value[i]) if value.dtype == torch.bool else float(value[i]))
                        for key, value in transition.items()
                    }
                    record["target_id"] = target_id
                    record["part_key"] = target_id.split("__orientation_")[0]
                    episodes.append(record)
                    seen.add(target_id)
                if len(seen) == len(env.unwrapped.target_ids):
                    break
            report = {
                "checkpoint": str(args.checkpoint),
                "catalog_split": args.catalog_split,
                "episodes": episodes,
                "episode_count": len(episodes),
                "successes": sum(x["success"] for x in episodes),
                "requested_targets": len(env.unwrapped.target_ids),
                "coverage_complete": len(seen) == len(env.unwrapped.target_ids),
            }
            grouped = {
                part: [x for x in episodes if x["part_key"] == part] for part in {x["part_key"] for x in episodes}
            }
            report["per_part_success"] = {
                part: sum(x["success"] for x in values) / len(values) for part, values in grouped.items()
            }
            report["macro_part_success"] = sum(report["per_part_success"].values()) / max(1, len(grouped))
            (run / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            if not report["coverage_complete"]:
                raise RuntimeError(f"Incomplete evaluation: {len(seen)}/{len(env.unwrapped.target_ids)} targets")
            env.close()
            print(f"[FRANKA EVALUATION] {run}: {report['successes']}/{len(episodes)} successes", flush=True)
            return
        for _ in range(10):
            result, reward, done, _ = wrapped.step(actions)
            tensor_obs = result["obs"] if isinstance(result, dict) else result
            assert torch.isfinite(tensor_obs).all() and torch.isfinite(reward).all()
        image_std_min = save_wrist_preview(env.unwrapped, run / f"smoke_rgb_rank_{global_rank}.png")
        collective = None
        if args.distributed:
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group("nccl", rank=global_rank, world_size=world_size)
            value = torch.tensor([global_rank + 1.0], device=args.device)
            torch.distributed.all_reduce(value)
            collective = value.item()
            assert collective == world_size * (world_size + 1) / 2
            torch.distributed.destroy_process_group()
        report = dict(
            passed=True,
            rank=global_rank,
            world_size=world_size,
            action_shape=list(actions.shape),
            optimizer_steps=0,
            simulation_steps=10,
            nccl_sum=collective,
            wrist_rgb_std_min=image_std_min,
        )
        (run / f"dry_run_rank_{global_rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if global_rank == 0:
            (run / "dry_run.json").write_text(json.dumps(report, indent=2) + "\n")
        env.close()
        print(f"[FRANKA DRY RUN] Model inference passed; no training: {run}", flush=True)
        return
    options = {"train": True, "play": False, "sigma": None}
    if args.checkpoint:
        options["checkpoint"] = str(args.checkpoint)
    try:
        runner.run(options)
    finally:
        if observer.writer is not None:
            observer.writer.close()

    def parameter_digest(model):
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    algo = observer.algorithm
    rank_report = dict(
        completed=True,
        rank=global_rank,
        world_size=world_size,
        epoch=int(algo.epoch_num),
        actor_sha256=parameter_digest(algo.model),
    )
    if getattr(algo, "has_central_value", False):
        rank_report["critic_sha256"] = parameter_digest(algo.central_value_net.model)
    (run / f"training_rank_{global_rank}.json").write_text(json.dumps(rank_report, indent=2) + "\n")
    if global_rank == 0:
        for checkpoint in run.rglob("*.pth"):
            checkpoint.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (run / "training_completed.json").write_text(
            json.dumps(dict(completed=True, epoch=int(observer.algorithm.epoch_num), world_size=world_size), indent=2)
            + "\n"
        )
    env.close()
    print(f"[FRANKA TRAIN] Completed bounded run: {run}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit shutdown can terminate the interpreter before Python propagates
        # an exception; preserve the real failure in the log first.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        release_startup_lock(startup_lock)
        from isaaclab.sim import SimulationContext

        context = SimulationContext.instance()
        if context:
            context.clear_all_callbacks()
            context.clear_instance()
        app.close(wait_for_replicator=False)
