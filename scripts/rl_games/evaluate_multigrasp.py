# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one RL-Games checkpoint on every visual-servo grasp target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep the external project importable when this file is launched directly by
# Isaac Sim. Python otherwise puts only isaac_rl/scripts/rl_games on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=("Run repeated, deterministic, full-path evaluations for every target in the multi-grasp catalog.")
)
parser.add_argument(
    "--task",
    type=str,
    default="Grasp-Visual-Servo-RGBD-Direct-Play-v0",
    help="Registered Isaac Lab play task.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rl_games_cfg_entry_point",
    help="RL-Games configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate.")
parser.add_argument("--dataset-index", type=Path, default=None)
parser.add_argument("--dataset-shard", type=int, default=None)
parser.add_argument(
    "--dataset-merged",
    action="store_true",
    help="Evaluate the complete merged Fabrica catalog instead of one distributed-training shard.",
)
parser.add_argument(
    "--controller",
    choices=("policy", "blind_nominal"),
    default="policy",
    help="Evaluate the learned policy or an open-loop nominal straight-line baseline.",
)
parser.add_argument("--runs_per_target", type=int, default=3)
parser.add_argument("--episode_seconds", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--nominal_noise_rad",
    type=float,
    default=0.0,
    help="Joint reset noise. Keep zero for collision-validated multipart evaluation.",
)
parser.add_argument(
    "--stress_noise_rad",
    type=float,
    default=0.0,
    help="Stress joint noise. Nonzero values are rejected by collision-safe tasks.",
)
parser.add_argument(
    "--mid_noise_rad",
    type=float,
    default=0.0,
    help="Joint reset noise used by the mid-path condition (progress 0.50).",
)
parser.add_argument(
    "--close_noise_rad",
    type=float,
    default=0.0,
    help="Joint reset noise used by the close-path condition (progress 0.85).",
)
parser.add_argument(
    "--rotation_deg",
    type=float,
    default=0.0,
    help="Fixed authored TCP-orientation reset magnitude for every condition.",
)
parser.add_argument(
    "--conditions",
    nargs="+",
    choices=("nominal", "stress", "far", "mid", "close"),
    default=("far", "mid", "close"),
    help=(
        "Evaluation conditions. nominal/far use progress 0, stress uses progress 0 "
        "with larger noise, mid uses progress 0.50, and close uses progress 0.85."
    ),
)
parser.add_argument(
    "--catalog_split",
    choices=("train", "validation", "test", "all"),
    default=None,
    help="Catalog split to evaluate. Defaults to the selected task configuration.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Defaults below the checkpoint run.",
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
    default="nominal",
    help="Reproducible sensor/camera/timing/appearance evaluation profile.",
)
parser.add_argument(
    "--policy-context",
    choices=("action", "action_twist", "action_twist_rotation"),
    default="action",
    help="Actor context contract used by the checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from statistics import mean, median  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from grasp_planning.d405_wrist_camera import (  # noqa: E402
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
)
from grasp_planning.rl.completion_diagnostics import CompletionDiagnostics  # noqa: E402
from grasp_planning.rl.fabrica_dataset import (  # noqa: E402
    DEFAULT_DATASET_INDEX,
    FABRICA_PLAY_TASK_ID,
    FABRICA_TASK_ID,
    configure_fabrica_env_cfg,
)
from grasp_planning.rl.policy_context import policy_observation_size, resolve_policy_context  # noqa: E402
from grasp_planning.rl.sim2real_profiles import apply_sim2real_profile  # noqa: E402
from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import (  # noqa: E402
    register_grasp_completion_runner,
)
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,  # noqa: E402
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    successes = sum(bool(row["success"]) for row in rows)
    operational_successes = sum(bool(row.get("operational_success", row["success"])) for row in rows)
    terminations = Counter(str(row["termination"]) for row in rows)
    attempts = len(rows)
    return {
        "attempts": attempts,
        "successes": successes,
        "success_rate": successes / attempts if rows else 0.0,
        "operational_successes": operational_successes,
        "operational_success_rate": operational_successes / attempts if rows else 0.0,
        "terminations": dict(sorted(terminations.items())),
        "termination_rates": {name: count / attempts for name, count in sorted(terminations.items())},
        "initial_position_error_mm_mean": mean(float(row["initial_position_error_mm"]) for row in rows),
        "initial_rotation_error_deg_mean": mean(float(row["initial_rotation_error_deg"]) for row in rows),
        "final_position_error_mm_mean": mean(float(row["final_position_error_mm"]) for row in rows),
        "final_position_error_mm_median": median(float(row["final_position_error_mm"]) for row in rows),
        "final_rotation_error_deg_mean": mean(float(row["final_rotation_error_deg"]) for row in rows),
        "final_rotation_error_deg_median": median(float(row["final_rotation_error_deg"]) for row in rows),
        "best_position_error_mm_mean": mean(float(row["best_position_error_mm"]) for row in rows),
        "best_rotation_error_deg_mean": mean(float(row["best_rotation_error_deg"]) for row in rows),
        "reset_progress_mean": mean(float(row["reset_progress"]) for row in rows),
        "reset_noise_rad_mean": mean(float(row["reset_noise_rad"]) for row in rows),
        "reset_rotation_command_deg_mean": mean(float(row["reset_rotation_command_deg"]) for row in rows),
        "reset_position_offset_mm_mean": mean(float(row["reset_position_offset_mm"]) for row in rows),
        "reset_position_requested_mm_mean": mean(float(row["reset_position_requested_mm"]) for row in rows),
        "reset_object_yaw_deg_mean": mean(abs(float(row["reset_object_yaw_deg"])) for row in rows),
        "reset_object_yaw_requested_deg_mean": mean(float(row["reset_object_yaw_requested_deg"]) for row in rows),
        "final_completion_probability_mean": mean(float(row["final_completion_probability"]) for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    *,
    checkpoint: Path,
    episode_seconds: float,
    rotation_deg: float,
    catalog_split: str,
    rows: list[dict[str, object]],
    completion_diagnostics: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_orientation: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_part: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_target: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        condition = str(row["condition"])
        by_condition[condition].append(row)
        by_orientation[(condition, str(row["orientation_id"]))].append(row)
        by_part[(condition, str(row["part_id"]))].append(row)
        by_target[(condition, int(row["target_index"]))].append(row)

    conditions = {name: _summarize(group) for name, group in by_condition.items()}
    orientations = {
        f"{condition}/{orientation}": _summarize(group) for (condition, orientation), group in by_orientation.items()
    }
    target_rows: list[dict[str, object]] = []
    target_summaries: dict[str, dict[str, object]] = {}
    for (condition, target_index), group in sorted(by_target.items()):
        metrics = _summarize(group)
        key = f"{condition}/{group[0]['target_id']}"
        target_summaries[key] = metrics
        target_rows.append(
            {
                "condition": condition,
                "target_index": target_index,
                "target_id": group[0]["target_id"],
                "orientation_id": group[0]["orientation_id"],
                "part_id": group[0]["part_id"],
                **metrics,
            }
        )

    note_parts = [f"Evaluation catalog split: {catalog_split}."]
    if catalog_split in ("validation", "test"):
        note_parts.append("Exact (part_id, grasp_id) groups are held out from the training split.")
    if "stress" in by_condition:
        note_parts.append("The stress condition is a held-out reset perturbation, not a held-out grasp set.")
    if rotation_deg > 0.0:
        note_parts.append(
            f"Every attempt used the full tapered rotation profile: up to "
            f"{rotation_deg:.1f} degrees at pregrasp, decreasing toward grasp."
        )
    if any(float(row["reset_position_requested_mm"]) > 0.0 for row in rows):
        note_parts.append(
            "Horizontal position offsets are measured from the nominal path waypoint "
            "and conservatively capped by each reset state's validated collision clearance."
        )
    if any(float(row["reset_object_yaw_requested_deg"]) > 0.0 for row in rows):
        note_parts.append(
            "Object yaw moves the physical part and its part-relative target rigidly while the canonical "
            "goal image remains fixed; surface displacement is collision-clearance capped."
        )
    note = " ".join(note_parts)
    payload = {
        "checkpoint": str(checkpoint),
        "controller": args_cli.controller,
        "episode_seconds": episode_seconds,
        "note": note,
        "conditions": conditions,
        "orientations": orientations,
        "parts": {f"{condition}/{part_id}": _summarize(group) for (condition, part_id), group in by_part.items()},
        "targets": target_summaries,
        "completion_diagnostics": completion_diagnostics,
    }

    lines = [
        "# Multi-grasp policy evaluation",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        note,
        "",
        f"Each attempt had up to {episode_seconds:.1f} s.",
        "",
        (
            "| Condition | Progress | Position offset | Object yaw | Rotation cmd. | Attempts | Strict | Operational "
            "| Initial pos. | Initial rot. | Final pos. | Final rot. |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in conditions.items():
        lines.append(
            f"| {condition} | {metrics['reset_progress_mean']:.2f} "
            f"| {metrics['reset_position_offset_mm_mean']:.2f} mm "
            f"| {metrics['reset_object_yaw_deg_mean']:.2f} deg "
            f"| {metrics['reset_rotation_command_deg_mean']:.2f} deg "
            f"| {metrics['attempts']} | {100.0 * metrics['success_rate']:.1f}% "
            f"| {100.0 * metrics['operational_success_rate']:.1f}% "
            f"| {metrics['initial_position_error_mm_mean']:.2f} mm "
            f"| {metrics['initial_rotation_error_deg_mean']:.2f} deg "
            f"| {metrics['final_position_error_mm_mean']:.2f} mm "
            f"| {metrics['final_rotation_error_deg_mean']:.2f} deg |"
        )
    lines.extend(
        [
            "",
            "## Completion-head diagnostics",
            "",
            (
                "These per-step metrics use the operational geometric completion labels during evaluation. "
                "Precision and recall apply the raw probability threshold before the four-frame deployment hold."
            ),
            "",
            (
                "| Condition | Samples | Ready | Precision | Recall | False positive "
                "| Brier | ECE | p(ready) | p(negative) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, metrics in completion_diagnostics.items():
        lines.append(
            f"| {condition} | {metrics['supervised_samples']} "
            f"| {100.0 * metrics['positive_rate']:.2f}% "
            f"| {100.0 * metrics['precision']:.2f}% "
            f"| {100.0 * metrics['recall']:.2f}% "
            f"| {100.0 * metrics['false_positive_rate']:.2f}% "
            f"| {metrics['brier_score']:.4f} "
            f"| {metrics['expected_calibration_error']:.4f} "
            f"| {metrics['ready_probability_mean']:.3f} "
            f"| {metrics['negative_probability_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per part",
            "",
            "| Condition | Part | Strict | Operational | Final pos. | Final rot. |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for (condition, part_id), group in sorted(by_part.items()):
        metrics = _summarize(group)
        lines.append(
            f"| {condition} | {part_id} | {100.0 * metrics['success_rate']:.1f}% "
            f"| {100.0 * metrics['operational_success_rate']:.1f}% "
            f"| {metrics['final_position_error_mm_mean']:.2f} mm "
            f"| {metrics['final_rotation_error_deg_mean']:.2f} deg |"
        )
    lines.extend(
        [
            "",
            "## Per orientation",
            "",
            "| Condition | Orientation | Strict | Operational | Final pos. | Final rot. |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for (condition, orientation), group in sorted(by_orientation.items()):
        metrics = _summarize(group)
        lines.append(
            f"| {condition} | {orientation} | {100.0 * metrics['success_rate']:.1f}% "
            f"| {100.0 * metrics['operational_success_rate']:.1f}% "
            f"| {metrics['final_position_error_mm_mean']:.2f} mm "
            f"| {metrics['final_rotation_error_deg_mean']:.2f} deg |"
        )
    return payload, target_rows, "\n".join(lines) + "\n"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(  # noqa: C901 - batched evaluation setup, rollout, and reporting
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    if args_cli.runs_per_target < 1:
        raise ValueError("--runs_per_target must be at least one.")
    if args_cli.episode_seconds <= 0.0:
        raise ValueError("--episode_seconds must be positive.")
    if args_cli.task in (FABRICA_TASK_ID, FABRICA_PLAY_TASK_ID):
        shard = configure_fabrica_env_cfg(
            env_cfg,
            explicit_shard=args_cli.dataset_shard,
            merged=args_cli.dataset_merged,
            index_path=args_cli.dataset_index or DEFAULT_DATASET_INDEX,
        )
        print(
            f"[INFO] Fabrica dataset shard={shard.shard_index}/{shard.shard_count} "
            f"targets={shard.target_count} parts={len(shard.part_names)}",
            flush=True,
        )
        if args_cli.dataset_merged:
            # The merged evaluation scene contains every part variant in every
            # environment. Keep PhysX broad-phase collisions reliable at the
            # full 137-target validation batch instead of accepting its
            # "simulation will miss interactions" warning.
            env_cfg.sim.physx.gpu_found_lost_pairs_capacity = max(
                int(env_cfg.sim.physx.gpu_found_lost_pairs_capacity),
                2**23,
            )
    sim2real_profile = apply_sim2real_profile(env_cfg, args_cli.sim2real_profile)
    context_spec = resolve_policy_context(args_cli.policy_context)
    env_cfg.policy_context_mode = context_spec.name
    env_cfg.observation_space = policy_observation_size(
        context_spec.name,
        image_value_count=VISUAL_SERVO_OBSERVATION_HEIGHT * VISUAL_SERVO_OBSERVATION_WIDTH * 8,
    )
    agent_cfg["params"]["network"]["policy_context_size"] = context_spec.size
    if any(
        value < 0.0
        for value in (
            args_cli.nominal_noise_rad,
            args_cli.stress_noise_rad,
            args_cli.mid_noise_rad,
            args_cli.close_noise_rad,
        )
    ):
        raise ValueError("Reset noise must be non-negative.")

    if args_cli.catalog_split is not None:
        env_cfg.catalog_split = args_cli.catalog_split
    with np.load(Path(env_cfg.goal_catalog_data_path).expanduser(), allow_pickle=False) as source:
        if "split_ids" in source and env_cfg.catalog_split not in ("", "all"):
            evaluation_target_count = int(np.sum(source["split_ids"].astype(str) == str(env_cfg.catalog_split)))
        else:
            evaluation_target_count = len(source["target_ids"])
    if evaluation_target_count < 1:
        raise ValueError(f"Catalog split '{env_cfg.catalog_split}' contains no targets.")
    # One environment per target makes each repeat exactly balanced.
    env_cfg.scene.num_envs = evaluation_target_count
    env_cfg.episode_length_s = args_cli.episode_seconds
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.fixed_target_index = -1
    env_cfg.fixed_target_id = ""
    # Play configurations intentionally draw targets independently. Evaluation
    # instead requires an exactly balanced pass with one environment per target.
    env_cfg.random_target_sampling = False
    env_cfg.sequential_target_sampling = True
    maximum_rotation_deg = math.degrees(env_cfg.reset_rotation_far_rad)
    if not 0.0 <= args_cli.rotation_deg <= maximum_rotation_deg + 1.0e-6:
        raise ValueError(f"--rotation_deg must be between 0 and {maximum_rotation_deg:.1f}.")
    rotation_fraction = min(1.0, max(0.0, args_cli.rotation_deg / maximum_rotation_deg))
    env_cfg.reset_rotation_fraction_min = rotation_fraction
    env_cfg.reset_rotation_fraction_max = rotation_fraction
    agent_cfg["params"]["seed"] = args_cli.seed

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    run_dir = checkpoint.parent.parent
    env_cfg.log_dir = str(run_dir)
    if args_cli.output_dir:
        output_dir = Path(args_cli.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rotation_label = f"rot{args_cli.rotation_deg:.1f}deg".replace(".", "p")
        output_dir = (
            run_dir
            / "evaluations"
            / (
                f"multigrasp_{env_cfg.catalog_split}_{args_cli.sim2real_profile}_"
                f"15s_{args_cli.runs_per_target}x_{rotation_label}_{stamp}"
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concatenate)
    task_env = env.unwrapped
    if task_env.target_count != task_env.num_envs:
        raise RuntimeError(
            f"Evaluator expects one environment per target, got {task_env.num_envs} "
            f"environments and {task_env.target_count} targets."
        )

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = task_env.num_envs
    runner = Runner()
    register_grasp_completion_runner(runner)
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    max_steps = int(math.ceil(args_cli.episode_seconds / task_env.step_dt))
    available_conditions = {
        "nominal": (0.0, args_cli.nominal_noise_rad),
        "stress": (0.0, args_cli.stress_noise_rad),
        "far": (0.0, args_cli.nominal_noise_rad),
        "mid": (0.50, args_cli.mid_noise_rad),
        "close": (0.85, args_cli.close_noise_rad),
    }
    conditions = tuple((condition, *available_conditions[condition]) for condition in args_cli.conditions)
    rows: list[dict[str, object]] = []
    completion_threshold = float(task_env.cfg.completion_probability_threshold)
    completion_diagnostics_overall = CompletionDiagnostics(threshold=completion_threshold)
    completion_diagnostics_by_condition: dict[str, CompletionDiagnostics] = {}
    print(
        f"[EVAL] checkpoint={checkpoint} targets={task_env.target_count} "
        f"runs/target={args_cli.runs_per_target} horizon={max_steps} steps "
        f"({args_cli.episode_seconds:.1f} s) rotation={args_cli.rotation_deg:.1f} deg",
        flush=True,
    )

    for condition, progress_value, noise_rad in conditions:
        condition_completion_diagnostics = CompletionDiagnostics(threshold=completion_threshold)
        completion_diagnostics_by_condition[condition] = condition_completion_diagnostics
        task_env.cfg.reset_progress_min = progress_value
        task_env.cfg.reset_progress_max = progress_value
        task_env.cfg.reset_joint_noise_far_rad = noise_rad
        task_env.cfg.reset_joint_noise_near_rad = noise_rad
        condition_start = len(rows)
        for repeat in range(args_cli.runs_per_target):
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            target_indices = _cpu(task_env.target_index).long()
            if sorted(target_indices.tolist()) != list(range(task_env.target_count)):
                raise RuntimeError("A repeat did not contain exactly one sample of every target.")
            initial_position = _cpu(task_env.initial_position_error) * 1000.0
            initial_rotation = torch.rad2deg(_cpu(task_env.initial_rotation_error))
            reset_progress = _cpu(task_env.reset_progress)
            reset_noise = _cpu(task_env.reset_noise_scale)
            reset_rotation = torch.rad2deg(_cpu(task_env.reset_rotation_command))
            reset_position = torch.linalg.norm(_cpu(task_env.reset_position_offset), dim=-1) * 1000.0
            reset_position_requested = _cpu(task_env.reset_position_requested) * 1000.0
            reset_object_yaw = torch.rad2deg(_cpu(task_env.reset_object_yaw_offset))
            reset_object_yaw_requested = torch.rad2deg(_cpu(task_env.reset_object_yaw_requested))
            best_position = initial_position.clone()
            best_rotation = initial_rotation.clone()
            final_position = initial_position.clone()
            final_rotation = initial_rotation.clone()
            final_completion_probability = torch.zeros(task_env.num_envs)
            termination = ["horizon"] * task_env.num_envs
            terminal_steps = torch.full((task_env.num_envs,), max_steps, dtype=torch.long)
            active = torch.ones(task_env.num_envs, dtype=torch.bool)

            # Open-loop baseline: remove the complete part-pose-induced target
            # displacement from the initial goal vector, then execute only the
            # nominal straight displacement. It never corrects translation or
            # orientation from observations after reset.
            initial_tcp_position, initial_tcp_quaternion, initial_position_error_w, _ = task_env._tcp_error()
            blind_displacement_w = initial_position_error_w + task_env.reset_goal_position_delta
            blind_distance_m = torch.linalg.norm(blind_displacement_w, dim=-1)
            blind_direction_w = blind_displacement_w / blind_distance_m.clamp_min(1.0e-9).unsqueeze(-1)
            rotation_w_camera = task_env._rotation_world_from_camera(initial_tcp_quaternion)
            blind_direction_camera = torch.bmm(
                rotation_w_camera.transpose(1, 2), blind_direction_w.unsqueeze(-1)
            ).squeeze(-1)

            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
            for step in range(1, max_steps + 1):
                if args_cli.controller == "blind_nominal":
                    actions = torch.zeros((task_env.num_envs, 7), dtype=torch.float32, device=task_env.device)
                    current_tcp_position = task_env.context.get_tcp_pose_w()[0]
                    traveled_m = torch.sum(
                        (current_tcp_position - initial_tcp_position) * blind_direction_w,
                        dim=-1,
                    )
                    moving = traveled_m < blind_distance_m
                    actions[moving, :3] = blind_direction_camera[moving]
                else:
                    with torch.inference_mode():
                        obs_t = agent.obs_to_torch(obs)
                        actions = agent.get_action(obs_t, is_deterministic=True)
                # Keep simulation stepping outside inference_mode. Isaac's
                # automatic reset writes into persistent asset buffers; if
                # those buffers are first materialized as inference tensors,
                # a later explicit repeat reset cannot update them in-place.
                obs, _, dones, infos = env.step(actions)
                evaluation = infos.get("evaluation")
                if evaluation is None:
                    raise RuntimeError("Environment did not export per-environment evaluation measurements.")
                position_mm = _cpu(evaluation["position_error_m"]) * 1000.0
                rotation_deg = torch.rad2deg(_cpu(evaluation["rotation_error_rad"]))
                done_cpu = _cpu(torch.as_tensor(dones)).bool()
                best_position[active] = torch.minimum(best_position[active], position_mm[active])
                best_rotation[active] = torch.minimum(best_rotation[active], rotation_deg[active])
                final_position[active] = position_mm[active]
                final_rotation[active] = rotation_deg[active]
                completion_probability = _cpu(evaluation["completion_probability"])
                final_completion_probability[active] = completion_probability[active]
                geometric_ready = _cpu(evaluation["geometric_ready"]).bool()
                completion_supervised = _cpu(evaluation["completion_supervised"]).bool()
                condition_completion_diagnostics.update(
                    completion_probability[active].tolist(),
                    geometric_ready[active].tolist(),
                    completion_supervised[active].tolist(),
                )
                completion_diagnostics_overall.update(
                    completion_probability[active].tolist(),
                    geometric_ready[active].tolist(),
                    completion_supervised[active].tolist(),
                )
                newly_done = active & done_cpu
                if newly_done.any():
                    strict_success = _cpu(evaluation["strict_success"]).bool()
                    operational_success = _cpu(evaluation["operational_success"]).bool()
                    premature = _cpu(evaluation["premature_completion"]).bool()
                    collision = _cpu(evaluation["collision"]).bool()
                    diverged = _cpu(evaluation["diverged"]).bool()
                    timed_out = _cpu(evaluation["timed_out"]).bool()
                    for env_index in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
                        if bool(strict_success[env_index]):
                            termination[env_index] = "success"
                        elif bool(operational_success[env_index]):
                            termination[env_index] = "operational_success"
                        elif bool(collision[env_index]):
                            termination[env_index] = "unsafe_collision"
                        elif bool(diverged[env_index]):
                            termination[env_index] = "diverged"
                        elif bool(premature[env_index]):
                            termination[env_index] = "premature_completion"
                        elif bool(timed_out[env_index]):
                            termination[env_index] = "timeout"
                        else:
                            termination[env_index] = "terminated"
                        terminal_steps[env_index] = step
                    active[newly_done] = False
                if agent.is_rnn and agent.states is not None and done_cpu.any():
                    dones_device = torch.as_tensor(dones).bool()
                    for state in agent.states:
                        state[:, dones_device, :] = 0.0
                if not active.any():
                    break

            for env_index in range(task_env.num_envs):
                target_index = int(target_indices[env_index])
                orientation_index = int(task_env.target_orientation_indices[target_index].item())
                part_index = int(task_env.target_part_indices[target_index].item())
                rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "target_index": target_index,
                        "target_id": task_env.target_ids[target_index],
                        "orientation_id": task_env.orientation_names[orientation_index],
                        "part_id": task_env.part_names[part_index],
                        "termination": termination[env_index],
                        "success": termination[env_index] == "success",
                        "operational_success": termination[env_index] in ("success", "operational_success"),
                        "steps": int(terminal_steps[env_index]),
                        "duration_s": float(terminal_steps[env_index]) * task_env.step_dt,
                        "reset_progress": float(reset_progress[env_index]),
                        "reset_noise_rad": float(reset_noise[env_index]),
                        "reset_rotation_command_deg": float(reset_rotation[env_index]),
                        "reset_position_offset_mm": float(reset_position[env_index]),
                        "reset_position_requested_mm": float(reset_position_requested[env_index]),
                        "reset_object_yaw_deg": float(reset_object_yaw[env_index]),
                        "reset_object_yaw_requested_deg": float(reset_object_yaw_requested[env_index]),
                        "initial_position_error_mm": float(initial_position[env_index]),
                        "initial_rotation_error_deg": float(initial_rotation[env_index]),
                        "final_position_error_mm": float(final_position[env_index]),
                        "final_rotation_error_deg": float(final_rotation[env_index]),
                        "final_completion_probability": float(final_completion_probability[env_index]),
                        "best_position_error_mm": float(best_position[env_index]),
                        "best_rotation_error_deg": float(best_rotation[env_index]),
                    }
                )
            completed = _summarize(rows[condition_start:])
            print(
                f"[EVAL] {condition} progress={progress_value:.2f} "
                f"noise={noise_rad:.3f} rad "
                f"position_offset={completed['reset_position_offset_mm_mean']:.2f} mm "
                f"object_yaw={completed['reset_object_yaw_deg_mean']:.2f} deg "
                f"repeat={repeat + 1}/{args_cli.runs_per_target} "
                f"strict={100.0 * completed['success_rate']:.1f}% "
                f"operational={100.0 * completed['operational_success_rate']:.1f}% "
                f"final={completed['final_position_error_mm_mean']:.2f} mm / "
                f"{completed['final_rotation_error_deg_mean']:.2f} deg",
                flush=True,
            )

    completion_summaries = {
        "overall": completion_diagnostics_overall.summary(),
        **{condition: diagnostics.summary() for condition, diagnostics in completion_diagnostics_by_condition.items()},
    }
    payload, target_rows, markdown = _build_report(
        checkpoint=checkpoint,
        episode_seconds=args_cli.episode_seconds,
        rotation_deg=args_cli.rotation_deg,
        catalog_split=str(env_cfg.catalog_split),
        rows=rows,
        completion_diagnostics=completion_summaries,
    )
    payload["seed"] = args_cli.seed
    payload["runs_per_target"] = args_cli.runs_per_target
    payload["nominal_noise_rad"] = args_cli.nominal_noise_rad
    payload["stress_noise_rad"] = args_cli.stress_noise_rad
    payload["mid_noise_rad"] = args_cli.mid_noise_rad
    payload["close_noise_rad"] = args_cli.close_noise_rad
    payload["rotation_deg"] = args_cli.rotation_deg
    payload["catalog_split"] = str(env_cfg.catalog_split)
    payload["sim2real_profile"] = sim2real_profile.name
    payload["sim2real_profile_id"] = sim2real_profile.identifier
    payload["sim2real_profile_description"] = sim2real_profile.description
    payload["sim2real_profile_overrides"] = dict(sim2real_profile.overrides)
    payload["camera_profile"] = D405_VISUAL_SERVO_CAMERA_PROFILE
    payload["observation_profile"] = D405_VISUAL_SERVO_OBSERVATION_PROFILE
    for row in rows:
        row["sim2real_profile"] = sim2real_profile.name
    markdown = f"Sim-to-real profile: `{sim2real_profile.identifier}`  \n{sim2real_profile.description}\n\n" + markdown
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    _write_csv(output_dir / "episodes.csv", rows)
    _write_csv(output_dir / "per_target.csv", target_rows)
    print(markdown, flush=True)
    print(f"[EVAL] Results written to: {output_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
