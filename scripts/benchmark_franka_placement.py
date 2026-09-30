#!/usr/bin/env python3
"""Paired canonical/new placement evaluation; frozen checkpoint, no learning.

The ONLY permitted contract transfer is the independent-placement profile. Both
conditions use identical saved goal pixels, source grasp, perturbation and light
seed. Simulator labels remain available for scoring, and actor invariance to
those labels is asserted. This is conditional on offline reachable placements.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "isaac_rl/source/isaac_rl")]
p = argparse.ArgumentParser()
p.add_argument("--catalog", type=Path, required=True)
p.add_argument("--source-catalog", type=Path, default=ROOT / "isaac_rl/data/franka_clutter_v5_fast_fxaa/catalog.npz")
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--agent", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--split", choices=["test", "validation", "all"], default="test")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
p.add_argument(
    "--symmetry-evaluation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Score against validated finite and continuous axial object symmetries; retain nominal diagnostics",
)
p.add_argument(
    "--record-videos", action="store_true", help="Record six randomized episodes simultaneously per distance at 15 fps"
)
p.add_argument(
    "--reference-results",
    type=Path,
    help="For the oracle check, reuse exact targets/reset banks from this policy result",
)
p.add_argument(
    "--oracle-check",
    action="store_true",
    help="Separate privileged near-start controller sanity check, never a policy score",
)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import numpy as np
import torch
import yaml
from grasp_planning.rl.franka_placement import placement_transfer_allowed
from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import register_grasp_completion_runner
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg
from PIL import Image, ImageDraw, ImageFont
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


def main():
    torch.set_num_threads(8)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.catalog) as data:
        selected = (
            np.arange(len(data["target_ids"])) if args.split == "all" else np.flatnonzero(data["split"] == args.split)
        )
        ids = data["target_ids"][selected]
        parts = data["part_keys"][selected]
        splits = data["split"][selected]
        source_banks = data["placement_source_bank"][selected]
        deltas = data["placement_delta_xy_yaw"][selected]
        goal_hash = hashlib.sha256(data["goal_rgbd"][selected].tobytes()).hexdigest()
    with np.load(args.source_catalog) as source:
        lookup = {str(v): i for i, v in enumerate(source["target_ids"])}
        rows = np.array([lookup[str(v)] for v in ids])
        assert goal_hash == hashlib.sha256(source["goal_rgbd"][rows].tobytes()).hexdigest(), "Goal images changed"
        canonical_joints = source["pose_reset_joints"][rows[:, None], np.maximum(source_banks, 0)]
        source_contract = json.loads(source["contract_json"].item())
    checkpoint_contract = json.loads(args.checkpoint.with_suffix(".contract.json").read_text())
    assert source_contract == checkpoint_contract, "Source catalog does not match checkpoint"
    n = len(set(parts))
    assert n >= 1
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.num_envs = n
    cfg.catalog_path = str(args.catalog)
    cfg.catalog_split = args.split
    cfg.robot_asset_manifest = str(ROOT / "assets/usd/franka_panda_offline/manifest.json")
    cfg.rgb_gain_randomization = 0.0
    cfg.reset_ready_fraction = 0.0
    cfg.fixed_waypoint_index = 0
    cfg.sequential_target_assignment = True
    cfg.pose_evaluation = True
    cfg.symmetry_evaluation = args.symmetry_evaluation
    num_views = 0 if args.oracle_check else min(6, n)
    for i in range(num_views):
        setattr(
            cfg.scene,
            f"inspect{i}",
            CameraCfg(
                prim_path=f"/World/Inspect{i}",
                width=640,
                height=400,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 15.0)),
            ),
        )
    env = FrankaZedEnv(cfg)
    if env.symmetry_report is not None:
        (out / "symmetry_evaluation.json").write_text(json.dumps(env.symmetry_report, indent=2) + "\n")
    cameras = [env.scene.sensors.pop(f"inspect{i}") for i in range(num_views)]
    assert placement_transfer_allowed(checkpoint_contract, env.contract()), "Unsupported checkpoint transfer"
    wrapped = RlGamesVecEnvWrapper(env, args.device, 5.0, 1.0)
    vecenv.register("IsaacRlgWrapper", lambda name, actors, **kw: RlGamesGpuEnv(name, actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    config = yaml.safe_load(args.agent.read_text())
    config["params"]["config"].update(device=args.device, device_name=args.device, num_actors=n, multi_gpu=False)
    runner = Runner()
    register_grasp_completion_runner(runner)
    runner.load(config)
    runner.reset()
    player = runner.create_player()
    player.restore(str(args.checkpoint))
    tables = env.part_target_table.clone()
    counts = env.part_target_counts.clone()
    valid = env.pose_reset_valid.clone()
    moved_joints = env.pose_reset_joints.clone()
    canonical_joints = torch.as_tensor(canonical_joints, device=env.device)
    moved_obj = env.placement_object_poses
    moved_goal = env.placement_goal_poses
    # Exercise the normal curriculum reset pools too, without learning.
    mode_counts = [0, 0, 0, 0]
    env.cfg.pose_evaluation = False
    for _ in range(12):
        wrapped.reset()
        t, b = env.target_index, env.reset_bank_index
        assert torch.equal(env.goal_rgbd, env.catalog["goal_rgbd"][t])
        assert torch.allclose(env.part.data.root_pos_w - env.scene.env_origins, moved_obj[t, b, :3], atol=2e-5)
        assert torch.allclose(env.goal_pose[:, :3] - env.scene.env_origins, moved_goal[t, b, :3], atol=2e-5)
        for mode in range(4):
            mode_counts[mode] += int((env.reset_mode == mode).sum())
    env.cfg.pose_evaluation = True
    print("[TRAINING RESET POOLS VERIFIED]", mode_counts, flush=True)
    target_parts = [str(parts[int(tables[i, 0])]) for i in range(n)]
    preferred = [
        "plumbers_block__part_3",
        "plumbers_block__part_2",
        "plumbers_block__part_4",
        "gamepad__part_2",
        "duct__part_2",
        "stool_circular__part_0",
    ]
    slots = [target_parts.index(v) for v in preferred if v in target_parts]
    slots = (slots + [i for i in range(n) if i not in slots])[:num_views]
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    records = []
    images = []
    videos = []
    started = time.monotonic()
    reference = json.loads(args.reference_results.read_text())["records"] if args.reference_results else None
    if reference is not None and not args.oracle_check:
        raise ValueError("Reference replay is only for the separately scored controller sanity check")
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        chosen = [int(tables[i, int(rng.integers(int(counts[i])))]) for i in range(n)]
        for progress in [0.94] if args.oracle_check else [0.0, 0.5, 0.94]:
            banks = [
                int(
                    rng.choice(
                        torch.where(
                            valid[t] & (env.pose_reset_kind == 0) & ((env.pose_reset_progress - progress).abs() < 1e-5)
                        )[0]
                        .cpu()
                        .numpy()
                    )
                )
                for t in chosen
            ]
            if reference is not None:
                matching = {
                    r["part_key"]: r
                    for r in reference
                    if r["seed"] == seed and r["progress"] == progress and r["condition"] == "canonical"
                }
                chosen = [env.target_ids.index(matching[part]["target_id"]) for part in target_parts]
                banks = [matching[part]["reset_bank"] for part in target_parts]
            initial_ref = None
            for condition in ["canonical", "randomized"]:
                env.cfg.pose_evaluation_progress = progress
                env.pose_reset_joints = canonical_joints if condition == "canonical" else moved_joints
                env.placement_object_poses = None if condition == "canonical" else moved_obj
                env.placement_goal_poses = None if condition == "canonical" else moved_goal
                env.pose_reset_valid.zero_()
                for i, (t, b) in enumerate(zip(chosen, banks)):
                    env.part_target_table[i, :] = t
                    env.part_target_counts[i] = 1
                    env.pose_reset_valid[t, b] = True
                env.part_target_cursor.zero_()
                env.seed(seed)
                result = wrapped.reset()
                obs = result["obs"] if isinstance(result, dict) else result
                player.get_batch_size(obs, 1)
                if player.is_rnn:
                    player.init_rnn()
                pe, re = env.pose_errors()
                initial = torch.stack((pe.norm(dim=1), re.norm(dim=1)), dim=1).cpu().numpy()
                if initial_ref is None:
                    initial_ref = initial.copy()
                else:
                    diff = np.max(np.abs(initial - initial_ref), axis=0)
                    assert diff[0] < 0.0015 and diff[1] < 0.025, ("Unmatched relative starting errors", diff)
                with torch.no_grad():
                    action = player.get_action(obs, is_deterministic=True)
                    hidden = obs.clone()
                    hidden[:, -8:] = 0
                    assert torch.equal(action, player.get_action(hidden, is_deterministic=True)), (
                        "Privileged labels affect actor"
                    )
                current = [
                    dict(
                        condition=condition,
                        controller="privileged_oracle" if args.oracle_check else "frozen_policy",
                        seed=seed,
                        progress=progress,
                        target_id=str(ids[t]),
                        part_key=target_parts[i],
                        split=str(splits[t]),
                        reset_bank=b,
                        source_bank=int(source_banks[t, b]),
                        placement_delta=([0.0, 0.0, 0.0] if condition == "canonical" else deltas[t, b].tolist()),
                        initial_position_mm=float(initial[i, 0] * 1000),
                        initial_rotation_deg=float(np.rad2deg(initial[i, 1])),
                        ever_geometric_ready=False,
                        trajectory=[],
                    )
                    for i, (t, b) in enumerate(zip(chosen, banks))
                ]
                # First seed: actual world, actual wrist and unchanged canonical RGB-D reference.
                if seed == args.seeds[0] and not args.oracle_check:
                    for k, slot in enumerate(slots):
                        origin = env.scene.env_origins[slot : slot + 1]
                        cameras[k].set_world_poses_from_view(
                            origin + torch.tensor([[1.25, -1.05, 0.95]], device=env.device),
                            origin + torch.tensor([[0.46, 0.0, 0.12]], device=env.device),
                        )
                    for _ in range(3):
                        env.sim.render()
                    for k, slot in enumerate(slots):
                        cameras[k].update(env.step_dt, force_recompute=True)
                        world = cameras[k].data.output["rgb"][0, ..., :3].cpu().numpy()
                        wrist = env.wrist_camera.data.output["rgb"][slot, ..., :3].cpu().numpy()
                        goal = (env.goal_rgbd[slot, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                        assert world.std() > 2 and wrist.std() > 2
                        canvas = Image.new("RGB", (1152, 640), "#101a25")
                        draw = ImageDraw.Draw(canvas)
                        draw.text(
                            (16, 12),
                            f"{condition.upper()} | {target_parts[slot]} | progress {progress:g}",
                            font=font,
                            fill="white",
                        )
                        draw.text(
                            (16, 40),
                            f"15 Hz | dx,dy,yaw: {current[slot]['placement_delta']}",
                            font=font,
                            fill="#bcd5e8",
                        )
                        canvas.paste(Image.fromarray(world).resize((768, 480)), (0, 94))
                        for y, title, img in [(116, "Live wrist", wrist), (354, "UNCHANGED canonical goal", goal)]:
                            draw.text((778, y - 25), title, font=font, fill="white")
                            canvas.paste(Image.fromarray(img).resize((368, 207)), (778, y))
                        draw.text(
                            (16, 592),
                            "Robot base / table fixed. Same source perturbation and light seed in paired images.",
                            font=font,
                            fill="white",
                        )
                        name = f"{condition}_{progress}_{target_parts[slot]}.png"
                        canvas.save(out / name)
                        images.append(dict(path=name, condition=condition, progress=progress, part=target_parts[slot]))
                writers = {}
                if args.record_videos and condition == "randomized" and seed == args.seeds[0] and not args.oracle_check:
                    from grasp_planning.rl.franka_episode_video import EpisodeVideo

                    for slot in slots:
                        goal = (env.goal_rgbd[slot, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                        path = out / f"randomized_{progress}_{target_parts[slot]}.mp4"
                        writers[slot] = EpisodeVideo(
                            path, target_parts[slot], progress, current[slot]["placement_delta"], goal
                        )
                active = set(range(n))
                for step in range(env.max_episode_length + 2):
                    with torch.no_grad():
                        if args.oracle_check:
                            p_error, r_error = env.pose_errors()
                            inverse_camera = env.camera_rotation().transpose(1, 2)
                            pc = (inverse_camera @ p_error[..., None]).squeeze(-1)
                            rc = (inverse_camera @ r_error[..., None]).squeeze(-1)
                            ready = (p_error.norm(dim=-1) < 0.002) & (r_error.norm(dim=-1) < np.deg2rad(1))
                            action = torch.cat(
                                (
                                    2 * pc / env.cfg.linear_action_scale_m_s,
                                    2 * rc / env.cfg.angular_action_scale_rad_s,
                                    ready[:, None].float(),
                                ),
                                dim=-1,
                            ).clamp(-1, 1)
                            action[ready, :6] = 0
                        else:
                            action = player.get_action(obs, is_deterministic=True)
                    assert torch.isfinite(action).all()
                    pe, re = env.evaluation_error_norms()
                    pn = pe.cpu().numpy()
                    rn = re.cpu().numpy()
                    for k, slot in enumerate(slots):
                        if slot in active and slot in writers:
                            cameras[k].update(env.step_dt, force_recompute=True)
                            world = cameras[k].data.output["rgb"][0, ..., :3].cpu().numpy()
                            wrist = (
                                obs[slot, : 72 * 128 * 8].view(72, 128, 8)[..., :3].clamp(0, 1).cpu().numpy() * 255
                            ).astype("uint8")
                            writers[slot].frame(
                                world,
                                wrist,
                                step,
                                float(pn[slot] * 1000),
                                float(np.rad2deg(rn[slot])),
                                float(action[slot, 6]),
                            )
                    for slot in active:
                        current[slot]["ever_geometric_ready"] |= bool(pn[slot] < 0.004 and rn[slot] < np.deg2rad(3))
                        current[slot]["trajectory"].append(
                            [step / 15, float(pn[slot] * 1000), float(np.rad2deg(rn[slot])), float(action[slot, 6])]
                        )
                    with torch.no_grad():
                        result, _, done, _ = wrapped.step(action)
                    obs = result["obs"] if isinstance(result, dict) else result
                    for slot in list(active):
                        if not bool(done[slot]):
                            continue
                        terminal = {
                            key: (bool(value[slot]) if value.dtype == torch.bool else float(value[slot]))
                            for key, value in env.last_transition.items()
                        }
                        current[slot].update(
                            terminal=terminal,
                            duration_s=(step + 1) / 15,
                            outcome=next(
                                key
                                for key in ["success", "collision", "premature", "divergence", "timeout"]
                                if terminal[key]
                            ),
                        )
                        current[slot]["ever_geometric_ready"] |= bool(
                            terminal["position_error_m"] < 0.004 and terminal["rotation_error_rad"] < np.deg2rad(3)
                        )
                        records.append(current[slot])
                        if slot in writers:
                            clip = writers.pop(slot).finish(current[slot]["outcome"], terminal)
                            clip.update(
                                seed=seed,
                                target_id=current[slot]["target_id"],
                                reset_bank=current[slot]["reset_bank"],
                                controller="frozen_policy",
                                checkpoint=str(args.checkpoint),
                            )
                            videos.append(clip)
                        active.remove(slot)
                    if not active:
                        break
                assert not active, active
                report = dict(
                    records=records,
                    images=images,
                    videos=videos,
                    elapsed_s=time.monotonic() - started,
                    checkpoint=str(args.checkpoint),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    canonical_goal_sha256=goal_hash,
                    actor_label_invariance_verified=True,
                    controller="privileged_oracle" if args.oracle_check else "frozen_policy",
                    reference_results=str(args.reference_results) if args.reference_results else None,
                    training_reset_mode_counts=mode_counts,
                    split=args.split,
                    placement_profile=env.cfg.placement_randomization,
                )
                (out / "results.json").write_text(json.dumps(report, indent=2))
                print(
                    "[BATCH]",
                    condition,
                    seed,
                    progress,
                    {
                        key: sum(x["outcome"] == key for x in current)
                        for key in ["success", "collision", "premature", "divergence", "timeout"]
                    },
                    "elapsed",
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
    env.close()
    print("[BENCHMARK COMPLETE]", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    sys.stdout.flush()
    os._exit(0)
