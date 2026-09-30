#!/usr/bin/env python3
"""Build measured, collision-tested Panda resets around the Clutter-v5 pose distribution."""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "isaac_rl/source/isaac_rl")]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--catalog", type=Path, default=ROOT / "isaac_rl/data/franka_fabrica_all_complete/catalog.npz")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--max-targets", type=int, default=0)
p.add_argument("--recipe", type=Path, default=ROOT / "configs/franka_clutter_v5_replica.json")
p.add_argument(
    "--reuse-catalog", type=Path, help="Reuse unchanged validated path/ready states; rebuild boundary states"
)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import numpy as np
import torch
from grasp_planning.rl.zed_mini import damped_joint_velocity
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg

from isaaclab.utils.math import compute_pose_error, quat_from_angle_axis, quat_mul


def main():
    data = dict(np.load(args.catalog, allow_pickle=False))
    count = len(data["target_ids"])
    contract = json.loads(str(data["contract_json"].item()))
    groups = [np.flatnonzero(data["target_part_indices"] == i).tolist() for i in range(len(contract["object_assets"]))]
    if args.max_targets:
        chosen = np.array([g[0] for g in groups if g][: args.max_targets])
        groups = [[i for i in g if i in chosen] for g in groups]
    allocated = np.array([int(bool(g)) for g in groups])
    while allocated.sum() < args.num_envs:
        priority = np.array([len(g) / (allocated[i] + 1) if g else 0 for i, g in enumerate(groups)])
        allocated[priority.argmax()] += 1
    queues = []
    assignment = []
    for part, g in enumerate(groups):
        for slot in range(allocated[part]):
            queues.append(g[slot :: allocated[part]] or [g[0]])
            assignment.append(part)
    cfg = FrankaZedEnvCfg()
    cfg.build_catalog = True
    cfg.seed = 43
    cfg.scene.num_envs = len(queues)
    cfg.sim.device = args.device
    cfg.object_assets = contract["object_assets"]
    cfg.env_part_indices = assignment
    cfg.robot_asset_manifest = str(ROOT / "assets/usd/franka_panda_offline/manifest.json")
    cfg.lab_asset_dir = contract["lab_scene"]["asset_dir"]
    cfg.lab_translation = tuple(contract["lab_scene"]["translation_m"])
    cfg.lab_props = contract["lab_scene"]["props"]
    env = FrankaZedEnv(cfg)
    device = env.device
    n = env.num_envs
    limits = env.robot.data.soft_joint_pos_limits[:, env.arm_ids]
    nominal_progress = np.linspace(0, 1, data["joint_paths"].shape[1]).tolist()
    progress = np.array([0, 0.25, 0.5, 0.75, 0.94, 0.985, 1.0, 1.0, 1.0, 1.0] + nominal_progress, dtype=np.float32)
    kinds = np.repeat([0, 0, 0, 0, 0, 2, 1, 3, 4, 1] + [5] * len(nominal_progress), 8)
    variants = 8
    S = len(progress) * variants
    arrays = {
        "pose_reset_joints": np.zeros((count, S, 7), np.float32),
        "pose_reset_valid": np.zeros((count, S), bool),
        "pose_reset_position_error_m": np.zeros((count, S), np.float32),
        "pose_reset_rotation_error_rad": np.zeros((count, S), np.float32),
        "pose_reset_lateral_m": np.zeros((count, S), np.float32),
        "pose_reset_contact_n": np.zeros((count, S), np.float32),
    }
    reuse_version = 0
    if args.reuse_catalog:
        previous = dict(np.load(args.reuse_catalog, allow_pickle=False))
        previous_contract = json.loads(str(previous["contract_json"].item()))
        reuse_version = previous_contract["pose_reset_profile"]["version"]
        assert reuse_version in (1, 2), "Unsupported reuse layout"
        assert (
            previous_contract["pose_reset_profile"]["source_catalog_sha256"]
            == hashlib.sha256(args.catalog.read_bytes()).hexdigest()
        )
        lookup = {str(value): i for i, value in enumerate(data["target_ids"])}
        rows = np.array([lookup[str(value)] for value in previous["target_ids"]])
        for key in arrays:
            if reuse_version == 2:
                arrays[key][rows, :80] = previous[key][:, :80]
            else:
                arrays[key][rows, :40] = previous[key][:, :40]
                arrays[key][rows, 48:56] = previous[key][:, 48:56]
                arrays[key][rows, 72:80] = previous[key][:, 40:48]
    axes = torch.tensor(
        [[x, y, z] for x in [-1.0, 1.0] for y in [-1.0, 1.0] for z in [-1.0, 1.0]], device=device
    ) / np.sqrt(3)

    def step():
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    rng = np.random.default_rng(43)
    for batch in range(max(map(len, queues))):
        ids = np.array([q[min(batch, len(q) - 1)] for q in queues])
        active = np.array([batch < len(q) for q in queues])

        def tensor(key):
            return torch.tensor(data[key][ids], device=device, dtype=torch.float32)

        goal = tensor("goal_poses")
        obj = tensor("object_poses")
        width = tensor("open_widths")
        paths = tensor("joint_paths")
        # Derive the validated approach axis from exact FK, not a gripper convention guess.
        env.write_state(paths[:, 0], obj, open_width=width)
        step()
        start = env.tcp_pose()[0] - env.scene.env_origins
        direction = goal[:, :3] - start
        direction /= direction.norm(dim=-1, keepdim=True)
        for pi, prog in enumerate(progress):
            if (reuse_version == 2 and pi < 10) or (reuse_version == 1 and (pi < 5 or pi in (6, 9))):
                continue
            base = paths[:, round(float(prog) * (paths.shape[1] - 1))]
            for vi in range(variants):
                if (pi == 8 or pi >= 10) and vi > 0:
                    continue  # One exact-goal state per target is sufficient.
                angle = (
                    (5 + 10 * (1 - float(prog)) ** 1.5)
                    if pi < 5
                    else {5: 0.0, 6: 1.5, 7: 5.0, 8: 0.0, 9: 2.5}.get(pi, 0.0)
                )
                theta = torch.full((n,), np.deg2rad(angle), device=device)
                desired = goal.clone()
                distance = (
                    [0.0035, 0.0045, 0.0065, 0.0035, 0.0045, 0.0065, 0.0045, 0.0065][vi]
                    if pi == 5
                    else 0.10 * (1 - float(prog))
                )
                if pi == 9:
                    distance = 0.0015
                desired[:, :3] -= direction * distance
                lateral_max = (
                    (0.003 + 0.007 * (1 - float(prog)) ** 1.5) if pi < 5 else {6: 0.001, 9: 0.0015}.get(pi, 0.0)
                )
                azimuth = torch.tensor(rng.uniform(-np.pi, np.pi, n), device=device, dtype=torch.float32)
                radius = torch.tensor(rng.uniform(0.5, 1.0, n) * lateral_max, device=device, dtype=torch.float32)
                desired[:, 0] += radius * torch.cos(azimuth)
                desired[:, 1] += radius * torch.sin(azimuth)
                desired[:, 3:] = quat_mul(quat_from_angle_axis(theta, axes[vi].expand(n, 3)), goal[:, 3:])
                q = base.clone()
                for iteration in range(0 if pi >= 10 else 45):
                    env.write_state(q, obj, open_width=width)
                    step()
                    pe, re = compute_pose_error(
                        *env.tcp_pose(),
                        desired[:, :3] + env.scene.env_origins,
                        desired[:, 3:],
                        rot_error_type="axis_angle",
                    )
                    dq = damped_joint_velocity(env.tcp_jacobian(), torch.cat((pe, re), -1), 0.02)
                    q = (q + dq.clamp(-0.08, 0.08)).clamp(limits[..., 0] + 0.003, limits[..., 1] - 0.003)
                    if iteration >= 8 and bool(((pe.norm(dim=-1) < 0.00025) & (re.norm(dim=-1) < 0.003)).all()):
                        break
                env.write_state(q, obj, open_width=width)
                env.scene.reset()
                env.sim.forward()
                peak = torch.zeros(n, device=device)
                for _ in range(12):
                    step()
                    peak = torch.maximum(peak, env.contact_force())
                pos, rot = env.tcp_pose()
                pe, re = compute_pose_error(
                    pos, rot, desired[:, :3] + env.scene.env_origins, desired[:, 3:], rot_error_type="axis_angle"
                )
                gp, gr = compute_pose_error(
                    pos, rot, goal[:, :3] + env.scene.env_origins, goal[:, 3:], rot_error_type="axis_angle"
                )
                valid = (pe.norm(dim=-1) < 0.001) & (re.norm(dim=-1) < 0.015) & (peak < 0.5) & torch.isfinite(q).all(-1)
                if pi == 5:
                    pn, rn = gp.norm(dim=-1), gr.norm(dim=-1)
                    label = (
                        (pn < 0.004)
                        if distance < 0.004
                        else ((pn > 0.006) if distance > 0.006 else ((pn > 0.004) & (pn < 0.006)))
                    )
                    valid &= label & (rn < np.deg2rad(0.5))
                elif pi == 7:
                    valid &= (
                        (gp.norm(dim=-1) < 0.001)
                        & (gr.norm(dim=-1) > np.deg2rad(4.5))
                        & (gr.norm(dim=-1) < np.deg2rad(5.5))
                    )
                elif pi in (6, 8, 9):
                    valid &= (gp.norm(dim=-1) < 0.004) & (gr.norm(dim=-1) < np.deg2rad(3))
                elif pi >= 10:
                    valid = (
                        (peak < 0.5)
                        & torch.isfinite(q).all(-1)
                        & (q >= limits[..., 0]).all(-1)
                        & (q <= limits[..., 1]).all(-1)
                    )
                si = pi * variants + vi
                for key, value in [
                    ("pose_reset_joints", q),
                    ("pose_reset_valid", valid),
                    ("pose_reset_position_error_m", gp.norm(dim=-1)),
                    ("pose_reset_rotation_error_rad", gr.norm(dim=-1)),
                    ("pose_reset_lateral_m", radius),
                    ("pose_reset_contact_n", peak),
                ]:
                    arrays[key][ids[active], si] = value.detach().cpu().numpy()[active]
            print(
                f"[POSE BANK] batch={batch + 1}/{max(map(len, queues))} progress={prog:.3f} "
                f"valid={arrays['pose_reset_valid'][ids[active], pi * variants : (pi + 1) * variants].sum()}",
                flush=True,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.parent / "partial_resets.npz", **arrays)
    valid = arrays["pose_reset_valid"]
    processed = sorted(set(i for q in queues for i in q))
    keep = (valid & np.isin(kinds, [1, 4])[None, :]).any(1)
    missing = [str(data["target_ids"][i]) for i in processed if keep[i] and not valid[i, :40].any()]
    excluded = data["target_ids"][~keep].tolist()
    if not args.max_targets:
        assert set(data["part_keys"][keep]) == set(data["part_keys"]), "Filtering would remove an entire part"
    valid = valid[keep]
    report = {
        "source_catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "processed_targets": len(processed),
        "total_targets": count,
        "retained_targets": int(keep.sum()),
        "excluded_no_safe_ready_state": excluded,
        "valid_states": int(valid.sum()),
        "missing_perturbed_targets": missing,
        "by_progress": [
            {
                "progress": float(t),
                "kind": int(kinds[i * 8]),
                "valid": int(valid[:, i * 8 : (i + 1) * 8].sum()),
                "targets": int(valid[:, i * 8 : (i + 1) * 8].any(1).sum()),
            }
            for i, t in enumerate(progress)
        ],
        "passed": not missing and len(processed) == count,
        "validation": (
            "GPU IK position<1mm/rotation<0.86deg; joint limits; "
            "12 physics steps with peak arm/hand/finger contact<0.5N. No interpolation of reset states."
        ),
    }
    (args.output.parent / "reset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        raise RuntimeError(f"No safe perturbed starts for {len(missing)} targets; see audit")
    data.update(arrays)
    data = {key: value[keep] if value.ndim > 0 and value.shape[0] == count else value for key, value in data.items()}
    data["pose_reset_progress"] = np.repeat(progress, variants)
    data["pose_reset_variant"] = np.tile(np.arange(variants), len(progress))
    data["pose_reset_kind"] = kinds
    contract["training_recipe"] = json.loads(args.recipe.read_text())
    contract["pose_reset_profile"] = {
        "version": 3,
        "variants": 8,
        "progress": progress.tolist(),
        "far_distance_m": 0.10,
        "rotation_far_deg": 15.0,
        "rotation_near_deg": 5.0,
        "lateral_far_m": 0.01,
        "lateral_near_m": 0.003,
        "minimum_fraction": 0.5,
        "source_catalog_sha256": report["source_catalog_sha256"],
        "boundary_position_distances_m": [0.0035, 0.0045, 0.0065],
        "boundary_rotation_deg": 5.0,
        "boundary_rotation_fallback": "validated 0.94 path state only when all goal rotation variants are infeasible",
        "boundary_position_fallback": (
            "validated 0.94 path state only when position-only boundary states are infeasible"
        ),
        "nominal_path_physics_revalidated": True,
        "excluded_no_safe_ready_state": excluded,
        "reused_catalog_sha256": hashlib.sha256(args.reuse_catalog.read_bytes()).hexdigest()
        if args.reuse_catalog
        else None,
    }
    data["contract_json"] = np.array(json.dumps(contract, sort_keys=True))
    np.savez_compressed(args.output, **data)
    print("[POSE BANK] COMPLETE", json.dumps(report), flush=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
