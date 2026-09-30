#!/usr/bin/env python3
"""Build independent object XY/yaw resets with unchanged canonical goal images."""

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
p.add_argument("--catalog", type=Path, default=ROOT / "isaac_rl/data/franka_clutter_v5_fast_fxaa/catalog.npz")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--variants", type=int, default=8)
p.add_argument("--max-targets-per-part", type=int, default=0)
p.add_argument("--seed", type=int, default=20260921)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import numpy as np
import torch
from grasp_planning.rl.franka_placement import (
    PLACEMENT_PROFILE,
    PandaUsdKinematics,
    sample_placement_deltas,
    transform_placement,
)
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg

from isaaclab.utils.math import compute_pose_error


def main():
    torch.set_num_threads(8)
    source = dict(np.load(args.catalog, allow_pickle=False))
    count = len(source["target_ids"])
    contract = json.loads(str(source["contract_json"].item()))
    assert not contract.get("placement_randomization")
    groups = [
        np.flatnonzero(source["target_part_indices"] == i).tolist() for i in range(len(contract["object_assets"]))
    ]
    if args.max_targets_per_part:
        groups = [g[: args.max_targets_per_part] for g in groups]
    allocated = np.array([int(bool(g)) for g in groups])
    assert args.num_envs >= allocated.sum()
    while allocated.sum() < args.num_envs:
        allocated[np.argmax([len(g) / (allocated[i] + 1) if g else 0 for i, g in enumerate(groups)])] += 1
    queues = []
    parts = []
    for part, g in enumerate(groups):
        for slot in range(allocated[part]):
            queues.append(g[slot :: allocated[part]] or [g[0]])
            parts.append(part)
    cfg = FrankaZedEnvCfg()
    cfg.build_catalog = True
    cfg.seed = args.seed
    cfg.scene.num_envs = len(queues)
    cfg.sim.device = args.device
    cfg.object_assets = contract["object_assets"]
    cfg.env_part_indices = parts
    cfg.robot_asset_manifest = str(ROOT / "assets/usd/franka_panda_offline/manifest.json")
    cfg.lab_asset_dir = contract["lab_scene"]["asset_dir"]
    cfg.lab_translation = tuple(contract["lab_scene"]["translation_m"])
    cfg.performance_profile = contract["performance_profile"]
    env = FrankaZedEnv(cfg)
    kin = PandaUsdKinematics(env)
    device = env.device
    n = env.num_envs
    limits = env.robot.data.soft_joint_pos_limits[:, env.arm_ids]
    q = env.robot.data.joint_pos[:, env.arm_ids]
    fk, jac = kin.forward(q)
    pp, qq = env.tcp_pose()
    pe, re = compute_pose_error(fk[:, :3] + env.scene.env_origins, fk[:, 3:], pp, qq, rot_error_type="axis_angle")
    assert pe.norm(dim=-1).max() < 0.0002 and re.norm(dim=-1).max() < 0.002, (
        "USD FK differs from PhysX",
        pe.norm(dim=-1).max(),
        re.norm(dim=-1).max(),
    )
    # PhysX's Jacobian cache is populated on the first physics step.
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    q = env.robot.data.joint_pos[:, env.arm_ids]
    fk, jac = kin.forward(q)
    # Verify point-Jacobian numerically. The existing controller's PhysX
    # Jacobian is intentionally not changed by this placement experiment.
    for joint in range(7):
        shifted = q.clone()
        shifted[:, joint] += 0.001
        shifted_pose, _ = kin.forward(shifted)
        dp, dr = compute_pose_error(
            fk[:, :3], fk[:, 3:], shifted_pose[:, :3], shifted_pose[:, 3:], rot_error_type="axis_angle"
        )
        assert (torch.cat((dp, dr), dim=1) / 0.001 - jac[:, :, joint]).abs().max() < 0.002, (
            "USD Jacobian finite difference mismatch"
        )
    print("[CONTROLLER JACOBIAN DIFFERENCE]", float((jac - env.tcp_jacobian()).abs().max()), flush=True)
    print("[FK VERIFIED]", float(pe.norm(dim=-1).max()), flush=True)
    S = source["pose_reset_valid"].shape[1]
    arrays = {
        k: np.zeros_like(source[k])
        for k in [
            "pose_reset_joints",
            "pose_reset_valid",
            "pose_reset_position_error_m",
            "pose_reset_rotation_error_rad",
            "pose_reset_contact_n",
            "pose_reset_lateral_m",
        ]
    }
    for key in ["placement_object_poses", "placement_goal_poses", "placement_start_poses"]:
        arrays[key] = np.zeros((count, S, 7), np.float32)
    arrays["placement_delta_xy_yaw"] = np.zeros((count, S, 3), np.float32)
    arrays["placement_source_bank"] = np.full((count, S), -1, np.int32)
    kinds = source["pose_reset_kind"]
    progress = source["pose_reset_progress"]
    spec = []
    for kind in [0, 1, 2, 3, 4, 5]:
        values = np.unique(progress[kinds == kind])
        values = values[[0, len(values) // 2, -2]] if kind == 5 else values
        for value in values:
            dest = np.flatnonzero((kinds == kind) & np.isclose(progress, value))
            spec.extend((int(si), kind, float(value)) for si in dest[: args.variants])
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    batches = max(map(len, queues))

    def step():
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    def measure(q, obj, width, desired):
        env.write_state(q, obj, open_width=width)
        env.scene.reset()
        env.sim.forward()
        peak = torch.zeros(n, device=device)
        for _ in range(12):
            step()
            peak = torch.maximum(peak, env.contact_force())
        pos, quat = env.tcp_pose()
        p, r = compute_pose_error(
            pos, quat, desired[:, :3] + env.scene.env_origins, desired[:, 3:], rot_error_type="axis_angle"
        )
        return (
            (p.norm(dim=-1) < 0.0008) & (r.norm(dim=-1) < 0.012) & (peak < 0.5),
            peak,
            pos - env.scene.env_origins,
            quat,
        )

    for batch in range(batches):
        ids = np.array([g[min(batch, len(g) - 1)] for g in queues])
        active = np.array([batch < len(g) for g in queues])

        def T(x):
            return torch.tensor(x, device=device, dtype=torch.float32)

        obj = T(source["object_poses"][ids])
        goal = T(source["goal_poses"][ids])
        width = T(source["open_widths"][ids])
        goal_q = T(source["joint_paths"][ids, -1])
        for si, kind, prog in spec:
            eligible = source["pose_reset_valid"][ids] & (kinds[None, :] == kind) & np.isclose(progress[None, :], prog)
            # Missing source states remain invalid, never replaced with another semantic pool.
            available = eligible.any(1)
            bank = np.array([rng.choice(np.flatnonzero(e)) if e.any() else 0 for e in eligible])
            base = T(source["pose_reset_joints"][ids, bank])
            start, _ = kin.forward(base)
            # Every target uses exactly the same absolute XY distribution.
            delta = T(sample_placement_deltas(rng, source["object_poses"][ids, :2]))
            moved_obj = transform_placement(obj, obj[:, :3], delta)
            moved_goal = transform_placement(goal, obj[:, :3], delta)
            desired = transform_placement(start, obj[:, :3], delta)
            solved = kin.solve(base, desired, limits)
            gq = kin.solve(goal_q, moved_goal, limits)
            gv, gforce, _, _ = measure(gq, moved_obj, width, moved_goal)
            ok, force, pos, quat = measure(solved, moved_obj, width, desired)
            p, r = compute_pose_error(pos, quat, moved_goal[:, :3], moved_goal[:, 3:], rot_error_type="axis_angle")
            pn, rn = p.norm(dim=-1), r.norm(dim=-1)
            ok &= gv & torch.tensor(available, device=device)
            if kind in [1, 4]:
                ok &= (pn < 0.004) & (rn < np.deg2rad(3))
            if kind == 2:
                ref = T(source["pose_reset_position_error_m"][ids, bank])
                ok &= torch.where(
                    ref < 0.004, pn < 0.004, torch.where(ref > 0.006, pn > 0.006, (pn > 0.004) & (pn < 0.006))
                ) & (rn < np.deg2rad(0.5))
            if kind == 3:
                ok &= (pn < 0.001) & (rn > np.deg2rad(4.5)) & (rn < np.deg2rad(5.5))
            values = {
                "pose_reset_joints": solved,
                "pose_reset_valid": ok,
                "pose_reset_position_error_m": pn,
                "pose_reset_rotation_error_rad": rn,
                "pose_reset_contact_n": torch.maximum(force, gforce),
                "pose_reset_lateral_m": T(source["pose_reset_lateral_m"][ids, bank]),
                "placement_object_poses": moved_obj,
                "placement_goal_poses": moved_goal,
                "placement_start_poses": torch.cat((pos, quat), dim=1),
                "placement_delta_xy_yaw": delta,
                "placement_source_bank": torch.tensor(bank, device=device),
            }
            for key, value in values.items():
                arrays[key][ids[active], si] = value.detach().cpu().numpy()[active]
        print(
            "[PLACEMENT]",
            batch + 1,
            "/",
            batches,
            "valid",
            int(arrays["pose_reset_valid"][ids[active]].sum()),
            "elapsed",
            round(time.monotonic() - started, 1),
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.parent / "partial_placement_resets.npz", **arrays)
    valid = arrays["pose_reset_valid"]
    processed = np.array(sorted(set(i for g in queues for i in g)))
    keep = np.zeros(count, bool)
    keep[processed] = True
    for kind in [0, 3, 5]:
        keep &= (valid & (kinds[None, :] == kind)).any(1)
    keep &= (valid & np.isin(kinds[None, :], [1, 4])).any(1)
    for prog in [0, 0.5, 0.94]:
        keep &= (valid & (kinds[None, :] == 0) & np.isclose(progress[None, :], prog)).any(1)
    report = {
        "source_catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "processed": len(processed),
        "total": count,
        "retained": int(keep.sum()),
        "excluded": [str(source["target_ids"][i]) for i in processed if not keep[i]],
        "valid_states": int(valid[keep].sum()),
        "accepted_states_before_target_filter": int(valid.sum()),
        "attempted_states": len(processed) * len(spec),
        "elapsed_s": time.monotonic() - started,
        "profile": PLACEMENT_PROFILE,
        "all_parts_retained": set(source["part_keys"][keep]) == set(source["part_keys"][processed]),
        "canonical_goal_images_unchanged": True,
        "accepted_delta_min": arrays["placement_delta_xy_yaw"][valid & keep[:, None]].min(axis=0).tolist()
        if keep.any()
        else [],
        "accepted_delta_max": arrays["placement_delta_xy_yaw"][valid & keep[:, None]].max(axis=0).tolist()
        if keep.any()
        else [],
        "parts": {
            str(part): {
                "source_targets": int((source["part_keys"] == part).sum()),
                "retained_targets": int(((source["part_keys"] == part) & keep).sum()),
            }
            for part in np.unique(source["part_keys"])
        },
    }
    assert keep.any(), report
    canonical_goal_hash = hashlib.sha256(source["goal_rgbd"][keep].tobytes()).hexdigest()
    source.update(arrays)
    # Explicit row arrays: reset metadata is bank-shaped and must not be accidentally sliced.
    for key, value in list(source.items()):
        if (
            value.ndim
            and value.shape[0] == count
            and key not in ["pose_reset_kind", "pose_reset_progress", "pose_reset_variant"]
        ):
            source[key] = value[keep]
    contract["placement_randomization"] = PLACEMENT_PROFILE
    source["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
    assert canonical_goal_hash == hashlib.sha256(source["goal_rgbd"].tobytes()).hexdigest()
    report["canonical_goal_sha256"] = canonical_goal_hash
    np.savez_compressed(args.output, **source)
    (args.output.with_suffix(".placement.json")).write_text(json.dumps(report, indent=2))
    print("[PLACEMENT COMPLETE]", json.dumps(report), flush=True)
    env.close()


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
