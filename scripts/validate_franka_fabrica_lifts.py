#!/usr/bin/env python3
"""Physical Panda closure/lift check; no object fixture or pose writes during pickup."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--catalog", type=Path, default=ROOT / "isaac_rl/data/franka_fabrica_plumbers/catalog.npz")
parser.add_argument("--camera-profile", type=Path, default=ROOT / "configs/franka_zed_mini.json")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/franka_fabrica/lifts.json")
parser.add_argument(
    "--filtered-catalog", type=Path, default=ROOT / "isaac_rl/data/franka_fabrica_plumbers/catalog_lift_validated.npz"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import numpy as np
import torch
from grasp_planning.rl.zed_mini import damped_joint_velocity
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg
from PIL import Image

from isaaclab.utils.math import compute_pose_error


def main():
    with np.load(args.catalog, allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.catalog_path = str(args.catalog)
    cfg.camera_profile_path = str(args.camera_profile)
    cfg.catalog_split = "all"
    cfg.dynamic_object = True
    cfg.scene.num_envs = min(args.batch_size, len(data["target_ids"]))
    env = FrankaZedEnv(cfg)
    n = env.num_envs
    env.robot.write_joint_effort_limit_to_sim(torch.full((n, 2), 20.0, device=env.device), joint_ids=env.finger_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def step():
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    def bilateral():
        forces = []
        for name in ("left_finger_contact", "right_finger_contact"):
            f = env.scene[name].data.force_matrix_w
            forces.append(torch.linalg.vector_norm(f, dim=-1).flatten(1).amax(1))
        return torch.minimum(*forces)

    for start in range(0, len(data["target_ids"]), n):
        count = min(n, len(data["target_ids"]) - start)
        indices = torch.tensor([start + i for i in range(count)] + [start] * (n - count), device=env.device)
        q = env.catalog["joint_paths"][indices, -1]
        obj = env.catalog["object_poses"][indices]
        goal = env.catalog["goal_poses"][indices].clone()
        goal[:, :3] += env.scene.env_origins
        width = env.catalog["open_widths"][indices]
        jaw = env.catalog["jaw_widths"][indices]
        env.write_state(q, obj, open_width=width)
        env.scene.reset()
        for _ in range(36):
            step()
        initial = env.part.data.root_pos_w.clone()
        before_close = (initial - (obj[:, :3] + env.scene.env_origins)).norm(dim=-1)
        closed = (jaw - 0.012).clamp_min(0.0)
        for progress in np.linspace(0, 1, 120):
            finger = (width + (closed - width) * float(progress))[:, None].expand(-1, 2) / 2
            env.robot.set_joint_position_target(finger, joint_ids=env.finger_ids)
            step()
        for _ in range(120):
            step()
        close_contact = bilateral().clone()
        close_width = env.robot.data.joint_pos[:, env.finger_ids].sum(-1).clone()
        pre_lift = env.part.data.root_pos_w.clone()
        # Lift smoothly 8 cm in world Z, using actual PD-driven joint motion.
        peak_forbidden = torch.zeros(n, device=env.device)
        held_height = []
        held_contact = []
        for tick in range(420):
            target = goal.clone()
            target[:, 2] += 0.08 * min((tick + 1) / 300, 1.0)
            p, r = compute_pose_error(*env.tcp_pose(), target[:, :3], target[:, 3:], rot_error_type="axis_angle")
            twist = torch.cat(((5 * p).clamp(-0.08, 0.08), (5 * r).clamp(-0.4, 0.4)), -1)
            velocity = damped_joint_velocity(env.tcp_jacobian(), twist, 0.03).clamp(-1, 1)
            current = env.robot.data.joint_pos[:, env.arm_ids]
            env.robot.set_joint_position_target(current + velocity * env.physics_dt, joint_ids=env.arm_ids)
            env.robot.set_joint_velocity_target(velocity, joint_ids=env.arm_ids)
            step()
            forces = [
                env.scene[name].data.net_forces_w.norm(dim=-1).amax(-1) for name in ("arm_contact", "hand_contact")
            ]
            peak_forbidden = torch.maximum(peak_forbidden, torch.stack(forces).amax(0))
            if tick >= 360:
                held_height.append(env.part.data.root_pos_w[:, 2] - initial[:, 2])
                held_contact.append(bilateral())
        height = torch.stack(held_height).amin(0)
        contact = torch.stack(held_contact).amin(0)
        good = (height > 0.05) & (contact > 0.1) & (close_contact > 0.1) & (peak_forbidden < 3) & (before_close < 0.003)
        for _ in range(4):
            env.sim.render()
            env.scene.update(env.physics_dt)
        raw = env.wrist_camera.data.output["rgb"]
        for i in range(count):
            index = start + i
            target_id = str(data["target_ids"][index])
            record = {
                "target_id": target_id,
                "success": bool(good[i]),
                "held_height_min_m": float(height[i]),
                "bilateral_close_min_n": float(close_contact[i]),
                "bilateral_hold_min_n": float(contact[i]),
                "closed_width_m": float(close_width[i]),
                "initial_settle_drift_m": float(before_close[i]),
                "pre_lift_height_change_m": float(pre_lift[i, 2] - initial[i, 2]),
                "peak_arm_hand_contact_n": float(peak_forbidden[i]),
            }
            records.append(record)
            if i == 0:
                Image.fromarray(raw[i, :, :, :3].cpu().numpy()).save(args.output.parent / f"lift_{target_id}.png")
        report = {
            "catalog": str(args.catalog),
            "contract": env.contract(),
            "finger_force_cap_n": 20,
            "lift_height_m": 0.08,
            "required_held_height_m": 0.05,
            "hold_window_s": 0.5,
            "successes": sum(r["success"] for r in records),
            "tested": len(records),
            "episodes": records,
            "object_fixture": False,
            "validation": "exact_goal_dynamic_close_lift_hold_v1",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[LIFT] {len(records)}/{len(data['target_ids'])} tested, successes={report['successes']}", flush=True)
    keep = np.asarray([r["success"] for r in records])
    if keep.sum() >= 3 and all(np.any((data["split"] == s) & keep) for s in ("train", "validation", "test")):
        target_keys = (
            "target_ids",
            "split",
            "validated",
            "joint_paths",
            "goal_rgbd",
            "goal_poses",
            "object_poses",
            "open_widths",
            "jaw_widths",
            "source_grasp_ids",
            "orientation_ids",
        )
        filtered = {k: (v[keep] if k in target_keys else v) for k, v in data.items()}
        filtered["lift_validated"] = np.ones(int(keep.sum()), dtype=bool)
        args.filtered_catalog.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.filtered_catalog, **filtered)
        print(f"[LIFT] Filtered catalog: {args.filtered_catalog}", flush=True)
    else:
        print("[LIFT] Insufficient split coverage for a lift-filtered catalog; inspect diagnostics.", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        from isaaclab.sim import SimulationContext

        context = SimulationContext.instance()
        if context:
            context.clear_all_callbacks()
            context.clear_instance()
        app.close(wait_for_replicator=False)
