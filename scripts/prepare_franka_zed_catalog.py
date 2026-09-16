#!/usr/bin/env python3
"""Build an explicit GPU-IK/PhysX-checked Panda goal/reset/RGB-D pilot catalog."""

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "isaac_rl/data/franka_zed_cube/catalog.npz")
parser.add_argument("--camera-profile", type=Path, default=ROOT / "configs/franka_zed_mini.json")
parser.add_argument("--goal-spec", type=Path, help="Optional JSON containing object_usd_path and targets with object_pose/goal_pose (XYZ+WXYZ)")
parser.add_argument("--targets", type=int, default=12)
parser.add_argument("--waypoints", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.targets < 3 or args.waypoints < 3:
    parser.error("At least 3 targets/waypoints are required")
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from PIL import Image
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, quat_mul
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg
from grasp_planning.rl.zed_mini import damped_joint_velocity


def main():
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.build_catalog = True
    cfg.camera_profile_path = str(args.camera_profile)
    cfg.sim.device = args.device
    spec = json.loads(args.goal_spec.read_text()) if args.goal_spec else None
    if spec:
        targets = spec["targets"]
        if len(targets) < 3:
            raise ValueError("At least three custom targets are required for the catalog splits")
        cfg.object_usd_path = str((args.goal_spec.parent / spec["object_usd_path"]).resolve())
        cfg.gripper_open_width_m = float(spec.get("gripper_open_width_m", .06))
        if not Path(cfg.object_usd_path).is_file():
            raise FileNotFoundError(cfg.object_usd_path)
        args.targets = len(targets)
        poses = np.asarray([[x["object_pose"], x["goal_pose"]] for x in targets], dtype=float)
        if poses.shape != (args.targets, 2, 7) or not np.isfinite(poses).all():
            raise ValueError("Each target needs finite XYZ+WXYZ object_pose and goal_pose")
        if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1., atol=1e-5):
            raise ValueError("Custom target quaternions must be unit WXYZ")
    cfg.scene.num_envs = args.targets
    env = FrankaZedEnv(cfg)
    n, device = args.targets, env.device
    zeros = torch.zeros(n, device=device)
    yaw = torch.linspace(-.5, .5, n, device=device)
    obj = torch.zeros((n, 7), device=device)
    obj[:, 0] = .43 + .035 * torch.sin(torch.arange(n, device=device)*2.4)
    obj[:, 1] = .045 * torch.cos(torch.arange(n, device=device)*2.4)
    obj[:, 2] = cfg.object_size_m[2]/2
    obj[:, 3:] = quat_from_euler_xyz(zeros, zeros, yaw)
    goals = obj.clone()
    goals[:, 3:] = quat_mul(obj[:, 3:], torch.tensor([0., 1., 0., 0.], device=device).expand(n, -1))
    if spec:
        obj = torch.tensor([x["object_pose"] for x in targets], device=device, dtype=torch.float32)
        goals = torch.tensor([x["goal_pose"] for x in targets], device=device, dtype=torch.float32)
    q = env.robot.data.default_joint_pos[:, env.arm_ids].clone()
    limits = env.robot.data.soft_joint_pos_limits[:, env.arm_ids]
    validated = torch.ones(n, dtype=torch.bool, device=device)
    worst_position = zeros.clone()
    worst_rotation = zeros.clone()
    peak_contact = zeros.clone()
    paths = []
    # Use a smooth Cartesian approach with a small orientation perturbation
    # tapering to the final grasp; solve and check every discrete reset pose.
    for waypoint, progress in enumerate(np.linspace(0, 1, args.waypoints)):
        desired = goals.clone()
        desired[:, 2] += .065 * (1-progress)
        desired[:, 0] += .012 * torch.sin(yaw*4) * (1-progress)
        perturb = quat_from_euler_xyz(.10*(1-progress)*torch.cos(yaw*3),
                                     .10*(1-progress)*torch.sin(yaw*3), zeros)
        desired[:, 3:] = quat_mul(perturb, goals[:, 3:])
        target_pos = desired[:, :3] + env.scene.env_origins
        for _ in range(100 if waypoint == 0 else 30):
            env.write_state(q, obj)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            pos, quat = env.tcp_pose()
            ep, er = compute_pose_error(pos, quat, target_pos, desired[:, 3:], rot_error_type="axis_angle")
            increment = damped_joint_velocity(env.tcp_jacobian(), torch.cat((ep, er), -1), .03)
            q = (q + increment.clamp(-.08, .08)).clamp(limits[..., 0]+.002, limits[..., 1]-.002)
        env.write_state(q, obj)
        env.scene.write_data_to_sim()
        for _ in range(3):
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        pos, quat = env.tcp_pose()
        ep, er = compute_pose_error(pos, quat, target_pos, desired[:, 3:], rot_error_type="axis_angle")
        pn, rn = ep.norm(dim=-1), er.norm(dim=-1)
        contact = env.contact_force()
        validated &= (pn < .002) & (rn < .025) & (contact < .5)
        worst_position = torch.maximum(worst_position, pn)
        worst_rotation = torch.maximum(worst_rotation, rn)
        peak_contact = torch.maximum(peak_contact, contact)
        paths.append(q.clone())
        print(f"[CATALOG] waypoint {waypoint+1}/{args.waypoints}: valid={int(validated.sum())}/{n}, "
              f"max position={float(pn.max()):.5f} max rotation={float(rn.max()):.5f}", flush=True)
    # Capture the actual goal configuration using the exact training scene.
    for _ in range(6):
        env.sim.render()
        env.scene.update(env.physics_dt)
    rgbd = env.rgbd()
    raw = env.wrist_camera.data.output["rgb"]
    rgb = raw[..., :3].float()
    blue = (rgb[..., 2] > rgb[..., 0]*1.3) & (rgb[..., 2] > rgb[..., 1]*1.1)
    # The generated cuboid is blue. Arbitrary supplied meshes instead need
    # visual review; retain the measurable depth/image checks in both cases.
    visible_pixels = blue.sum(dim=(1, 2))
    if spec is None:
        validated &= visible_pixels >= 12
    validated &= torch.isfinite(rgbd).all(dim=(1, 2, 3)) & (rgb.std(dim=(1, 2, 3)) > 1)
    keep = torch.nonzero(validated).flatten()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"requested": n, "accepted": len(keep), "validation": "discrete_gpu_ik_and_hand_contact_v1",
              "not_validated": ["MoveIt paths", "continuous swept collision", "real camera calibration", "physical lift"],
              "max_position_error_m": worst_position.tolist(), "max_rotation_error_rad": worst_rotation.tolist(),
              "peak_hand_contact_n": peak_contact.tolist(), "visible_blue_pixels": visible_pixels.tolist(),
              "accepted_indices": keep.tolist(), "camera": env.camera_profile, "contract": env.contract()}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    for index in range(n):
        Image.fromarray(raw[index, ..., :3].cpu().numpy()).save(args.output.parent / f"goal_{index:03}.png")
    if len(keep) < 3:
        raise RuntimeError(f"Only {len(keep)} valid targets; inspect {args.output.with_suffix('.json')}")
    splits = np.array(["train"]*len(keep), dtype="<U10")
    splits[-2:] = ["validation", "test"]
    def array(x):
        return x[keep].detach().cpu().numpy()
    payload = {
        "contract_json": np.asarray(json.dumps(env.contract(), sort_keys=True)),
        "target_ids": np.array([f"franka_{i:03}" for i in keep.tolist()]),
        "split": splits, "validated": np.ones(len(keep), dtype=bool),
        "joint_paths": array(torch.stack(paths, dim=1)), "goal_rgbd": array(rgbd),
        "goal_poses": array(goals), "object_poses": array(obj),
    }
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(args.output)
    print(f"[CATALOG] Saved {len(keep)} targets to {args.output}", flush=True)
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
