#!/usr/bin/env python3
"""Physical multipart Panda closure/lift validation; one matched geometry per environment."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--catalog", type=Path, default=ROOT / "isaac_rl/data/franka_fabrica_all/catalog.npz")
parser.add_argument("--camera-profile", type=Path, default=ROOT / "configs/franka_zed_mini.json")
parser.add_argument(
    "--allow-missing-splits",
    action="store_true",
    help="Diagnostic catalogs only; production requires train/validation/test",
)
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/franka_all_20260915/lifts.json")
parser.add_argument(
    "--filtered-catalog", type=Path, default=ROOT / "isaac_rl/data/franka_fabrica_all/catalog_lift_validated.npz"
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

from isaaclab.utils.math import compute_pose_error, quat_apply


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
    cfg.robot_asset_manifest = "assets/usd/franka_panda_offline/manifest.json"
    part_indices = np.unique(data["target_part_indices"])
    cfg.scene.num_envs = len(part_indices)
    groups = [np.flatnonzero(data["target_part_indices"] == i).tolist() for i in part_indices]
    env = FrankaZedEnv(cfg)
    n = env.num_envs
    # The short pilot's 8 cm motion can leave the far edge of a larger part on
    # the table. Use part size for commanded height and measure the actual
    # mesh's clearance, retaining the existing lift and contact requirements.
    from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle
    from grasp_planning.mujoco import build_bundle_local_mesh
    from scipy.spatial import ConvexHull

    hulls = []
    heights = []
    assets = json.loads(str(data["contract_json"].item()))["object_assets"]
    for part_index in part_indices:
        item = assets[int(part_index)]
        assembly, part_id = item["part_key"].split("__part_")
        source = next(
            str(p) for p in data["source_bundle_paths"] if f"/parts/{assembly}/{part_id}/orientations/" in str(p)
        )
        mesh = build_bundle_local_mesh(load_grasp_bundle(ROOT / source))
        vertices = np.asarray(mesh.vertices_obj, dtype=np.float64)
        hulls.append(vertices[ConvexHull(vertices).vertices])
        heights.append(max(0.08, float(np.ptp(vertices, axis=0).max()) + 0.04))
    vertex_array = np.zeros((n, max(map(len, hulls)), 3), dtype=np.float32)
    vertex_valid = np.zeros(vertex_array.shape[:2], dtype=bool)
    for i, vertices in enumerate(hulls):
        vertex_array[i, : len(vertices)] = vertices
        vertex_valid[i, : len(vertices)] = True
    vertices = torch.tensor(vertex_array, device=env.device)
    vertex_mask = torch.tensor(vertex_valid, device=env.device)
    lift_height = torch.tensor(heights, device=env.device, dtype=torch.float32)
    lift_ticks = max(300, int(np.ceil(max(heights) / 0.05 / env.physics_dt)))
    total_ticks = lift_ticks + 120

    def clearance():
        q = env.part.data.root_quat_w[:, None, :].expand(-1, vertices.shape[1], -1)
        rotated = quat_apply(q, vertices)
        z = rotated[:, :, 2] + env.part.data.root_pos_w[:, None, 2] - env.scene.env_origins[:, None, 2]
        return z.masked_fill(~vertex_mask, float("inf")).amin(-1)

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

    for batch_index in range(max(map(len, groups))):
        active = [batch_index < len(group) for group in groups]
        indices = torch.tensor([group[min(batch_index, len(group) - 1)] for group in groups], device=env.device)
        count = n
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
        initial_tcp = env.tcp_pose()[0].clone()
        initial_quat = env.part.data.root_quat_w.clone()
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
        # Lift with sufficient clearance for this geometry, using PD-driven joints.
        peak_forbidden = torch.zeros(n, device=env.device)
        held_height = []
        held_contact = []
        held_clearance = []
        for tick in range(total_ticks):
            target = goal.clone()
            target[:, 2] += lift_height * min((tick + 1) / lift_ticks, 1.0)
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
            if tick >= total_ticks - 60:
                held_height.append(env.part.data.root_pos_w[:, 2] - initial[:, 2])
                held_contact.append(bilateral())
                held_clearance.append(clearance())
        height = torch.stack(held_height).amin(0)
        contact = torch.stack(held_contact).amin(0)
        actual_tcp = env.tcp_pose()[0]
        tcp_lift = actual_tcp[:, 2] - initial_tcp[:, 2]
        tcp_error = (actual_tcp - target[:, :3]).norm(dim=-1)
        _, rotation_change = compute_pose_error(
            initial, initial_quat, env.part.data.root_pos_w, env.part.data.root_quat_w, rot_error_type="axis_angle"
        )
        clear_height = torch.stack(held_clearance).amin(0)
        good = (
            (height > 0.05)
            & (clear_height > 0.01)
            & (tcp_error < 0.005)
            & (contact > 0.1)
            & (close_contact > 0.1)
            & (peak_forbidden < 3)
            & (before_close < 0.003)
        )
        for _ in range(4):
            env.sim.render()
            env.scene.update(env.physics_dt)
        raw = env.wrist_camera.data.output["rgb"]
        for i in range(count):
            if not active[i]:
                continue
            index = int(indices[i])
            target_id = str(data["target_ids"][index])
            record = {
                "source_index": index,
                "target_id": target_id,
                "success": bool(good[i]),
                "held_height_min_m": float(height[i]),
                "commanded_lift_height_m": float(lift_height[i]),
                "held_mesh_clearance_min_m": float(clear_height[i]),
                "bilateral_close_min_n": float(close_contact[i]),
                "bilateral_hold_min_n": float(contact[i]),
                "closed_width_m": float(close_width[i]),
                "initial_settle_drift_m": float(before_close[i]),
                "tcp_lift_height_m": float(tcp_lift[i]),
                "tcp_target_error_m": float(tcp_error[i]),
                "object_rotation_change_rad": float(rotation_change[i].norm()),
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
            "lift_height_policy": "max(0.08, largest_part_extent + 0.04)",
            "lift_duration_s": lift_ticks * env.physics_dt,
            "required_held_height_m": 0.05,
            "required_mesh_clearance_m": 0.01,
            "hold_window_s": 0.5,
            "successes": sum(r["success"] for r in records),
            "tested": len(records),
            "episodes": records,
            "object_fixture": False,
            "validation": "exact_goal_dynamic_close_lift_clearance_hold_v2",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[LIFT] {len(records)}/{len(data['target_ids'])} tested, successes={report['successes']}", flush=True)
    records.sort(key=lambda r: r["source_index"])
    assert [r["source_index"] for r in records] == list(range(len(data["target_ids"])))
    keep = np.asarray([r["success"] for r in records])
    if keep.sum() >= 3 and (
        args.allow_missing_splits or all(np.any((data["split"] == s) & keep) for s in ("train", "validation", "test"))
    ):
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
            "target_part_indices",
            "part_keys",
            "lab_approach_validated",
        )
        filtered = {k: (v[keep] if k in target_keys else v) for k, v in data.items()}
        filtered["lift_validated"] = np.ones(int(keep.sum()), dtype=bool)
        filtered["lift_clearance_validated"] = np.ones(int(keep.sum()), dtype=bool)
        filtered["lift_validation_json"] = np.asarray(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "validation",
                        "lift_height_policy",
                        "required_held_height_m",
                        "required_mesh_clearance_m",
                        "hold_window_s",
                        "finger_force_cap_n",
                        "object_fixture",
                    )
                },
                sort_keys=True,
            )
        )
        args.filtered_catalog.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.filtered_catalog, **filtered)
        print(f"[LIFT] Filtered catalog: {args.filtered_catalog}", flush=True)
    else:
        raise RuntimeError("Insufficient split coverage for a lift-filtered catalog; inspect diagnostics.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        from isaaclab.sim import SimulationContext

        context = SimulationContext.instance()
        if context:
            context.clear_all_callbacks()
            context.clear_instance()
        app.close(wait_for_replicator=False)
