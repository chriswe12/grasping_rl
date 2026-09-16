#!/usr/bin/env python3
"""Refresh goal RGB-D for a scene appearance change, preserving validated grasp geometry."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lab-assets", type=Path, help="Integrate a lab package and recheck driven approach contacts")
parser.add_argument("--lab-props", action="store_true", help="Include authored collision-enabled kinematic props")
parser.add_argument("--lab-seed", type=int, default=-1, help="Optional fixed setup appearance randomization seed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.source.resolve() == args.output.resolve():
    parser.error("Keep the original catalog; output must be a different path")
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from grasp_planning.rl.franka_fabrica import resolve_project_path, sha256_file
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg
from PIL import Image


def main():
    with np.load(args.source, allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    old = json.loads(str(data["contract_json"].item()))
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.build_catalog = True
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.object_usd_path = str(resolve_project_path(old["object_usd"]))
    cfg.object_mass_kg = old["object_mass_kg"]
    cfg.gripper_open_width_m = old["gripper_open_width_m"]
    if args.lab_assets:
        cfg.lab_asset_dir = str(args.lab_assets)
        cfg.lab_props = args.lab_props
        cfg.lab_appearance_seed = args.lab_seed
    env = FrankaZedEnv(cfg)
    new = env.contract()
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    allowed = {"scene_profile", "lab_scene"} if args.lab_assets else {"scene_profile"}
    if not changed or not changed <= allowed:
        raise ValueError(f"Appearance refresh cannot migrate physical or camera contracts: {changed}")
    for path, digest in zip(data["source_bundle_paths"], data["source_bundle_sha256"]):
        assert sha256_file(resolve_project_path(str(path))) == str(digest), f"Changed source: {path}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements = []
    with torch.inference_mode():
        for i, target in enumerate(data["target_ids"]):

            def tensor(x):
                return torch.as_tensor(x, device=env.device, dtype=torch.float32).unsqueeze(0)

            peak_contact = 0.0
            if args.lab_assets:
                path = tensor(data["joint_paths"][i])[0]
                env.write_state(path[0:1], tensor(data["object_poses"][i]), open_width=tensor(data["open_widths"][i]))
                for wi in range(1, len(path)):
                    velocity = (path[wi] - path[wi - 1]) / (24 * env.physics_dt)
                    for fraction in np.linspace(1 / 24, 1, 24):
                        q = path[wi - 1] + float(fraction) * (path[wi] - path[wi - 1])
                        env.robot.set_joint_position_target(q[None], joint_ids=env.arm_ids)
                        env.robot.set_joint_velocity_target(velocity[None], joint_ids=env.arm_ids)
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(env.physics_dt)
                        peak_contact = max(peak_contact, float(env.contact_force().max()))
                assert peak_contact < cfg.unsafe_contact_force_n, f"Lab approach collision: {target}: {peak_contact} N"
            env.write_state(
                tensor(data["joint_paths"][i, -1]),
                tensor(data["object_poses"][i]),
                open_width=tensor(data["open_widths"][i]),
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            for _ in range(5):
                env.sim.render()
                env.scene.update(env.physics_dt)
            rgbd = env.rgbd()[0].cpu().numpy()
            assert np.isfinite(rgbd).all()
            old_rgbd = data["goal_rgbd"][i].copy()
            data["goal_rgbd"][i] = rgbd
            raw = env.wrist_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            Image.fromarray(raw).save(args.output.parent / f"{target}.png")
            rgb = raw.astype(float) / 255
            blue = (rgb[..., 2] > rgb[..., 0] + 0.08) & (rgb[..., 2] > rgb[..., 1] + 0.025)
            measurements.append(
                {
                    "target_id": str(target),
                    "blue_pixels": int(blue.sum()),
                    "peak_approach_contact_n": peak_contact if args.lab_assets else None,
                    "rgb_mean_change": float(np.abs(rgbd[..., :3] - old_rgbd[..., :3]).mean()),
                    "depth_mean_change": float(np.abs(rgbd[..., 3] - old_rgbd[..., 3]).mean()),
                }
            )
            assert blue.sum() >= 24, f"Part material is missing or invisible: {target}"
            print(f"[RERENDER] {i + 1}/{len(data['target_ids'])} {target} blue_pixels={blue.sum()}", flush=True)
    data["contract_json"] = np.asarray(json.dumps(new, sort_keys=True))
    # Everything except appearance and the appearance contract is preserved byte-for-byte.
    with np.load(args.source, allow_pickle=False) as source:
        for key in data.keys() - {"goal_rgbd", "contract_json"}:
            assert np.array_equal(data[key], source[key]), key
    if args.lab_assets:
        if "lift_validated" in data:
            data.setdefault("source_lift_validated", data["lift_validated"].copy())
            data["lift_validated"] = np.zeros_like(data["lift_validated"])
        data["lab_approach_validated"] = np.ones(len(data["target_ids"]), dtype=bool)
    np.savez_compressed(args.output, **data)
    report = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "passed": True,
        "old_contract": old,
        "new_contract": new,
        "targets": measurements,
        "preserved": "All paths, poses, split assignments and source hashes",
        "physics_checks": "Driven 120 Hz arm/hand contact check for every approach"
        if args.lab_assets
        else "Appearance-only refresh",
        "lift_revalidation_required": bool(args.lab_assets),
    }
    args.output.with_suffix(".rerender.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(f"[RERENDER] COMPLETE {args.output}", flush=True)


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
