#!/usr/bin/env python3
"""Build independently colored Isaac/MuJoCo references and inspection scenes; no training."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "isaac_rl/source/isaac_rl")]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, default=ROOT / "isaac_rl/data/franka_clutter_v5_placement/catalog.npz")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--gallery", type=Path, required=True)
p.add_argument("--parts", nargs="+")
p.add_argument("--limit-per-part", type=int, default=0)
p.add_argument("--camera-profile", type=Path, help="Regenerate all references with these measured optics")
p.add_argument(
    "--native-resolution",
    action="store_true",
    help="Rebuild camera and network contracts at the explicit native resolution",
)
p.add_argument(
    "--feature-lighting",
    action="store_true",
    help="Use the approved soft feature lights with bounded intensity variation",
)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.source.resolve() == args.output.resolve():
    p.error("Original catalog must be preserved")
args.enable_cameras = True
app = AppLauncher(args).app
import numpy as np
import torch
from grasp_planning.rl.franka_goal_variants import GOAL_VARIANTS_PROFILE, validate_variants, variant_digest
from grasp_planning.rl.franka_mujoco_goals import IsaacMeshMujocoRenderer
from grasp_planning.rl.zed_mini import load_zed_profile, profile_id, resolve_zed_profile
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg
from PIL import Image

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg


def save_image(path, rgb):
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).save(path)


def main():  # noqa: C901 - offline build keeps the state/render sequence together
    torch.set_num_threads(8)
    with np.load(args.source, allow_pickle=False) as src:
        data = {k: src[k].copy() for k in src.files}
    contract = json.loads(data["contract_json"].item())
    source_profile = resolve_zed_profile(contract)
    camera_profile = load_zed_profile(args.camera_profile) if args.camera_profile else source_profile
    for key in (
        "position_m",
        "quaternion_wxyz",
        "parent_link",
        "tcp_offset_in_hand_m",
        "depth_min_m",
        "depth_max_m",
    ):
        if camera_profile[key] != source_profile[key]:
            raise ValueError(f"Optics-only rebuild cannot change {key}")
    resolution_changed = any(
        camera_profile[k] != source_profile[k] for k in ("observation_width", "observation_height")
    )
    if resolution_changed and not args.native_resolution:
        raise ValueError("Resolution migration requires --native-resolution and freshly rendered goals")
    if args.feature_lighting:
        from grasp_planning.rl.franka_feature_lighting import FEATURE_LIGHTING_PROFILE

        contract["feature_lighting"] = FEATURE_LIGHTING_PROFILE
    height, width = camera_profile["observation_height"], camera_profile["observation_width"]
    if resolution_changed:
        contract["training_recipe"]["agent"]["params"]["network"].update(image_height=height, image_width=width)
    if args.camera_profile or contract.get("camera_profile_data"):
        contract["camera_profile"] = profile_id(camera_profile)
        contract["camera_profile_data"] = camera_profile
    count = len(data["target_ids"])
    preferred = [
        "plumbers_block__part_3",
        "plumbers_block__part_4",
        "gamepad__part_2",
        "duct__part_2",
        "stool_circular__part_0",
        "beam__part_0",
    ]
    parts = args.parts or list(dict.fromkeys(data["part_keys"].tolist()))
    parts = sorted(parts, key=lambda v: (preferred.index(v) if v in preferred else 100, v))
    selected = []
    for part in parts:
        ids = np.flatnonzero(data["part_keys"] == part)
        if args.limit_per_part:
            # Spread preview across the catalog instead of taking neighboring grasps.
            ids = ids[np.linspace(0, len(ids) - 1, min(len(ids), args.limit_per_part)).astype(int)]
        selected.extend(ids.tolist())
    selected = np.array(selected)
    for k, v in data.items():
        if v.ndim and v.shape[0] == count:
            data[k] = v[selected]
    count = len(selected)
    n = len(parts)
    assets = contract["object_assets"]
    mapping = [next(i for i, a in enumerate(assets) if a["part_key"] == part) for part in parts]
    cfg = FrankaZedEnvCfg()
    cfg.camera_profile_data = contract.get("camera_profile_data")
    cfg.feature_lighting = contract.get("feature_lighting")
    cfg.seed = 4242
    cfg.build_catalog = True
    cfg.sim.device = args.device
    cfg.scene.num_envs = n
    cfg.object_assets = assets
    cfg.env_part_indices = mapping
    cfg.robot_asset_manifest = "assets/usd/franka_panda_offline/manifest.json"
    cfg.gripper_open_width_m = contract["gripper_open_width_m"]
    cfg.lab_asset_dir = contract["lab_scene"]["asset_dir"]
    cfg.lab_translation = tuple(contract["lab_scene"]["translation_m"])
    cfg.lab_appearance_seed = contract["lab_scene"]["appearance_seed"]
    cfg.lab_props = contract["lab_scene"]["props"]
    cfg.appearance_randomization = contract["appearance_randomization"]
    cfg.performance_profile = contract["performance_profile"]
    cfg.depth_source = "radial_to_optical_z_v1"
    for i in range(min(6, n)):
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
    cameras = [env.scene.sensors.pop(f"inspect{i}") for i in range(min(6, n))]
    for key in ["camera_profile", "lab_scene", "robot_usd"]:
        assert env.contract()[key] == contract[key], key
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.gallery.mkdir(parents=True, exist_ok=True)
    data["goal_rgbd"] = np.empty((count, height, width, 4), np.float16)
    data.pop("goal_rgbd_variants", None)
    variants = np.empty((count, 4, height, width, 4), np.float16)
    colors = np.empty((count, 4, 3), np.float32)
    color_names = np.empty((count, 4), dtype="<U20")
    palette = cfg.appearance_randomization["palette"]
    palette_names = list(palette)
    buckets = [np.flatnonzero(data["part_keys"] == part) for part in parts]
    models = []
    records = []
    depth_checks = []

    def render():
        env.scene.write_data_to_sim()
        env.sim.forward()
        for _ in range(5):
            env.sim.render()
            env.scene.update(env.physics_dt)
        return env.rgbd().cpu().numpy()

    def write(rows, joints, poses):
        env.write_state(
            torch.tensor(joints, device=env.device),
            torch.tensor(poses, device=env.device),
            open_width=torch.tensor(data["open_widths"][rows], device=env.device),
        )

    try:
        with torch.no_grad():
            for batch in range(max(map(len, buckets))):
                rows = np.array([bucket[min(batch, len(bucket) - 1)] for bucket in buckets])
                active = [i for i, bucket in enumerate(buckets) if batch < len(bucket)]
                write(rows, data["joint_paths"][rows, -1], data["object_poses"][rows])
                render()
                if not models:
                    models = [IsaacMeshMujocoRenderer(env, i) for i in range(n)]
                    if cfg.feature_lighting:
                        for model in models:
                            model.lighting_mode = "soft"
                    print("[MESH EXPORT]", [(parts[i], m.mesh_count) for i, m in enumerate(models)], flush=True)
                env.appearance.apply_many(list(range(n)), [4242] * n)
                for slot in range(n):
                    env.appearance.handles[slot]["part"].GetInput("diffuseColor").Set(Gf.Vec3f(*palette["blue"]))
                canonical = render()
                for slot in active:
                    data["goal_rgbd"][rows[slot]] = canonical[slot]
                for variant in range(4):
                    seeds = []
                    for slot, t in enumerate(rows):
                        seed = (
                            int.from_bytes(
                                hashlib.sha256(f"{data['target_ids'][t]}:{variant}:mixed-v1".encode()).digest()[:4],
                                "little",
                            )
                            % 2**31
                        )
                        seeds.append(seed)
                        rng = np.random.default_rng(seed)
                        color_seed = int.from_bytes(
                            hashlib.sha256(str(data["target_ids"][t]).encode()).digest()[:4], "little"
                        )
                        name = np.random.default_rng(color_seed).permutation(palette_names)[variant]
                        colors[t, variant] = np.clip(np.array(palette[name]) * rng.uniform(0.85, 1.15), 0, 0.95)
                        color_names[t, variant] = name
                    if variant < 2:
                        env.appearance.apply_many(list(range(n)), seeds)
                        for slot, t in enumerate(rows):
                            env.appearance.handles[slot]["part"].GetInput("diffuseColor").Set(
                                Gf.Vec3f(*colors[t, variant].tolist())
                            )
                        rgbd = render()
                        for slot in active:
                            t = rows[slot]
                            variants[t, variant] = rgbd[slot]
                    else:
                        for slot in active:
                            t = rows[slot]
                            rgbd = models[slot].render(colors[t, variant], seeds[slot])
                            variants[t, variant] = rgbd
                            if variant == 2:
                                # Ignore large-depth background and boundary pixels for geometry audit.
                                ref = variants[t, 0, :, :, 3].astype(float) * 0.9 + 0.1
                                other = rgbd[:, :, 3] * 0.9 + 0.1
                                interior = np.ones_like(ref, dtype=bool)
                                for axis in (0, 1):
                                    for shift in (-1, 1):
                                        interior &= abs(ref - np.roll(ref, shift, axis)) < 0.002
                                mask = interior & (ref > 0.1) & (ref < 0.45) & (other < 0.95)
                                error = abs(ref[mask] - other[mask]) * 1000
                                check = dict(
                                    target_id=str(data["target_ids"][t]),
                                    pixels=int(mask.sum()),
                                    median_mm=float(np.median(error)),
                                    p95_mm=float(np.percentile(error, 95)),
                                )
                                depth_checks.append(check)
                                if not (mask.sum() > 20 and check["median_mm"] < 5):
                                    save_image(args.gallery / "debug_isaac.png", variants[t, 0, :, :, :3])
                                    save_image(args.gallery / "debug_mujoco.png", rgbd[:, :, :3])
                                    np.savez(args.gallery / "debug_depth.npz", isaac=ref, mujoco=other, mask=mask)
                                    m = models[slot]
                                    print(
                                        "DEBUG",
                                        m.model.cam_fovy,
                                        m.model.cam_pos,
                                        m.model.cam_quat,
                                        m.data.cam_xpos,
                                        m.model.body_pos[m.body_ids[-1]],
                                        flush=True,
                                    )
                                    raise ValueError(("Cross-renderer geometry mismatch", check))
                for slot in active:
                    t = rows[slot]
                    if slot < len(cameras) and batch < 2:
                        stem = f"{parts[slot]}_{batch}"
                        for v in range(4):
                            save_image(args.gallery / f"{stem}_goal_{v}.png", variants[t, v, :, :, :3])
                        save_image(args.gallery / f"{stem}_canonical.png", data["goal_rgbd"][t, :, :, :3])
                # Independent live scene: reachable placement and appearance with unrelated seeds.
                preview = [slot for slot in active if slot < len(cameras) and batch < 2]
                if preview:
                    banks = []
                    for t in rows:
                        valid = (
                            data["pose_reset_valid"][t]
                            & (data["pose_reset_kind"] == 0)
                            & np.isclose(data["pose_reset_progress"], 0.5)
                        )
                        choices = np.flatnonzero(valid)
                        assert len(choices)
                        banks.append(int(choices[(batch + 3) % len(choices)]))
                    banks = np.array(banks)
                    write(rows, data["pose_reset_joints"][rows, banks], data["placement_object_poses"][rows, banks])
                    samples = env.appearance.apply_many(
                        list(range(n)), [71200 + slot * 100 + batch for slot in range(n)]
                    )
                    live = render()
                    for slot in preview:
                        t = rows[slot]
                        stem = f"{parts[slot]}_{batch}"
                        origin = env.scene.env_origins[slot : slot + 1]
                        cameras[slot].set_world_poses_from_view(
                            origin + torch.tensor([[1.25, -1.05, 0.95]], device=env.device),
                            origin + torch.tensor([[0.46, 0, 0.12]], device=env.device),
                        )
                    for _ in range(3):
                        env.sim.render()
                    for slot in preview:
                        t = rows[slot]
                        stem = f"{parts[slot]}_{batch}"
                        cameras[slot].update(env.step_dt, force_recompute=True)
                        Image.fromarray(cameras[slot].data.output["rgb"][0, :, :, :3].cpu().numpy()).save(
                            args.gallery / f"{stem}_scene.png"
                        )
                        save_image(args.gallery / f"{stem}_live.png", live[slot, :, :, :3])
                        # Same metric limits for the actual depth channels in all panels.
                        for label, dep in [
                            ("live", live[slot, :, :, 3]),
                            *[(f"goal_{v}", variants[t, v, :, :, 3]) for v in range(4)],
                        ]:
                            save_image(
                                args.gallery / f"{stem}_{label}_depth.png", np.repeat((1 - dep[..., None]), 3, -1)
                            )
                        records.append(
                            dict(
                                stem=stem,
                                target_id=str(data["target_ids"][t]),
                                part=parts[slot],
                                live_color=samples[slot]["part_color_name"],
                                goal_colors=color_names[t].tolist(),
                                placement_delta=data["placement_delta_xy_yaw"][t, banks[slot]].tolist(),
                            )
                        )
                print(
                    f"[MIXED GOALS] batch {batch + 1}/{max(map(len, buckets))} "
                    f"targets {sum(min(batch + 1, len(b)) for b in buckets)}/{count}",
                    flush=True,
                )
        data["goal_rgbd_variants"] = variants
        data["goal_variant_colors"] = colors
        data["goal_variant_color_names"] = color_names
        contract["depth_source"] = cfg.depth_source
        contract["rgbd_packing"] = "dense_depth_channel_v2"
        contract["goal_randomization"] = {**GOAL_VARIANTS_PROFILE, "images_sha256": variant_digest(variants)}
        data["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
        validate_variants(data, contract["goal_randomization"])
        temp = args.output.with_suffix(".partial.npz")
        np.savez_compressed(temp, **data)
        temp.replace(args.output)
        (args.gallery / "renders.json").write_text(
            json.dumps(
                dict(
                    catalog=str(args.output),
                    targets=count,
                    parts=parts,
                    profile=contract["goal_randomization"],
                    records=records,
                    depth_checks=depth_checks,
                    geometry_source="Isaac USD visual triangles and PhysX body poses; no MuJoCo physics",
                    mujoco_background="USD lab geometry and material base colors; textures and thin decals simplified",
                ),
                indent=2,
            )
        )
        print("[MIXED GOALS COMPLETE]", args.output, flush=True)
    finally:
        for model in models:
            model.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(1)  # Simulator shutdown must not turn a failed build into exit code 0.
    app.close(wait_for_replicator=False)
