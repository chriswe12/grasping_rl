#!/usr/bin/env python3
"""Exercise mixed goal selection through actual resets/steps without learning."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "isaac_rl/source/isaac_rl")]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--catalog", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import numpy as np
import torch
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    torch.set_num_threads(8)
    with np.load(args.catalog) as src:
        parts = len(set(src["part_keys"]))
    cfg = FrankaZedEnvCfg()
    cfg.catalog_path = str(args.catalog)
    cfg.catalog_split = "all"
    cfg.scene.num_envs = parts
    cfg.sim.device = args.device
    cfg.robot_asset_manifest = "assets/usd/franka_panda_offline/manifest.json"
    env = FrankaZedEnv(cfg)
    counts = [0] * 5
    try:
        with torch.inference_mode():
            for fixed in range(5):
                env.cfg.goal_variant_index = fixed
                obs, _ = env.reset()
                assert torch.all(env.goal_variant_indices == fixed)
                expected, indices = env.goal_variants.sample(env.target_index, env.catalog["goal_rgbd"], fixed)
                assert torch.equal(env.goal_rgbd, expected)
                before = env.goal_rgbd.clone()
                # Fixed references must survive normal control steps, including
                # resets that also use the same explicitly chosen backend.
                for _ in range(3):
                    obs, _, terminated, truncated, _ = env.step(torch.zeros((parts, 7), device=env.device))
                    unchanged = ~(terminated | truncated)
                    assert torch.equal(env.goal_rgbd[unchanged], before[unchanged])
                    before = env.goal_rgbd.clone()
                    assert torch.isfinite(obs["policy"]).all()
            env.cfg.goal_variant_index = -1
            for _ in range(12):
                env.reset()
                for i in range(5):
                    counts[i] += int((env.goal_variant_indices == i).sum())
            assert all(counts), counts
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        passed=True,
                        parts=parts,
                        sampled_variant_counts=counts,
                        physics_hz=120,
                        policy_hz=15,
                        optimizer_steps=0,
                    ),
                    indent=2,
                )
            )
            print("[MIXED GOAL SMOKE PASSED]", counts, flush=True)
    finally:
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
