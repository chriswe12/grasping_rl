#!/usr/bin/env python3
"""Verify rendered per-env appearance isolation, deterministic resets and stable canonical goals."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'isaac_rl/source/isaac_rl'))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--catalog', type=Path, default=ROOT/'isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz')
p.add_argument('--output', type=Path, default=ROOT/'artifacts/franka_randomized/appearance_check.json')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras=True
app=AppLauncher(a).app
import torch
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg

def main():
    cfg=FrankaZedEnvCfg();cfg.seed=42;cfg.scene.num_envs=4;cfg.sim.device=a.device
    cfg.catalog_path=str(a.catalog);cfg.catalog_split='all';cfg.fixed_target_index=0;cfg.fixed_waypoint_index=0
    env=FrankaZedEnv(cfg)
    def capture():
        for _ in range(5): env.sim.render();env.scene.update(env.physics_dt)
        return env.wrist_camera.data.output['rgb'][..., :3].float().clone()/255
    with torch.inference_mode():
        env.reset(seed=42); before=capture(); samples=list(env.appearance.samples); goals=env.goal_rgbd.clone()
        env.appearance.apply(0, 912345)
        after=capture(); deltas=(after-before).abs().mean((1,2,3))
        assert deltas[0]>.01, f'Appearance did not update in renderer: {deltas}'
        assert deltas[1:].max()<.01, f'Resetting env0 changed another room: {deltas}'
        changed=list(env.appearance.samples)
        for _ in range(3): env.step(torch.zeros((4,7),device=env.device))
        assert env.appearance.samples==changed, 'Appearance flickered during episode'
        assert torch.equal(env.goal_rgbd,goals), 'Canonical goal changed with live appearance'
        env.reset(seed=42);capture()
        assert env.appearance.samples==samples, 'Seeded resets are not reproducible'
        env.reset(seed=43);capture()
        assert env.appearance.samples!=samples, 'Appearance did not resample'
    report=dict(passed=True, catalog_sha256=hashlib.sha256(a.catalog.read_bytes()).hexdigest(),
                image_change_per_env=deltas.cpu().tolist(), samples=samples,
                canonical_goals_preserved=True, deterministic_reset=True, stable_within_episode=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    env.close();print('[APPEARANCE] PASSED',flush=True)
try:
    main()
finally:
    from isaaclab.sim import SimulationContext
    c=SimulationContext.instance()
    if c: c.clear_all_callbacks();c.clear_instance()
    app.close(wait_for_replicator=False)
