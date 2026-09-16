#!/usr/bin/env python3
"""Bounded GPU reset/controller readiness test; oracle uses privileged poses."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'isaac_rl/source/isaac_rl'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--catalog', type=Path, default=ROOT / 'isaac_rl/data/franka_zed_cube/catalog.npz')
parser.add_argument('--camera-profile', type=Path, default=ROOT / 'configs/franka_zed_mini.json')
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/franka_zed/check.json')
parser.add_argument('--num-envs', type=int, default=0, help='Zero checks every catalog target once')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import torch
import numpy as np
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.sim.device = args.device
    with np.load(args.catalog, allow_pickle=False) as source:
        cfg.scene.num_envs = args.num_envs or len(source['target_ids'])
    cfg.sequential_target_assignment = True
    cfg.catalog_path = str(args.catalog)
    cfg.camera_profile_path = str(args.camera_profile)
    cfg.catalog_split = 'all'
    cfg.fixed_waypoint_index = 0
    cfg.rgb_gain_randomization = 0
    env = FrankaZedEnv(cfg)
    report = {'contract': env.contract(), 'catalog_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              'physics_device': env.device, 'cases': {}}
    for mode in ('zero', 'oracle'):
        obs, _ = env.reset(seed=42)
        assert obs['policy'].shape == (env.num_envs, 73742) and obs['critic'].shape == (env.num_envs, 26)
        assert all(torch.isfinite(x).all() for x in obs.values())
        initial = env.pose_errors()[0].norm(dim=-1).clone()
        finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        final = {}
        for step in range(env.max_episode_length + 2):
            action = torch.zeros((env.num_envs, 7), device=env.device)
            if mode == 'oracle':
                p, r = env.pose_errors()
                rot = env.camera_rotation().transpose(1, 2)
                action[:, :3] = (rot @ (2*p)[..., None]).squeeze(-1) / cfg.linear_action_scale_m_s
                action[:, 3:6] = (rot @ (2*r)[..., None]).squeeze(-1) / cfg.angular_action_scale_rad_s
                # Saturate each vector uniformly: component-wise clipping bends
                # a straight approach and can drive fingers into the object.
                for block in (slice(0,3), slice(3,6)):
                    action[:,block] /= action[:,block].abs().amax(-1,keepdim=True).clamp_min(1.)
                action[:, 6] = env._labels(p.norm(dim=-1), r.norm(dim=-1)).ready.float()
            obs, reward, terminated, timeout, _ = env.step(action.clamp(-1, 1))
            assert torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values())
            done = (terminated | timeout) & ~finished
            if not final:
                final = {key: torch.zeros_like(value) for key, value in env.last_transition.items()}
            for key, value in env.last_transition.items():
                final[key][done] = value[done]
            finished |= done
            if finished.all():
                break
        assert finished.all(), 'Not all test episodes terminated'
        report['cases'][mode] = {
            'episodes': env.num_envs, 'initial_position_mean_m': initial.mean().item(),
            'final_position_mean_m': final['position_error_m'].mean().item(),
            'final_position_median_m': final['position_error_m'].median().item(),
            'final_rotation_mean_rad': final['rotation_error_rad'].mean().item(),
            'terminations': {k: int(v.sum()) for k, v in final.items() if v.dtype == torch.bool},
            'episodes_raw': {k: v.tolist() for k, v in final.items()},
            'target_ids': [env.target_ids[i % len(env.target_ids)] for i in range(env.num_envs)],
        }
        print(f'[CHECK] {mode}: {report["cases"][mode]["terminations"]}', flush=True)
    cfg = env.cfg
    cfg.fixed_waypoint_index = env.catalog['joint_paths'].shape[1]-1
    obs, _ = env.reset(seed=42)
    p, r = env.pose_errors()
    report['exact_goal_reset'] = {'max_position_m': p.norm(dim=-1).max().item(),
                                  'max_rotation_rad': r.norm(dim=-1).max().item(),
                                  'image_mae': (env.rgbd()-env.goal_rgbd).abs().mean().item()}
    assert env._labels(p.norm(dim=-1), r.norm(dim=-1)).ready.all()
    report['passed'] = (report['cases']['oracle']['terminations']['success'] >= .95*env.num_envs
                        and report['cases']['zero']['terminations']['success'] == 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    env.close()
    assert report['cases']['oracle']['terminations']['success'] >= .95*env.num_envs, 'Oracle tracking readiness failed'
    assert report['cases']['zero']['terminations']['success'] == 0
    print(f'[CHECK] PASSED: {args.output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        from isaaclab.sim import SimulationContext
        context = SimulationContext.instance()
        if context:
            context.clear_all_callbacks()
            context.clear_instance()
        app.close(wait_for_replicator=False)
