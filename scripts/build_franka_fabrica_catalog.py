#!/usr/bin/env python3
"""Build Panda/ZED training data from authoritative Fabrica stage-2 bundles."""
import argparse
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'isaac_rl/source/isaac_rl'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark-root', type=Path, default=ROOT/'artifacts/fabrica_franka_pilot')
parser.add_argument('--output', type=Path, default=ROOT/'isaac_rl/data/franka_fabrica_plumbers/catalog.npz')
parser.add_argument('--camera-profile', type=Path, default=ROOT/'configs/franka_zed_mini.json')
parser.add_argument('--per-orientation', type=int, default=24)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--waypoints', type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.per_orientation,args.batch_size,args.waypoints)<1 or args.waypoints<3:
    parser.error('Positive counts and at least three waypoints required')
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import trimesh
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.math import compute_pose_error
from grasp_planning.mujoco import write_temporary_triangle_mesh_stl
from grasp_planning.rl.franka_fabrica import collect_targets, portable_path
from grasp_planning.rl.zed_mini import load_zed_profile, damped_joint_velocity
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile=load_zed_profile(args.camera_profile)
    manifest, mesh=collect_targets(args.benchmark_root, per_orientation=args.per_orientation,
                                  tcp_offset=profile['tcp_offset_in_hand_m'])
    targets=manifest['targets']
    if len(targets)<3:
        raise ValueError('Too few source targets')
    shape=trimesh.Trimesh(vertices=mesh.vertices_obj, faces=mesh.faces, process=False)
    mass=float(np.clip(abs(shape.volume)*1240., .005, .5))
    usd=args.output.parent/'part_bundle_local.usd'
    stl=write_temporary_triangle_mesh_stl(mesh, prefix='franka_fabrica_', dir=args.output.parent)
    try:
        MeshConverter(MeshConverterCfg(asset_path=str(stl), usd_dir=str(usd.parent), usd_file_name=usd.name,
            force_usd_conversion=True, make_instanceable=False, scale=(1.,1.,1.),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True,kinematic_enabled=False,disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True,contact_offset=.0001,rest_offset=0.),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg()))
    finally:
        Path(stl).unlink(missing_ok=True)
    manifest.update(object_usd=portable_path(usd), object_mass_kg=mass,
                    object_extents_m=np.ptp(mesh.vertices_obj,axis=0).tolist())
    (args.output.parent/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cfg=FrankaZedEnvCfg()
    cfg.seed=42; cfg.build_catalog=True; cfg.sim.device=args.device
    cfg.scene.num_envs=min(args.batch_size,len(targets)); cfg.object_usd_path=str(usd)
    cfg.object_mass_kg=mass; cfg.camera_profile_path=str(args.camera_profile)
    env=FrankaZedEnv(cfg)
    n=env.num_envs; device=env.device
    limits=env.robot.data.soft_joint_pos_limits[:,env.arm_ids]
    accepted=[]; diagnostics=[]; arrays={k:[] for k in ('joint_paths','goal_rgbd','goal_poses','object_poses','open_widths','jaw_widths')}

    def step():
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)

    def render():
        for _ in range(4):
            env.sim.render(); env.scene.update(env.physics_dt)

    for start in range(0,len(targets),n):
        rows=targets[start:start+n]; count=len(rows); padded=rows+[rows[-1]]*(n-count)
        tensor=lambda key: torch.tensor([x[key] for x in padded],device=device,dtype=torch.float32)
        obj=tensor('object_pose'); goal=tensor('goal_pose'); axis=tensor('approach_axis'); widths=tensor('open_width_m')
        q=env.robot.data.default_joint_pos[:,env.arm_ids].clone()
        valid=torch.ones(n,dtype=torch.bool,device=device)
        maxpos=torch.zeros(n,device=device); maxrot=maxpos.clone(); peak=maxpos.clone()
        paths=[]
        for wi,progress in enumerate(np.linspace(0,1,args.waypoints)):
            desired=goal.clone(); desired[:,:3]-=axis*(.065*(1-progress))
            for _ in range(180 if wi==0 else 40):
                env.write_state(q,obj,open_width=widths); step()
                pos,quat=env.tcp_pose()
                p,r=compute_pose_error(pos,quat,desired[:,:3]+env.scene.env_origins,desired[:,3:],rot_error_type='axis_angle')
                dq=damped_joint_velocity(env.tcp_jacobian(),torch.cat((p,r),-1),.03)
                q=(q+dq.clamp(-.08,.08)).clamp(limits[...,0]+.002,limits[...,1]-.002)
            env.write_state(q,obj,open_width=widths)
            for _ in range(4): step()
            p,r=compute_pose_error(*env.tcp_pose(),desired[:,:3]+env.scene.env_origins,desired[:,3:],rot_error_type='axis_angle')
            pn=p.norm(dim=-1); rn=r.norm(dim=-1); force=env.contact_force()
            valid&=(pn<.002)&(rn<.025)&(force<.5)
            maxpos=torch.maximum(maxpos,pn); maxrot=torch.maximum(maxrot,rn); peak=torch.maximum(peak,force)
            paths.append(q.clone())
        paths=torch.stack(paths,1)
        # Actually drive every approach segment, monitoring contact at 120 Hz.
        env.write_state(paths[:,0],obj,open_width=widths); step()
        for wi in range(1,args.waypoints):
            velocity=(paths[:,wi]-paths[:,wi-1])/(24*env.physics_dt)
            for fraction in np.linspace(1/24,1,24):
                target=paths[:,wi-1]+float(fraction)*(paths[:,wi]-paths[:,wi-1])
                env.robot.set_joint_position_target(target,joint_ids=env.arm_ids)
                env.robot.set_joint_velocity_target(velocity,joint_ids=env.arm_ids)
                step(); peak=torch.maximum(peak,env.contact_force())
        env.robot.set_joint_velocity_target(torch.zeros_like(q),joint_ids=env.arm_ids)
        for _ in range(60): step(); peak=torch.maximum(peak,env.contact_force())
        p,r=compute_pose_error(*env.tcp_pose(),goal[:,:3]+env.scene.env_origins,goal[:,3:],rot_error_type='axis_angle')
        driven_p=p.norm(dim=-1); driven_r=r.norm(dim=-1)
        valid&=(peak<.5)&(driven_p<.003)&(driven_r<.03)
        env.write_state(paths[:,-1],obj,open_width=widths); step(); render()
        rgbd=env.rgbd().clone(); raw=env.wrist_camera.data.output['rgb'].clone()
        depth=env.wrist_camera.data.output['distance_to_image_plane'].clone()
        # Object-only visibility from a background render at identical robot/camera pose.
        hidden=obj.clone(); hidden[:,2]=-5
        env.write_state(paths[:,-1],hidden,open_width=widths); step(); render()
        background=env.wrist_camera.data.output['distance_to_image_plane']
        mask=torch.isfinite(depth)&(depth>=profile['depth_min_m'])&(depth<profile['depth_max_m'])
        mask&=(~torch.isfinite(background))|((background-depth)>.002)
        pixels=mask.flatten(1).sum(-1)
        valid&=(pixels>=24)&torch.isfinite(rgbd).flatten(1).all(-1)
        for i,row in enumerate(rows):
            passed=bool(valid[i]); diagnostics.append(dict(row,accepted=passed,
                max_ik_position_m=float(maxpos[i]),max_ik_rotation_rad=float(maxrot[i]),
                peak_arm_hand_contact_n=float(peak[i]),driven_position_m=float(driven_p[i]),
                driven_rotation_rad=float(driven_r[i]),visible_object_pixels=int(pixels[i])))
            if passed:
                accepted.append(row)
                for k,v in (('joint_paths',paths),('goal_rgbd',rgbd),('goal_poses',goal),('object_poses',obj),
                            ('open_widths',widths),('jaw_widths',tensor('jaw_width_m'))):
                    arrays[k].append(v[i].cpu().numpy())
                Image.fromarray(raw[i,:,:,:3].cpu().numpy()).save(args.output.parent/f'{row["target_id"]}.png')
        print(f'[FABRICA] {min(start+n,len(targets))}/{len(targets)} checked; accepted={len(accepted)}',flush=True)
        report={'requested':len(targets),'accepted':len(accepted),'contract':env.contract(),'targets':diagnostics,
                'checks':['bundle_frame','GPU_IK','arm_hand_contact','driven_120Hz_approach','object_visibility'],
                'not_validated':['physical_lift','real_calibration','global_start_to_pregrasp_motion'],
                'source_manifest':'source_manifest.json'}
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(accepted)<3:
        raise RuntimeError(f'Only {len(accepted)} valid targets; inspect the report')
    payload={k:np.stack(v) for k,v in arrays.items()}
    payload.update(contract_json=np.asarray(json.dumps(env.contract(),sort_keys=True)),
        target_ids=np.asarray([r['target_id'] for r in accepted]),
        split=np.asarray([r['split'] for r in accepted]),validated=np.ones(len(accepted),dtype=bool),
        source_grasp_ids=np.asarray([r['source_grasp_id'] for r in accepted]),
        orientation_ids=np.asarray([r['orientation_id'] for r in accepted]),
        source_bundle_paths=np.asarray([r['path'] for r in manifest['sources']]),
        source_bundle_sha256=np.asarray([r['sha256'] for r in manifest['sources']]))
    for split in ('train','validation','test'):
        if not np.any(payload['split']==split):
            raise RuntimeError(f'No accepted {split} targets; expand candidate selection')
    temporary=args.output.with_suffix('.tmp.npz'); np.savez_compressed(temporary,**payload); temporary.replace(args.output)
    env.close(); print(f'[FABRICA] Saved {len(accepted)} checked targets: {args.output}',flush=True)


if __name__=='__main__':
    try: main()
    finally:
        from isaaclab.sim import SimulationContext
        context=SimulationContext.instance()
        if context: context.clear_all_callbacks(); context.clear_instance()
        app.close(wait_for_replicator=False)
