"""GPU Panda/ZED Mini goal-image alignment task with an explicit pilot catalog.

This reuses the project's visual actor and hybrid completion PPO, but owns its
robot/camera/catalog contract instead of pretending PDZ assets are compatible.
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from grasp_planning.rl.franka_fabrica import portable_path, resolve_project_path, sha256_file
from grasp_planning.rl.franka_training_scene import (
    FRANKA_SCENE_PROFILE,
    FrankaTrainingSceneCfg,
    make_franka_part_material_cfg,
    make_franka_render_cfg,
)
from grasp_planning.rl.zed_mini import (
    damped_joint_velocity,
    offset_jacobian,
    pack_zed_rgbd,
    profile_id,
    reproject_intrinsics,
    scaled_intrinsics,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_apply, quat_mul

from .completion import completion_masks, completion_quality, graded_completion_terminal_reward
from .franka_pose_recipe import FrankaPoseRecipeMixin

TASK_ID = "Grasp-Franka-ZEDMini-RGBD-Direct-v0"
DEFAULT_CATALOG = REPO_ROOT / "isaac_rl/data/franka_zed_cube/catalog.npz"


@configclass
class FrankaZedEnvCfg(DirectRLEnvCfg):
    decimation = 8
    episode_length_s = 8.0
    action_space = 7
    observation_space = 72 * 128 * 8 + 14
    state_space = 26
    sim = sim_utils.SimulationCfg(dt=1 / 120, render_interval=8, render=make_franka_render_cfg())
    scene = FrankaTrainingSceneCfg(num_envs=16, env_spacing=2.0)
    camera_profile_path = None
    camera_profile_data = None
    feature_lighting = None  # Explicit catalog-bound soft feature illumination.
    catalog_path = str(DEFAULT_CATALOG)
    catalog_split = "train"
    build_catalog = False
    object_usd_path = ""
    object_assets = None  # Multipart catalogs: one fixed object geometry per environment.
    env_part_indices = None  # Builder-only explicit assignment, never sampled across geometries.
    robot_asset_manifest = ""  # Optional byte-identical offline mirror of the task robot.
    object_size_m = (0.04, 0.025, 0.035)
    object_mass_kg = 0.06
    dynamic_object = False  # used only for separate physical lift validation
    gripper_open_width_m = 0.06
    linear_action_scale_m_s = 0.04
    angular_action_scale_rad_s = 0.24
    maximum_joint_speed_rad_s = 1.0
    dls_damping = 0.05
    unsafe_contact_force_n = 3.0
    ready_position_m = 0.004
    ready_rotation_rad = 0.05235987756
    # Evaluation only: retain the checkpoint's original training objective.
    symmetry_evaluation = False
    symmetry_training = False  # Opt-in objective, checkpoint-contract protected.
    symmetry_objective = "coupled_ready_minimax_tcp_v1"  # Old direct callers remain compatible.
    negative_position_m = 0.008
    negative_rotation_rad = 0.10471975512
    reset_ready_fraction = 0.15
    # Only validated catalog waypoints are sampled. No unchecked joint noise.
    fixed_target_index = -1
    fixed_waypoint_index = -1
    sequential_target_assignment = False
    num_rerenders_on_reset = 2
    # Generic visual perturbations only. No D405 disparity/depth-noise model.
    rgb_gain_randomization = 0.10
    lab_asset_dir = ""  # Empty preserves the existing plain-table scene and contract.
    lab_translation = (0.48, -0.10, 0.0)
    lab_appearance_seed = -1  # Nonnegative: seeded setup appearance, fixed across resets.
    lab_props = False  # Optional collision-enabled kinematic distractors.
    appearance_randomization = None  # Full profile lives in the catalog/checkpoint contract.
    performance_profile = None
    training_recipe = None
    pose_reset_profile = None
    placement_randomization = None
    goal_randomization = None
    depth_source = "legacy_image_plane"
    goal_variant_index = -1  # -1 samples once per episode; 0 is original canonical.
    pose_evaluation = False
    pose_evaluation_progress = 0.0


class FrankaZedEnv(FrankaPoseRecipeMixin, DirectRLEnv):
    cfg: FrankaZedEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):  # noqa: C901 - ordered simulator initialization
        cfg = deepcopy(cfg)
        self.robot_usd_identity = cfg.scene.robot.spawn.usd_path
        self._multipart_source_parts = None
        self._multipart_assignment = None
        if cfg.robot_asset_manifest:
            from grasp_planning.rl.franka_offline_asset import verified_robot_asset

            cfg.scene.robot.spawn.usd_path = verified_robot_asset(
                resolve_project_path(cfg.robot_asset_manifest), self.robot_usd_identity
            )
        # A self-describing Fabrica catalog supplies its asset and physical metadata.
        if not cfg.build_catalog and Path(cfg.catalog_path).is_file():
            with np.load(cfg.catalog_path, allow_pickle=False) as source:
                stored = json.loads(str(source["contract_json"].item()))
                cfg.camera_profile_data = stored.get("camera_profile_data")
                cfg.feature_lighting = stored.get("feature_lighting")
                cfg.performance_profile = stored.get("performance_profile")
                cfg.training_recipe = stored.get("training_recipe")
                cfg.pose_reset_profile = stored.get("pose_reset_profile")
                cfg.placement_randomization = stored.get("placement_randomization")
                cfg.goal_randomization = stored.get("goal_randomization")
                cfg.depth_source = stored.get("depth_source", "legacy_image_plane")
                if cfg.training_recipe:
                    r = cfg.training_recipe["source_settings"]
                    cfg.decimation = 8  # User requires 15 Hz policy, 120 Hz physics.
                    cfg.sim.render_interval = 8
                    cfg.episode_length_s = 15.0 if cfg.pose_evaluation else 12.0
                    cfg.dls_damping = r["dls_damping"]
                    cfg.unsafe_contact_force_n = r["unsafe_contact_force_threshold_n"]
                    cfg.negative_position_m = r["completion_negative_position_m"]
                    cfg.negative_rotation_rad = r["completion_negative_rotation_rad"]
                if stored.get("object_assets"):
                    cfg.object_assets = stored["object_assets"]
                    selected = (
                        np.arange(len(source["target_ids"]))
                        if cfg.catalog_split == "all"
                        else np.flatnonzero(source["split"] == cfg.catalog_split)
                    )
                    self._multipart_source_parts = source["target_part_indices"][selected].copy()
                    from grasp_planning.rl.franka_multipart import part_assignment

                    self._multipart_assignment = part_assignment(
                        self._multipart_source_parts,
                        num_envs=cfg.scene.num_envs,
                        rank=int(os.environ.get("RANK", "0")),
                        world_size=int(os.environ.get("WORLD_SIZE", "1")),
                        sequential=cfg.sequential_target_assignment,
                    )
                    cfg.env_part_indices = self._multipart_assignment[0].tolist()
            if stored.get("object_usd") and not cfg.object_usd_path:
                cfg.object_usd_path = str(resolve_project_path(stored["object_usd"]))
                cfg.object_mass_kg = stored.get("object_mass_kg", cfg.object_mass_kg)
                cfg.gripper_open_width_m = stored["gripper_open_width_m"]
            if stored.get("lab_scene") and not cfg.lab_asset_dir:
                lab = stored["lab_scene"]
                cfg.lab_asset_dir = lab["asset_dir"]
                cfg.lab_translation = tuple(lab["translation_m"])
                cfg.lab_appearance_seed = lab["appearance_seed"]
                cfg.lab_props = lab["props"]
            if cfg.appearance_randomization is None:
                cfg.appearance_randomization = stored.get("appearance_randomization")
        self.lab_contract = None
        self.lab_prop_names = []
        if cfg.lab_asset_dir:
            from grasp_planning.rl.video_lab_scene import training_asset_digest

            from pxr import Usd, UsdGeom, UsdPhysics

            folder = resolve_project_path(cfg.lab_asset_dir)
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["meters_per_unit"] != 1.0 or manifest["up_axis"] != "Z" or manifest["table_top_z_m"] != 0.0:
                raise ValueError("Lab must be metre/Z-up with tabletop at z=0")
            source_stage = Usd.Stage.Open(str(folder / "environment.usdc"))
            if UsdGeom.GetStageMetersPerUnit(source_stage) != 1.0 or UsdGeom.GetStageUpAxis(source_stage) != "Z":
                raise ValueError("Lab USD metadata disagrees with its metre/Z-up manifest")
            for path in manifest["collision_paths"]:
                prim = source_stage.GetPrimAtPath(path)
                if (
                    not prim
                    or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                ):
                    raise ValueError(f"Missing or disabled lab collider: {path}")
            del source_stage
            cfg.scene.table = None
            cfg.scene.env_spacing = 6.0
            cfg.scene.lab = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Lab",
                init_state=AssetBaseCfg.InitialStateCfg(pos=cfg.lab_translation),
                spawn=sim_utils.UsdFileCfg(usd_path=str(folder / "environment.usdc")),
            )
            if cfg.lab_props:
                for prop in manifest["props"]:
                    name = "lab_prop_" + prop["name"]
                    yaw = prop["preview_yaw_rad"]
                    import math

                    setattr(
                        cfg.scene,
                        name,
                        RigidObjectCfg(
                            prim_path="{ENV_REGEX_NS}/" + name,
                            init_state=RigidObjectCfg.InitialStateCfg(
                                pos=tuple(a + b for a, b in zip(cfg.lab_translation, prop["preview_position"])),
                                rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                            ),
                            spawn=sim_utils.UsdFileCfg(
                                usd_path=str(folder / prop["asset"]),
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            ),
                        ),
                    )
                    self.lab_prop_names.append(name)
            self.lab_contract = dict(
                asset_dir=portable_path(folder),
                asset_sha256=training_asset_digest(folder),
                translation_m=list(cfg.lab_translation),
                appearance_seed=cfg.lab_appearance_seed,
                props=cfg.lab_props,
                prop_mode="authored_kinematic_v1",
                layout="panda_left_mount_v1",
            )
        from grasp_planning.rl.zed_mini import resolve_zed_profile

        self.camera_profile = resolve_zed_profile(
            {"camera_profile_data": cfg.camera_profile_data}, cfg.camera_profile_path
        )
        profile = self.camera_profile
        cfg.observation_space = profile["observation_height"] * profile["observation_width"] * 8 + 14
        if cfg.symmetry_training and cfg.symmetry_objective not in (
            "coupled_ready_minimax_tcp_v1",
            "orbit_potential_pose_set_v2",
        ):
            raise ValueError("Unsupported symmetry training objective")
        if cfg.symmetry_training and cfg.symmetry_objective == "orbit_potential_pose_set_v2":
            if not cfg.training_recipe:
                raise ValueError("Symmetry potential v2 requires a pose training recipe")
            cfg.observation_space += 18  # 24-value set descriptor replaces six pose labels.
        if not 0 < cfg.gripper_open_width_m <= 0.08:
            raise ValueError("Panda open width must be in (0, 0.08] metres")
        if cfg.depth_source not in ("legacy_image_plane", "radial_to_optical_z_v1"):
            raise ValueError("Unsupported depth source")
        if cfg.depth_source == "radial_to_optical_z_v1":
            cfg.scene.wrist_camera.data_types = ["rgb", "distance_to_camera"]
            cfg.scene.wrist_camera.depth_clipping_behavior = "none"
            # Radial clipping must not discard optical-Z-valid corner rays.
            # Pack/reproject applies the policy's 1 m optical-Z limit afterwards.
        cfg.scene.wrist_camera.width = profile["render_width"]
        cfg.scene.wrist_camera.height = profile["render_height"]
        cfg.scene.wrist_camera.offset.pos = tuple(profile["position_m"])
        cfg.scene.wrist_camera.offset.rot = tuple(profile["quaternion_wxyz"])
        cfg.scene.wrist_camera.spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=scaled_intrinsics(profile, profile["render_width"], profile["render_height"]),
            width=profile["render_width"],
            height=profile["render_height"],
            clipping_range=(0.01, profile["depth_max_m"] * (2 if cfg.depth_source == "radial_to_optical_z_v1" else 1)),
        )
        if cfg.object_assets:
            if cfg.object_usd_path:
                raise ValueError("Multipart catalog cannot override its object geometry")
            if cfg.env_part_indices is None or len(cfg.env_part_indices) != cfg.scene.num_envs:
                raise ValueError("Every multipart environment needs an explicit part assignment")
            assets = []
            for part_index in cfg.env_part_indices:
                item = cfg.object_assets[part_index]
                usd = resolve_project_path(item["object_usd"])
                if sha256_file(usd) != item["object_sha256"]:
                    raise ValueError(f"Multipart object changed: {usd}")
                assets.append(
                    sim_utils.UsdFileCfg(
                        usd_path=str(usd),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=not cfg.dynamic_object),
                        mass_props=sim_utils.MassPropertiesCfg(mass=item["object_mass_kg"]),
                        visual_material=make_franka_part_material_cfg(),
                    )
                )
            cfg.scene.part.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=False)
            cfg.scene.replicate_physics = False
        elif cfg.object_usd_path:
            cfg.scene.part.spawn = sim_utils.UsdFileCfg(
                usd_path=str(resolve_project_path(cfg.object_usd_path)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=not cfg.dynamic_object),
                mass_props=sim_utils.MassPropertiesCfg(mass=cfg.object_mass_kg),
                visual_material=make_franka_part_material_cfg(),
            )
        else:
            cfg.scene.part.spawn.size = tuple(cfg.object_size_m)
            cfg.scene.part.spawn.rigid_props.kinematic_enabled = not cfg.dynamic_object
        cfg.scene.arm_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_link[1-7]", history_length=3)
        cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = cfg.gripper_open_width_m / 2
        if cfg.appearance_randomization:
            if not cfg.lab_asset_dir:
                raise ValueError("Per-episode Franka appearance requires the lab scene")
            from grasp_planning.rl.franka_appearance import validate_profile

            validate_profile(cfg.appearance_randomization)
            # Let USD material/light updates reach RTX before the reset observation.
            cfg.num_rerenders_on_reset = max(cfg.num_rerenders_on_reset, 5)
            cfg.scene.key_light = None
            cfg.scene.light.spawn.intensity = 180.0
            cfg.scene.random_key = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/RandomKey",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, -0.2, 1.5)),
                spawn=sim_utils.SphereLightCfg(intensity=4000.0, radius=0.15, normalize=True),
            )
        if cfg.performance_profile:
            from grasp_planning.rl.franka_performance import configure_fast_render

            configure_fast_render(cfg, cfg.performance_profile)
        super().__init__(cfg, render_mode, **kwargs)
        self.arm_ids, _ = self.robot.find_joints("panda_joint[1-7]", preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints("panda_finger_joint[12]")
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.jacobian_id = self.hand_id - int(self.robot.is_fixed_base)
        self.tcp_offset = torch.tensor(profile["tcp_offset_in_hand_m"], device=self.device).expand(self.num_envs, -1)
        self.mount_quat = torch.tensor(profile["quaternion_wxyz"], device=self.device).expand(self.num_envs, -1)
        self.actions = torch.zeros((self.num_envs, 7), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.previous_potential = torch.zeros(self.num_envs, device=self.device)
        self.target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.open_width = torch.full((self.num_envs,), cfg.gripper_open_width_m, device=self.device)
        self.rgb_gain = torch.ones((self.num_envs, 1, 1, 3), device=self.device)
        self.terminal_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_collision = torch.zeros_like(self.terminal_success)
        self.terminal_premature = torch.zeros_like(self.terminal_success)
        self.terminal_divergence = torch.zeros_like(self.terminal_success)
        self.last_transition = {}
        self.catalog = None
        if not cfg.build_catalog:
            self._load_catalog()

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.part = self.scene["part"]
        self.wrist_camera = self.scene["wrist_camera"]
        if self.lab_contract:
            from grasp_planning.rl.video_lab_scene import randomize_environment, repair_exported_curve_widths

            for root in self.scene.env_prim_paths:
                repair_exported_curve_widths(self.sim.stage, root + "/Lab")
                if self.cfg.lab_appearance_seed >= 0:
                    randomize_environment(
                        self.sim.stage,
                        root + "/Lab",
                        self.cfg.lab_appearance_seed,
                        resolve_project_path(self.cfg.lab_asset_dir),
                    )
        self.appearance = None
        if self.cfg.appearance_randomization:
            from grasp_planning.rl.franka_appearance import FrankaAppearanceRandomizer

            self.appearance = FrankaAppearanceRandomizer(
                self.sim.stage,
                self.scene.env_prim_paths,
                self.scene.env_origins.cpu().tolist(),
                self.lab_prop_names,
                self.cfg.appearance_randomization,
                fixed_wear=bool(self.cfg.performance_profile),
            )

        if self.cfg.feature_lighting:
            from grasp_planning.rl.franka_feature_lighting import install

            install(self)

        self.simplified_background_paths = []
        if self.cfg.performance_profile:
            from grasp_planning.rl.franka_performance import simplify_background

            self.simplified_background_paths = simplify_background(self.sim.stage, self.scene.env_prim_paths)

    def step(self, action):
        if not self.cfg.performance_profile:
            return super().step(action)
        # Same Isaac Lab 2.3 DirectRLEnv transition order; render after ALL
        # physics and resets, because rewards/dones never consume camera images.
        # This removes the pre-reset image that would otherwise be discarded.
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids):
            self._reset_idx(reset_ids)
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            renders = self.cfg.num_rerenders_on_reset if len(reset_ids) else 1
            for _ in range(renders):
                self.sim.render()
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def contract(self):
        result = {
            "schema_version": 3,
            "task": TASK_ID,
            "contact_monitor": "arm_links_1_7_hand_fingers_v1",
            "image_processing": "rectified_intrinsics_warp_area_depth_v1",
            "controller": "tcp_camera_velocity_dls_position_velocity_pd_v1",
            "camera_profile": profile_id(self.camera_profile),
            "scene_profile": FRANKA_SCENE_PROFILE,
            "robot_usd": self.robot_usd_identity,
            "object_usd": portable_path(resolve_project_path(self.cfg.object_usd_path))
            if self.cfg.object_usd_path
            else "",
            "object_sha256": sha256_file(resolve_project_path(self.cfg.object_usd_path))
            if self.cfg.object_usd_path
            else "",
            "object_mass_kg": self.cfg.object_mass_kg,
            "object_size_m": list(self.cfg.object_size_m),
            "gripper_open_width_m": self.cfg.gripper_open_width_m,
            "tcp_offset_in_hand_m": self.camera_profile["tcp_offset_in_hand_m"],
        }
        if self.cfg.camera_profile_data is not None:
            result["camera_profile_data"] = self.camera_profile
        if self.cfg.feature_lighting:
            result["feature_lighting"] = self.cfg.feature_lighting
        if self.cfg.performance_profile:
            result["performance_profile"] = self.cfg.performance_profile
        if self.lab_contract:
            result["scene_profile"] += "+video_lab_v1"
            result["lab_scene"] = self.lab_contract
        if self.cfg.appearance_randomization:
            result["appearance_randomization"] = self.cfg.appearance_randomization
        if self.cfg.object_assets:
            result["object_assets"] = self.cfg.object_assets
            result["multipart_assignment"] = "fixed_geometry_per_environment_v1"
        if self.cfg.placement_randomization:
            result["placement_randomization"] = self.cfg.placement_randomization
        if self.cfg.depth_source != "legacy_image_plane":
            result["depth_source"] = self.cfg.depth_source
            result["rgbd_packing"] = "dense_depth_channel_v2"
        if self.cfg.goal_randomization:
            result["goal_randomization"] = self.cfg.goal_randomization
        if self.cfg.symmetry_training:
            result["symmetry_training"] = {
                "profile": "franka_object_symmetry_v2",
                "objective": self.cfg.symmetry_objective,
                "assets": getattr(self, "symmetry_training_assets", {}),
                "ready_position_m": self.cfg.ready_position_m,
                "ready_rotation_rad": self.cfg.ready_rotation_rad,
            }
        if self.cfg.symmetry_training and self.cfg.symmetry_objective == "orbit_potential_pose_set_v2":
            result["symmetry_training"].update(
                dense_potential="max_weighted_progress_precision_v2",
                auxiliary_loss="minimum_pose_set_huber_geodesic_v2",
                auxiliary_descriptor="local_tcp7_goal7_camera9_orbit1_v1",
                axial_solver={"intervals": 64, "iterations": 24, "method": "all_interval_golden_v1"},
            )
        if self.cfg.training_recipe:
            result["training_recipe"] = self.cfg.training_recipe
            result["pose_reset_profile"] = self.cfg.pose_reset_profile
        return result

    def _sample_goal_images(self, env_ids, target):
        if self.goal_variants is None:
            self.goal_rgbd[env_ids] = self.catalog["goal_rgbd"][target]
            return
        images, indices = self.goal_variants.sample(target, self.catalog["goal_rgbd"], self.cfg.goal_variant_index)
        self.goal_rgbd[env_ids] = images
        self.goal_variant_indices[env_ids] = indices

    def _load_catalog(self):
        path = Path(self.cfg.catalog_path)
        if not path.is_file():
            raise FileNotFoundError(f"Build the Franka catalog first: {path}")
        with np.load(path, allow_pickle=False) as src:
            data = {k: src[k].copy() for k in src.files}
        catalog_contract = self.contract().copy()
        # Catalog geometry/images predate the opt-in objective; checkpoint
        # contracts below still require the exact training objective on resume.
        catalog_contract.pop("symmetry_training", None)
        if json.loads(str(data["contract_json"].item())) != catalog_contract:
            raise ValueError("Franka catalog robot/camera/scene contract mismatch; rebuild it")
        count = len(data["target_ids"])
        if data["joint_paths"].shape[0] != count or data["joint_paths"].shape[-1] != 7:
            raise ValueError("Invalid Franka reset paths")
        if data["goal_rgbd"].shape != (
            count,
            self.camera_profile["observation_height"],
            self.camera_profile["observation_width"],
            4,
        ):
            raise ValueError("Invalid Franka goal RGB-D")
        for key in ("joint_paths", "goal_rgbd", "goal_poses", "object_poses"):
            if not np.isfinite(data[key]).all():
                raise ValueError(f"Nonfinite catalog array {key}")
        if not data["validated"].all():
            raise ValueError("Catalog contains unvalidated paths/goals")
        for path, digest in zip(data.get("source_bundle_paths", []), data.get("source_bundle_sha256", [])):
            if sha256_file(resolve_project_path(str(path))) != str(digest):
                raise ValueError(f"Source grasp bundle changed: {path}; rebuild the catalog")
        selected = (
            np.arange(count)
            if self.cfg.catalog_split == "all"
            else np.flatnonzero(data["split"] == self.cfg.catalog_split)
        )
        if len(selected) == 0:
            raise ValueError(f"Empty split: {self.cfg.catalog_split}")
        self.target_ids = [str(data["target_ids"][i]) for i in selected]
        if self._multipart_assignment is not None:
            mapping, table, counts = self._multipart_assignment
            self.part_indices = torch.as_tensor(mapping, device=self.device, dtype=torch.long)
            self.part_target_table = torch.as_tensor(table, device=self.device, dtype=torch.long)
            self.part_target_counts = torch.as_tensor(counts, device=self.device, dtype=torch.long)
            self.part_target_cursor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.catalog = {
            key: torch.as_tensor(data[key][selected], device=self.device, dtype=torch.float32)
            for key in ("joint_paths", "goal_rgbd", "goal_poses", "object_poses")
        }
        self.symmetry_evaluator = None
        self.symmetry_report = None
        if self.cfg.symmetry_evaluation or self.cfg.symmetry_training:
            if self.cfg.dynamic_object or (self.cfg.symmetry_evaluation and not self.cfg.pose_evaluation):
                raise ValueError("Symmetry requires static objects; symmetry_evaluation also requires pose_evaluation")
            from grasp_planning.rl.franka_symmetry import SymmetryEvaluator, catalog_symmetries

            orbits, self.symmetry_report = catalog_symmetries(data, selected)
            if self.cfg.symmetry_training:
                assemblies = sorted({str(key).split("__part_")[0] for key in data["part_keys"]})
                paths = [REPO_ROOT / "assets/obj/fabrica" / a / "symmetries.json" for a in assemblies]
                paths.append(REPO_ROOT / "assets/obj/fabrica/continuous_symmetries.json")
                self.symmetry_training_assets = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}
            self.symmetry_aux_config = (
                {
                    "orbits": [o.tolist() for o in orbits],
                    "continuous": [r["continuous"] for r in self.symmetry_report["targets"]],
                    "rotation_scale": self.recipe["auxiliary_rotation_scale_rad"]
                    if hasattr(self, "recipe")
                    else self.cfg.training_recipe["source_settings"]["auxiliary_rotation_scale_rad"],
                }
                if self.cfg.symmetry_training and self.cfg.symmetry_objective == "orbit_potential_pose_set_v2"
                else None
            )
            self.symmetry_evaluator = SymmetryEvaluator(
                orbits, self.device, continuous=[r["continuous"] for r in self.symmetry_report["targets"]]
            )
        self.catalog["open_widths"] = torch.as_tensor(
            data.get("open_widths", np.full(count, self.cfg.gripper_open_width_m))[selected],
            device=self.device,
            dtype=torch.float32,
        )
        self.catalog["jaw_widths"] = torch.as_tensor(
            data.get("jaw_widths", np.full(count, 0.04))[selected], device=self.device, dtype=torch.float32
        )
        self.goal_variants = None
        self.goal_variant_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg.goal_randomization:
            from grasp_planning.rl.franka_goal_variants import GoalVariantBank

            self.goal_variants = GoalVariantBank(data, selected, self.cfg.goal_randomization, self.device)
        self.goal_rgbd = torch.zeros(
            (self.num_envs, self.camera_profile["observation_height"], self.camera_profile["observation_width"], 4),
            device=self.device,
        )
        self.goal_pose = torch.zeros((self.num_envs, 7), device=self.device)
        if self.cfg.training_recipe:
            self._init_pose_recipe(data, selected)

    def tcp_pose(self):
        quat = self.robot.data.body_quat_w[:, self.hand_id]
        offset = quat_apply(quat, self.tcp_offset)
        return self.robot.data.body_pos_w[:, self.hand_id] + offset, quat

    def camera_rotation(self):
        return matrix_from_quat(quat_mul(self.robot.data.body_quat_w[:, self.hand_id], self.mount_quat))

    def tcp_jacobian(self):
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jacobian_id][:, :, self.arm_ids]
        # PhysX articulation Jacobians are in the root frame. Rotate both
        # blocks to world, then move the linear Jacobian to the physical TCP.
        rotation = matrix_from_quat(self.robot.data.root_quat_w)
        jac = torch.cat((rotation @ jac[:, :3], rotation @ jac[:, 3:]), dim=1)
        return offset_jacobian(jac, quat_apply(self.robot.data.body_quat_w[:, self.hand_id], self.tcp_offset))

    def symmetry_potential(self):
        from grasp_planning.rl.franka_symmetry_objective import orbit_potential

        pos, quat = self.tcp_pose()
        return orbit_potential(
            self.symmetry_evaluator, torch.cat((pos, quat), -1), self.goal_pose, self.target_index, self.recipe
        )

    def pose_errors(self):
        pos, quat = self.tcp_pose()
        goal = self.goal_pose
        if self.cfg.symmetry_training:
            # A single equivalent TCP supplies translation AND rotation, including
            # off-centre displacement. Keep rendered goal images unchanged.
            *_, goal = self.symmetry_evaluator.evaluate(
                torch.cat((pos, quat), -1),
                goal,
                self.target_index,
                self.cfg.ready_position_m,
                self.cfg.ready_rotation_rad,
                return_goal=True,
            )
        return compute_pose_error(pos, quat, goal[:, :3], goal[:, 3:], rot_error_type="axis_angle")

    def evaluation_error_norms(self):
        """Display/scoring distances; leave the policy's nominal pose errors intact."""
        if self.cfg.symmetry_evaluation and self.cfg.pose_evaluation:
            pos, quat = self.tcp_pose()
            p, r, _, _ = self.symmetry_evaluator.evaluate(
                torch.cat((pos, quat), -1),
                self.goal_pose,
                self.target_index,
                self.cfg.ready_position_m,
                self.cfg.ready_rotation_rad,
            )
            return p, r
        p, r = self.pose_errors()
        return p.norm(dim=-1), r.norm(dim=-1)

    def contact_force(self):
        forces = [
            torch.linalg.vector_norm(self.scene[name].data.net_forces_w, dim=-1).amax(dim=-1)
            for name in ("arm_contact", "hand_contact", "left_finger_contact", "right_finger_contact")
        ]
        return torch.stack(forces).amax(dim=0)

    def write_state(self, q, object_pose, env_ids=None, open_width=None):
        """Catalog builder/reset helper. Object poses use env-local XYZ + WXYZ."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if open_width is not None:
            self.open_width[env_ids] = open_width
        joints = self.robot.data.default_joint_pos[env_ids].clone()
        joints[:, self.arm_ids] = q
        joints[:, self.finger_ids] = self.open_width[env_ids, None] / 2
        self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
        self.robot.set_joint_position_target(joints, env_ids=env_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(joints), env_ids=env_ids)
        state = self.part.data.default_root_state[env_ids].clone()
        state[:, :7] = object_pose
        state[:, :3] += self.scene.env_origins[env_ids]
        state[:, 7:] = 0.0
        self.part.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        self.part.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)

    def rgbd(self):
        output = self.wrist_camera.data.output
        p = self.camera_profile
        if self.cfg.depth_source == "radial_to_optical_z_v1":
            from grasp_planning.rl.zed_mini import optical_depth_from_radial

            metric_depth = optical_depth_from_radial(
                output["distance_to_camera"], self.wrist_camera.data.intrinsic_matrices
            )
        else:
            metric_depth = output["distance_to_image_plane"]
        rgb, depth = reproject_intrinsics(
            output["rgb"],
            metric_depth,
            self.wrist_camera.data.intrinsic_matrices,
            scaled_intrinsics(p, p["render_width"], p["render_height"]),
        )
        return pack_zed_rgbd(rgb, depth, p, legacy_batch_layout=self.cfg.depth_source == "legacy_image_plane")[0]

    def _pre_physics_step(self, actions):
        if self.cfg.training_recipe:
            self._pose_actions(torch.nan_to_num(actions))
            return
        self.actions = torch.nan_to_num(actions.clone()).clamp(-1, 1)
        self.actions[:, 6] = self.actions[:, 6].clamp(0, 1)

    def _apply_action(self):
        rotation = self.camera_rotation()
        linear = self.actions[:, :3] * self.cfg.linear_action_scale_m_s
        angular = self.actions[:, 3:6] * self.cfg.angular_action_scale_rad_s
        twist = torch.cat(((rotation @ linear[..., None]).squeeze(-1), (rotation @ angular[..., None]).squeeze(-1)), -1)
        velocity = damped_joint_velocity(self.tcp_jacobian(), twist, self.cfg.dls_damping)
        velocity.clamp_(-self.cfg.maximum_joint_speed_rad_s, self.cfg.maximum_joint_speed_rad_s)
        q = self.robot.data.joint_pos[:, self.arm_ids] + velocity * self.physics_dt
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        self.robot.set_joint_position_target(q.clamp(limits[..., 0], limits[..., 1]), joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(velocity, joint_ids=self.arm_ids)
        self.previous_actions.copy_(self.actions[:, :6])

    def _labels(self, pos, rot):
        return completion_masks(
            pos,
            rot,
            ready_position_m=self.cfg.ready_position_m,
            ready_rotation_rad=self.cfg.ready_rotation_rad,
            negative_position_m=self.cfg.negative_position_m,
            negative_rotation_rad=self.cfg.negative_rotation_rad,
            collision_free=self.contact_force() < self.cfg.unsafe_contact_force_n,
        )

    def _get_dones(self):
        p, r = self.pose_errors()
        pn, rn = p.norm(dim=-1), r.norm(dim=-1)
        declared = self.completion_declaration if self.cfg.training_recipe else self.actions[:, 6] >= 0.5
        collision = self.contact_force() >= self.cfg.unsafe_contact_force_n
        ready = self._labels(pn, rn).ready
        divergence = (pn > (0.18 if self.cfg.training_recipe else 0.20)) | (rn > 1.2) | ~torch.isfinite(pn + rn)
        symmetry_diagnostics = {}
        if self.cfg.symmetry_evaluation and self.cfg.pose_evaluation:
            pos, quat = self.tcp_pose()
            tcp = torch.cat((pos, quat), -1)
            symmetry_diagnostics = {
                "nominal_position_error_m": pn.clone(),
                "nominal_rotation_error_rad": rn.clone(),
                "nominal_success": (declared & ready).clone(),
            }
            pn, rn, representative, ready = self.symmetry_evaluator.evaluate(
                tcp, self.goal_pose, self.target_index, self.cfg.ready_position_m, self.cfg.ready_rotation_rad
            )
            ready &= ~collision
            # An equivalent valid goal must not trigger nominal-pose divergence.
            limit = 0.18 if self.cfg.training_recipe else 0.20
            divergence = ~self.symmetry_evaluator.evaluate(tcp, self.goal_pose, self.target_index, limit, 1.2)[-1]
            symmetry_diagnostics["symmetry_representative"] = representative.clone()
            # Keep complete poses so later tolerance checks can select their own
            # paired representative instead of reusing a strict-tolerance minimum.
            for column, name in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
                symmetry_diagnostics["tcp_" + name] = tcp[:, column].clone()
                symmetry_diagnostics["goal_" + name] = self.goal_pose[:, column].clone()
        if self.cfg.symmetry_training:
            pos, quat = self.tcp_pose()
            limit = 0.18 if self.cfg.training_recipe else 0.20
            divergence = ~self.symmetry_evaluator.evaluate(
                torch.cat((pos, quat), -1), self.goal_pose, self.target_index, limit, 1.2
            )[-1]
        self.terminal_success = declared & ready
        self.terminal_collision = collision
        self.terminal_premature = declared & ~ready
        self.terminal_divergence = divergence
        timeout = self.episode_length_buf >= (
            self.pose_timeout_steps - 1 if self.cfg.training_recipe else self.max_episode_length - 1
        )
        self.last_transition = {
            "position_error_m": pn.clone(),
            "rotation_error_rad": rn.clone(),
            "success": self.terminal_success.clone(),
            "collision": collision.clone(),
            "premature": self.terminal_premature.clone(),
            "divergence": divergence.clone(),
            "timeout": timeout.clone(),
            **symmetry_diagnostics,
        }
        if self.cfg.training_recipe:
            self.last_transition.update(
                initial_position_error_m=self.initial_position_error.clone(),
                initial_rotation_error_rad=self.initial_rotation_error.clone(),
                reset_progress=self.reset_progress.clone(),
                reset_mode=self.reset_mode.clone(),
                reset_bank_index=self.reset_bank_index.clone(),
            )
        for name in ("arm_contact", "hand_contact", "left_finger_contact", "right_finger_contact"):
            self.last_transition[name + "_n"] = self.scene[name].data.net_forces_w.norm(dim=-1).amax(-1).clone()
        return declared | collision | divergence, timeout

    def _get_rewards(self):
        p, r = self.pose_errors()
        pn, rn = p.norm(dim=-1), r.norm(dim=-1)
        if self.cfg.training_recipe:
            return self._pose_reward(pn, rn)
        potential = -10 * pn - rn
        quality = completion_quality(
            pn,
            rn,
            ready_position_m=self.cfg.ready_position_m,
            ready_rotation_rad=self.cfg.ready_rotation_rad,
            negative_position_m=self.cfg.negative_position_m,
            negative_rotation_rad=self.cfg.negative_rotation_rad,
            collision_free=~self.terminal_collision,
        )
        terminal = graded_completion_terminal_reward(
            self.actions[:, 6] >= 0.5, quality, correct_reward=5.0, premature_penalty=2.0
        )
        reward = potential - self.previous_potential - 0.002 * self.actions[:, :6].square().sum(-1) - 0.002
        reward += terminal - 2 * self.terminal_collision.float() - self.terminal_divergence.float()
        self.previous_potential.copy_(potential)
        return reward

    def _get_observations(self):
        p, r = self.pose_errors()
        rotation = self.camera_rotation().transpose(1, 2)
        pc, rc = (rotation @ p[..., None]).squeeze(-1), (rotation @ r[..., None]).squeeze(-1)
        live = self.rgbd()
        if self.cfg.training_recipe:
            live = self._pose_live(live)
        live = torch.cat(((live[..., :3] * self.rgb_gain).clamp(0, 1), live[..., 3:]), -1)
        images = torch.cat((live, self.goal_rgbd), -1).flatten(start_dim=1)
        labels = self._labels(p.norm(dim=-1), r.norm(dim=-1))
        pose_labels = torch.cat(
            (pc / 0.10, rc / (self.recipe["auxiliary_rotation_scale_rad"] if self.cfg.training_recipe else 0.5)), -1
        )
        if self.cfg.symmetry_training and self.cfg.symmetry_objective == "orbit_potential_pose_set_v2":
            pos, quat = self.tcp_pose()
            pose_labels = torch.cat(
                (
                    pos - self.scene.env_origins,
                    quat,
                    self.goal_pose[:, :3] - self.scene.env_origins,
                    self.goal_pose[:, 3:],
                    self.camera_rotation().flatten(1),
                    (self.target_index / len(self.symmetry_evaluator.valid))[:, None],
                ),
                -1,
            )
        policy = torch.cat(
            (
                images,
                self.previous_actions,
                pose_labels,
                labels.ready[:, None].float(),
                labels.supervised[:, None].float(),
            ),
            -1,
        )
        critic = torch.cat(
            (
                self.robot.data.joint_pos[:, self.arm_ids],
                self.robot.data.joint_vel[:, self.arm_ids],
                p,
                r,
                self.previous_actions,
            ),
            -1,
        )
        return {"policy": policy, "critic": critic}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.catalog is None:
            raise RuntimeError("Builder mode cannot be used as a training environment")
        n = len(env_ids)
        target = torch.randint(len(self.target_ids), (n,), device=self.device)
        if self._multipart_assignment is not None:
            counts = self.part_target_counts[env_ids]
            choice = (torch.rand(n, device=self.device) * counts).long()
            if self.cfg.sequential_target_assignment:
                choice = self.part_target_cursor[env_ids] % counts
                self.part_target_cursor[env_ids] += 1
            target = self.part_target_table[env_ids, choice]
        elif self.cfg.sequential_target_assignment:
            target = env_ids % len(self.target_ids)
        if self.cfg.fixed_target_index >= 0:
            if self._multipart_assignment is not None:
                raise ValueError("Global fixed target override is incompatible with fixed multipart geometry")
            target[:] = self.cfg.fixed_target_index
        if self.cfg.training_recipe:
            self._pose_reset(env_ids, target)
            return
        paths = self.catalog["joint_paths"]
        waypoint = torch.randint(paths.shape[1] - 1, (n,), device=self.device)
        waypoint[torch.rand(n, device=self.device) < self.cfg.reset_ready_fraction] = paths.shape[1] - 1
        if self.cfg.fixed_waypoint_index >= 0:
            waypoint[:] = self.cfg.fixed_waypoint_index
        self.target_index[env_ids] = target
        self.goal_pose[env_ids] = self.catalog["goal_poses"][target]
        self.goal_pose[env_ids, :3] += self.scene.env_origins[env_ids]
        self._sample_goal_images(env_ids, target)
        self.write_state(
            paths[target, waypoint],
            self.catalog["object_poses"][target],
            env_ids,
            open_width=self.catalog["open_widths"][target],
        )
        self.scene.reset(env_ids)
        self.sim.forward()
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        p, r = self.pose_errors()
        self.previous_potential[env_ids] = (-10 * p.norm(dim=-1) - r.norm(dim=-1))[env_ids]
        amount = self.cfg.rgb_gain_randomization
        self.rgb_gain[env_ids] = 1 + amount * (2 * torch.rand((n, 1, 1, 3), device=self.device) - 1)
        if self.appearance:
            seeds = torch.randint(2**31 - 1, (n,), device=self.device).cpu().tolist()
            samples = self.appearance.apply_many(env_ids.cpu().tolist(), seeds)
            self.rgb_gain[env_ids] = torch.tensor(
                [sample["rgb_gain"] for sample in samples], device=self.device
            ).reshape(n, 1, 1, 3)
        self.extras["log"] = {
            "success": self.terminal_success[env_ids].float().mean(),
            "unsafe_collision": self.terminal_collision[env_ids].float().mean(),
            "premature_completion": self.terminal_premature[env_ids].float().mean(),
        }
