"""Direct Isaac Lab environment for RGB-D goal-conditioned grasp alignment."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[7]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    D405WristCameraConfig,
)
from grasp_planning.envs.fr3_part_env import _spawn_local_ground_plane
from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_MATERIAL_PROFILE,
    apply_visual_servo_materials,
)
from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_GROUND_COLOR,
    VISUAL_SERVO_SCENE_PROFILE,
    spawn_visual_servo_lights,
)
from grasp_planning.planning.fr3_motion_context import FR3MotionContext, grasp_pose_to_tcp_pose
from grasp_planning.rl.live_observation_randomization import (
    LiveObservationRandomizationCfg,
    LiveObservationRandomizer,
)
from grasp_planning.rl.scene_appearance_randomization import (
    SceneAppearanceRandomizationCfg,
    SceneAppearanceRandomizer,
)
from grasp_planning.start_poses import (
    KUKA_Y_GRIPPER_APPROACH_PROFILE,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul

from .completion import (
    completion_declared,
    completion_masks,
    completion_terminal_reward,
    update_completion_streak,
)
from .isaac_rl_env_cfg import GraspVisualServoEnvCfg
from .multigrasp_catalog import (
    load_multigrasp_catalog,
    load_multigrasp_rotation_resets,
    select_catalog_split,
)
from .reset_position_sampling import (
    position_offset_profile,
    sample_collision_safe_xy_offsets_from_profile,
)
from .training_curriculum import (
    RESET_MODE_BOUNDARY,
    RESET_MODE_NO_NOISE,
    RESET_MODE_PATH,
    RESET_MODE_READY,
    apply_failure_replay,
    curriculum_state,
    reset_timeout_seconds,
    sample_reset_modes,
    update_failure_scores,
)
from .visual_servo_math import (
    approach_progress_bucket_masks,
    balanced_group_target_progress,
    balanced_target_progress,
    interpolate_joint_trajectory,
    path_conditioned_noise_scale,
    world_pose_error_to_camera,
)


class GraspVisualServoEnv(DirectRLEnv):
    cfg: GraspVisualServoEnvCfg

    def __init__(self, cfg: GraspVisualServoEnvCfg, render_mode: str | None = None, **kwargs):
        self.live_observation_randomizer: LiveObservationRandomizer | None = None
        self.scene_appearance_randomizer: SceneAppearanceRandomizer | None = None
        self.visual_light_paths: dict[str, str] = {}
        super().__init__(cfg, render_mode, **kwargs)
        self.visual_material_bindings = apply_visual_servo_materials()
        if self.cfg.scene_appearance_randomization_enabled:
            self.scene_appearance_randomizer = SceneAppearanceRandomizer(
                SceneAppearanceRandomizationCfg(
                    enabled=True,
                    interval_steps=int(self.cfg.scene_appearance_randomization_interval_steps),
                    key_yaw_delta_deg=tuple(self.cfg.scene_key_yaw_delta_deg),
                    key_pitch_delta_deg=tuple(self.cfg.scene_key_pitch_delta_deg),
                    key_intensity_scale=tuple(self.cfg.scene_key_intensity_scale),
                    key_angle_deg=tuple(self.cfg.scene_key_angle_deg),
                    dome_intensity_scale=tuple(self.cfg.scene_dome_intensity_scale),
                    light_temperature_shift=tuple(self.cfg.scene_light_temperature_shift),
                    part_color_scale=tuple(self.cfg.scene_part_color_scale),
                    part_hue_shift_deg=tuple(self.cfg.scene_part_hue_shift_deg),
                    part_roughness=tuple(self.cfg.scene_part_roughness),
                    finger_color_scale=tuple(self.cfg.scene_finger_color_scale),
                    ground_color_scale=tuple(self.cfg.scene_ground_color_scale),
                    ground_hue_shift_deg=tuple(self.cfg.scene_ground_hue_shift_deg),
                    ground_roughness=tuple(self.cfg.scene_ground_roughness),
                ),
                light_paths=self.visual_light_paths,
                material_paths=self.visual_material_bindings["materials"],
                device=self.device,
            )
            self.scene_appearance_randomizer.maybe_randomize(
                0,
                force=True,
                strength=0.0 if self.cfg.training_curriculum_enabled else 1.0,
            )
        self.context = FR3MotionContext(
            robot=self.robot,
            scene=self.scene,
            sim=self.sim,
            fixed_gripper_width=0.084,
        )
        self.arm_ids = self.context.arm_joint_ids
        self.previous_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.completion_probability = torch.zeros(self.num_envs, device=self.device)
        self.completion_stop_candidate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.completion_declaration = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_positive_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_exact_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_mode = torch.full((self.num_envs,), RESET_MODE_PATH, dtype=torch.long, device=self.device)
        self.reset_timeout_steps = torch.full(
            (self.num_envs,), self.max_episode_length, dtype=torch.long, device=self.device
        )
        self.reset_timeout_s = torch.full((self.num_envs,), float(self.cfg.episode_length_s), device=self.device)
        self.reset_failure_replay = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_position_error = torch.zeros(self.num_envs, device=self.device)
        self.previous_rotation_error = torch.zeros(self.num_envs, device=self.device)
        self.reset_progress = torch.zeros(self.num_envs, device=self.device)
        self.reset_noise_scale = torch.zeros(self.num_envs, device=self.device)
        self.reset_rotation_command = torch.zeros(self.num_envs, device=self.device)
        self.reset_position_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.reset_position_requested = torch.zeros(self.num_envs, device=self.device)
        self.reset_position_safe_cap = torch.zeros(self.num_envs, device=self.device)
        self.initial_position_error = torch.zeros(self.num_envs, device=self.device)
        self.initial_rotation_error = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_fraction = 0.0
        self.curriculum_progress_min = float(self.cfg.reset_progress_min)
        self.curriculum_perturbation_scale = 1.0
        self.curriculum_visual_strength = 1.0
        self.rotation_tcp_from_camera = torch.tensor(
            D405WristCameraConfig().rotation_camera_in_calibration_parent,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3, 3)
        self.live_observation_randomizer = LiveObservationRandomizer(
            LiveObservationRandomizationCfg(
                enabled=bool(self.cfg.live_observation_randomization_enabled),
                exposure_stops=tuple(self.cfg.live_rgb_exposure_stops),
                contrast=tuple(self.cfg.live_rgb_contrast),
                gamma=tuple(self.cfg.live_rgb_gamma),
                white_balance_gain=tuple(self.cfg.live_rgb_white_balance_gain),
                vignette_strength=tuple(self.cfg.live_rgb_vignette_strength),
                rgb_noise_std=tuple(self.cfg.live_rgb_noise_std),
                blur_probability=float(self.cfg.live_rgb_blur_probability),
                blur_mix=tuple(self.cfg.live_rgb_blur_mix),
                depth_scale=tuple(self.cfg.live_depth_scale),
                depth_bias_m=tuple(self.cfg.live_depth_bias_m),
                depth_noise_std_m=tuple(self.cfg.live_depth_noise_std_m),
                depth_quantization_m=float(self.cfg.live_depth_quantization_m),
                depth_dropout_probability=tuple(self.cfg.live_depth_dropout_probability),
                depth_edge_dropout_probability=tuple(self.cfg.live_depth_edge_dropout_probability),
                depth_edge_threshold_m=float(self.cfg.live_depth_edge_threshold_m),
                rgb_patch_occlusion_probability=float(self.cfg.live_rgb_patch_occlusion_probability),
                depth_patch_dropout_probability=float(self.cfg.live_depth_patch_dropout_probability),
                patch_area_fraction=tuple(self.cfg.live_patch_area_fraction),
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

        catalog_path = Path(self.cfg.goal_catalog_data_path).expanduser()
        if catalog_path.is_file():
            complete_catalog = load_multigrasp_catalog(
                catalog_path,
                expected_arm_joint_count=len(self.arm_ids),
                require_complete=True,
            )
            complete_target_ids = tuple(str(value) for value in complete_catalog["target_ids"].tolist())
            approach_profile = str(np.asarray(complete_catalog.get("approach_gripper_profile", "")).item())
            if approach_profile != KUKA_Y_GRIPPER_APPROACH_PROFILE:
                raise ValueError(
                    "Goal catalog approach-gripper mismatch: "
                    f"catalog='{approach_profile or 'unlabeled'}', "
                    f"environment='{KUKA_Y_GRIPPER_APPROACH_PROFILE}'. Rebuild the "
                    "path asset and re-capture every Isaac goal image before training "
                    "or playback."
                )
            material_profile = str(np.asarray(complete_catalog.get("visual_material_profile", "")).item())
            if material_profile != VISUAL_SERVO_MATERIAL_PROFILE:
                raise ValueError(
                    "Goal catalog visual material mismatch: "
                    f"catalog='{material_profile or 'unlabeled'}', "
                    f"environment='{VISUAL_SERVO_MATERIAL_PROFILE}'. Re-capture the "
                    "Isaac goal RGB-D catalog before training or playback."
                )
            scene_profile = str(np.asarray(complete_catalog.get("visual_scene_profile", "")).item())
            if scene_profile != VISUAL_SERVO_SCENE_PROFILE:
                raise ValueError(
                    "Goal catalog visual scene mismatch: "
                    f"catalog='{scene_profile or 'unlabeled'}', "
                    f"environment='{VISUAL_SERVO_SCENE_PROFILE}'. Re-capture the "
                    "catalog under the canonical lighting and RTX profile."
                )
            camera_profile = str(np.asarray(complete_catalog.get("goal_camera_profile", "")).item())
            if camera_profile != D405_VISUAL_SERVO_CAMERA_PROFILE:
                raise ValueError(
                    "Goal catalog camera mismatch: "
                    f"catalog='{camera_profile or 'unlabeled'}', "
                    f"environment='{D405_VISUAL_SERVO_CAMERA_PROFILE}'. Re-capture "
                    "the goal catalog with native D405 intrinsics; do not scale the "
                    "focal lengths to the 256x144 render buffer."
                )
            observation_profile = str(np.asarray(complete_catalog.get("goal_observation_profile", "")).item())
            if observation_profile != D405_VISUAL_SERVO_OBSERVATION_PROFILE:
                raise ValueError(
                    "Goal catalog observation preprocessing mismatch: "
                    f"catalog='{observation_profile or 'unlabeled'}', "
                    f"environment='{D405_VISUAL_SERVO_OBSERVATION_PROFILE}'. "
                    "Re-capture the goal catalog before training or playback."
                )
            catalog, catalog_indices = select_catalog_split(complete_catalog, str(self.cfg.catalog_split))
            print(
                f"[INFO] Loaded {len(catalog['target_ids'])}/"
                f"{len(complete_catalog['target_ids'])} validated grasp targets "
                f"from {catalog_path} (split={self.cfg.catalog_split}).",
                flush=True,
            )
        else:
            if self.cfg.require_multigrasp_catalog:
                raise FileNotFoundError(
                    f"Required multi-grasp catalog does not exist: {catalog_path}. "
                    "Generate and validate it with "
                    "python3 isaac_rl/scripts/prepare_multigrasp_catalog.py."
                )
            print(
                f"[WARNING] Multi-grasp catalog not found at {catalog_path}; "
                "using the legacy single target for playback.",
                flush=True,
            )
            catalog = self._legacy_single_goal_catalog()
            complete_catalog = catalog
            complete_target_ids = tuple(str(value) for value in catalog["target_ids"].tolist())
            catalog_indices = np.arange(len(complete_target_ids), dtype=np.int64)

        self.target_ids = tuple(str(value) for value in catalog["target_ids"].tolist())
        self.orientation_names = tuple(str(value) for value in catalog["orientation_names"].tolist())
        self.target_count = len(self.target_ids)
        self.catalog_target_indices = torch.as_tensor(catalog_indices, dtype=torch.long, device=self.device)
        self.part_names = tuple(
            str(value) for value in catalog.get("part_names", np.asarray(self.cfg.part_names)).tolist()
        )
        configured_part_names = tuple(str(value) for value in self.cfg.part_names)
        if self.part_names != configured_part_names:
            raise ValueError(
                f"Catalog part_names={self.part_names} do not match configured scene parts={configured_part_names}."
            )
        self.fixed_target_index = int(self.cfg.fixed_target_index)
        if str(self.cfg.fixed_target_id):
            if self.fixed_target_index >= 0:
                raise ValueError("Set only one of fixed_target_index and fixed_target_id.")
            try:
                self.fixed_target_index = self.target_ids.index(str(self.cfg.fixed_target_id))
            except ValueError as exc:
                raise ValueError(
                    f"Unknown fixed_target_id='{self.cfg.fixed_target_id}'. Available targets: {self.target_ids}."
                ) from exc
        if self.fixed_target_index >= self.target_count:
            raise ValueError(
                f"fixed_target_index={self.fixed_target_index} is outside the {self.target_count}-target catalog."
            )
        self.target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_sample_counts = torch.zeros(self.target_count, dtype=torch.long, device=self.device)
        self.target_failure_scores = torch.zeros(self.target_count, dtype=torch.float32, device=self.device)
        self.target_terminal_counts = torch.zeros(self.target_count, dtype=torch.long, device=self.device)
        self.target_cursor = 0
        self.target_orientation_indices = torch.as_tensor(
            catalog["orientation_indices"], dtype=torch.long, device=self.device
        )
        self.target_part_indices = torch.as_tensor(
            catalog.get("part_indices", np.zeros(self.target_count, dtype=np.int64)),
            dtype=torch.long,
            device=self.device,
        )
        self.part_cursor = 0
        self.part_target_cursors = torch.zeros(
            torch.unique(self.target_part_indices).numel(),
            dtype=torch.long,
            device=self.device,
        )
        self.target_split_ids = tuple(
            str(value) for value in catalog.get("split_ids", np.asarray(["train"] * self.target_count)).tolist()
        )
        self.object_positions_catalog = torch.as_tensor(
            catalog["object_positions_w"], dtype=torch.float32, device=self.device
        )
        object_quaternions_xyzw = torch.as_tensor(
            catalog["object_orientations_xyzw_w"],
            dtype=torch.float32,
            device=self.device,
        )
        self.object_quaternions_catalog = object_quaternions_xyzw[:, (3, 0, 1, 2)]
        self.goal_tcp_positions_catalog = torch.as_tensor(
            catalog["goal_tcp_positions_w"], dtype=torch.float32, device=self.device
        )
        goal_tcp_quaternions_xyzw = torch.as_tensor(
            catalog["goal_tcp_orientations_xyzw_w"],
            dtype=torch.float32,
            device=self.device,
        )
        self.goal_tcp_quaternions_catalog = goal_tcp_quaternions_xyzw[:, (3, 0, 1, 2)]
        self.reset_joint_trajectories = torch.as_tensor(
            catalog["reset_joint_trajectories"],
            dtype=torch.float32,
            device=self.device,
        )
        self.reset_path_progress_catalog = torch.as_tensor(
            catalog["reset_path_progress"],
            dtype=torch.float32,
            device=self.device,
        )
        self.approach_gripper_widths_catalog = torch.as_tensor(
            catalog["approach_gripper_widths_m"],
            dtype=torch.float32,
            device=self.device,
        )
        raw_rgb = catalog["goal_rgb"]
        raw_depth = catalog["goal_depth"]
        expected_raw_shape = (self.cfg.wrist_camera.height, self.cfg.wrist_camera.width)
        if raw_rgb.shape[1:3] != expected_raw_shape:
            raise ValueError(
                f"Catalog goal images have shape {raw_rgb.shape[1:3]}, but this task's "
                f"camera requires {expected_raw_shape}."
            )
        observation_size = (
            VISUAL_SERVO_OBSERVATION_HEIGHT,
            VISUAL_SERVO_OBSERVATION_WIDTH,
        )
        goal_rgb_t = torch.as_tensor(raw_rgb, device=self.device).float().div_(255.0)
        goal_rgb_t = F.interpolate(
            goal_rgb_t.permute(0, 3, 1, 2),
            size=observation_size,
            mode="area",
        ).permute(0, 2, 3, 1)
        goal_depth_t = torch.as_tensor(raw_depth, device=self.device).float().unsqueeze(1)
        goal_depth_t = F.interpolate(
            goal_depth_t,
            size=observation_size,
            mode="area",
        ).permute(0, 2, 3, 1)
        goal_depth_t = goal_depth_t.sub_(0.04).div_(0.46).clamp_(0.0, 1.0)
        self.goal_rgbd_catalog = torch.cat((goal_rgb_t, goal_depth_t), dim=-1)
        self.goal_rgbd = self.goal_rgbd_catalog[0:1].repeat(self.num_envs, 1, 1, 1)
        self.goal_tcp_position = self.goal_tcp_positions_catalog[0:1].repeat(self.num_envs, 1)
        self.goal_tcp_position += self.scene.env_origins
        self.goal_tcp_quaternion = self.goal_tcp_quaternions_catalog[0:1].repeat(self.num_envs, 1)

        self.rotation_reset_joint_trajectories: torch.Tensor | None = None
        self.rotation_reset_angle_profile: torch.Tensor | None = None
        self.rotation_reset_collision_clearance: torch.Tensor | None = None
        self.nominal_reset_collision_clearance: torch.Tensor | None = None
        self.rotation_reset_minimum_collision_clearance_m = 0.0
        rotation_reset_path = Path(self.cfg.rotation_reset_data_path).expanduser()
        if self.cfg.reset_rotation_randomization_enabled and rotation_reset_path.is_file():
            rotation_resets = load_multigrasp_rotation_resets(
                rotation_reset_path,
                expected_target_ids=complete_target_ids,
                expected_waypoint_count=self.reset_joint_trajectories.shape[1],
                expected_arm_joint_count=len(self.arm_ids),
                expected_approach_gripper_widths_m=complete_catalog["approach_gripper_widths_m"],
            )
            authored_far_rotation = float(rotation_resets["rotation_angle_profile_rad"][0])
            if abs(authored_far_rotation - self.cfg.reset_rotation_far_rad) > 1.0e-5:
                raise ValueError(
                    "Rotation reset asset/config mismatch: "
                    f"asset={authored_far_rotation:.6f} rad, "
                    f"config={self.cfg.reset_rotation_far_rad:.6f} rad."
                )
            self.rotation_reset_joint_trajectories = torch.as_tensor(
                rotation_resets["rotation_joint_trajectories"][catalog_indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.rotation_reset_angle_profile = torch.as_tensor(
                rotation_resets["rotation_angle_profile_rad"],
                dtype=torch.float32,
                device=self.device,
            )
            self.rotation_reset_collision_clearance = torch.as_tensor(
                rotation_resets["collision_clearance_m"][catalog_indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.nominal_reset_collision_clearance = torch.as_tensor(
                rotation_resets["nominal_collision_clearance_m"][catalog_indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.rotation_reset_minimum_collision_clearance_m = float(
                np.asarray(rotation_resets["minimum_collision_clearance_m"]).item()
            )
            print(
                f"[INFO] Loaded {self.rotation_reset_joint_trajectories.shape[1]} "
                f"collision-validated rotation resets per grasp from {rotation_reset_path}.",
                flush=True,
            )
        elif self.cfg.reset_rotation_randomization_enabled:
            if self.cfg.require_rotation_reset_data:
                raise FileNotFoundError(
                    f"Required rotation reset asset does not exist: {rotation_reset_path}. "
                    "Generate it with python3 "
                    "isaac_rl/scripts/build_multigrasp_rotation_reset_asset.py."
                )
            print(
                f"[WARNING] Rotation reset asset not found at {rotation_reset_path}; "
                "rotation randomization is disabled for this playback.",
                flush=True,
            )
        if self.cfg.reset_position_randomization_enabled and self.rotation_reset_collision_clearance is None:
            raise ValueError(
                "Position reset randomization requires a rotation-reset asset with per-state collision clearances."
            )

    def _legacy_single_goal_catalog(self) -> dict[str, np.ndarray]:
        """Promote the old compact asset to the validated catalog interface."""

        goal_tcp_position, goal_tcp_quaternion_xyzw = grasp_pose_to_tcp_pose(
            self.cfg.goal_grasp_position_w,
            self.cfg.goal_grasp_orientation_xyzw,
            grasp_to_tcp_quat_wxyz=self.context._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=self.context._TCP_TO_GRASP_CENTER_OFFSET,
        )
        legacy_path = Path(self.cfg.legacy_goal_reset_data_path).expanduser()
        with np.load(legacy_path, allow_pickle=False) as source:
            if "reset_joint_trajectory" not in source:
                raise ValueError(
                    f"{legacy_path} does not contain reset_joint_trajectory; rebuild it with "
                    "isaac_rl/scripts/build_reset_trajectory_asset.py."
                )
            goal_rgb = source["rgb_goal"].copy()
            goal_depth = source["depth_goal"].copy()
            trajectory = source["reset_joint_trajectory"].copy()
            grasp_id = str(source.get("reset_source_grasp_id", np.asarray("g1973")))
        return {
            "target_ids": np.asarray([f"current__{grasp_id}"]),
            "orientation_names": np.asarray(["current"]),
            "orientation_ids": np.asarray(["current"]),
            "orientation_indices": np.asarray([0], dtype=np.int64),
            "grasp_ids": np.asarray([grasp_id]),
            "goal_rgb": goal_rgb[None, ...],
            "goal_depth": goal_depth[None, ...],
            "object_positions_w": np.asarray([self.cfg.object_position_w], dtype=np.float32),
            "object_orientations_xyzw_w": np.asarray([self.cfg.object_orientation_xyzw], dtype=np.float32),
            "goal_grasp_positions_w": np.asarray([self.cfg.goal_grasp_position_w], dtype=np.float32),
            "goal_grasp_orientations_xyzw_w": np.asarray([self.cfg.goal_grasp_orientation_xyzw], dtype=np.float32),
            "goal_tcp_positions_w": np.asarray([goal_tcp_position], dtype=np.float32),
            "goal_tcp_orientations_xyzw_w": np.asarray([goal_tcp_quaternion_xyzw], dtype=np.float32),
            "reset_joint_trajectories": trajectory[None, ...],
            "reset_path_progress": np.linspace(0.0, 1.0, trajectory.shape[0], dtype=np.float32),
            "approach_gripper_widths_m": np.asarray([KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M], dtype=np.float32),
            "moveit_plan_validated": np.ones(1, dtype=np.bool_),
            "isaac_goal_rgbd_captured": np.ones(1, dtype=np.bool_),
        }

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        configured_paths = tuple(str(value) for value in self.cfg.part_usd_paths)
        configured_names = tuple(str(value) for value in self.cfg.part_names)
        if len(configured_paths) != len(configured_names) or not configured_paths:
            raise ValueError("part_names and part_usd_paths must have the same non-zero length.")
        self.parts: list[RigidObject] = []
        for part_index, (part_name, raw_path) in enumerate(zip(configured_names, configured_paths, strict=True)):
            part_path = Path(raw_path).expanduser().resolve()
            if not part_path.is_file():
                raise FileNotFoundError(
                    f"Part USD for catalog part '{part_name}' does not exist: {part_path}. "
                    "Complete the deferred Isaac asset/capture stage first."
                )
            part_cfg = deepcopy(self.cfg.part_cfg)
            part_cfg.prim_path = (
                "/World/envs/env_.*/Part" if len(configured_paths) == 1 else f"/World/envs/env_.*/Part_{part_index}"
            )
            part_cfg.spawn.usd_path = str(part_path)
            self.parts.append(RigidObject(part_cfg))
        self.part = self.parts[0]
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.debug_camera = TiledCamera(self.cfg.debug_camera) if self.cfg.debug_camera_enabled else None
        self.gripper_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=("/World/envs/env_.*/Robot/(gripper_base_link|left_finger_link|right_finger_link)"),
                update_period=0.0,
                history_length=1,
                debug_vis=False,
            )
        )
        ground_cfg = sim_utils.GroundPlaneCfg(
            func=_spawn_local_ground_plane,
            color=VISUAL_SERVO_GROUND_COLOR,
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        for part_index, part in enumerate(self.parts):
            key = "part" if len(self.parts) == 1 else f"part_{part_index}"
            self.scene.rigid_objects[key] = part
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["gripper_contact"] = self.gripper_contact_sensor
        if self.debug_camera is not None:
            self.scene.sensors["debug_camera"] = self.debug_camera
        self.visual_light_paths = spawn_visual_servo_lights()

    def _tcp_error(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tcp_position, tcp_quaternion = self.context.get_tcp_pose_w()
        position_error = self.goal_tcp_position - tcp_position
        quaternion_error = quat_mul(self.goal_tcp_quaternion, quat_conjugate(tcp_quaternion))
        quaternion_error = torch.where(quaternion_error[:, :1] < 0.0, -quaternion_error, quaternion_error)
        rotation_error = 2.0 * quaternion_error[:, 1:4]
        return tcp_position, tcp_quaternion, position_error, rotation_error

    def _rotation_world_from_camera(self, tcp_quaternion: torch.Tensor) -> torch.Tensor:
        """Return the calibrated camera orientation in world coordinates."""

        return torch.bmm(
            matrix_from_quat(tcp_quaternion),
            self.rotation_tcp_from_camera.expand(tcp_quaternion.shape[0], -1, -1),
        )

    def _gripper_contact_force(self) -> torch.Tensor:
        """Return maximum external hand/finger contact force per environment."""

        net_forces_w = self.gripper_contact_sensor.data.net_forces_w
        return torch.linalg.norm(net_forces_w, dim=-1).amax(dim=-1)

    def _gripper_collision(self) -> torch.Tensor:
        """Return environments with unsafe contact on the hand or fingers."""

        return self._gripper_contact_force() >= float(self.cfg.unsafe_contact_force_threshold_n)

    def _curriculum(self):
        state = curriculum_state(
            self.common_step_counter,
            enabled=bool(self.cfg.training_curriculum_enabled),
            warmup_steps=int(self.cfg.curriculum_warmup_steps),
            full_steps=int(self.cfg.curriculum_full_steps),
            initial_progress_min=float(self.cfg.curriculum_initial_progress_min),
            final_progress_min=float(self.cfg.curriculum_final_progress_min),
            final_failure_replay_fraction=float(self.cfg.failure_replay_fraction),
        )
        self.curriculum_fraction = state.fraction
        self.curriculum_progress_min = state.progress_min
        self.curriculum_perturbation_scale = state.perturbation_scale
        self.curriculum_visual_strength = state.visual_randomization_strength
        return state

    def _timed_out(self) -> torch.Tensor:
        return self.episode_length_buf >= self.reset_timeout_steps - 1

    def _tcp_speed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world-frame linear/angular speeds of the TCP parent link."""

        body_velocity = self.robot.data.body_link_vel_w[:, self.context.ee_body_idx]
        return (
            torch.linalg.norm(body_velocity[:, :3], dim=-1),
            torch.linalg.norm(body_velocity[:, 3:], dim=-1),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        curriculum = self._curriculum()
        if self.scene_appearance_randomizer is not None:
            self.scene_appearance_randomizer.maybe_randomize(
                self.common_step_counter,
                strength=curriculum.visual_randomization_strength,
            )
        if actions.shape[-1] != 7:
            raise ValueError(f"Expected six motion actions plus completion, got shape {tuple(actions.shape)}.")
        requested_actions = actions[:, :6].clamp(-1.0, 1.0)
        action_delta = (requested_actions - self.previous_actions).clamp(
            -float(self.cfg.action_delta_limit),
            float(self.cfg.action_delta_limit),
        )
        self.actions = self.previous_actions + action_delta
        self.completion_probability.copy_(actions[:, 6].clamp(0.0, 1.0))
        self.completion_stop_candidate.copy_(
            completion_declared(
                self.completion_probability,
                threshold=float(self.cfg.completion_probability_threshold),
            )
        )
        linear_speed, angular_speed = self._tcp_speed()
        stable = (linear_speed <= float(self.cfg.completion_max_linear_speed_m_s)) & (
            angular_speed <= float(self.cfg.completion_max_angular_speed_rad_s)
        )
        streak_probability = torch.where(
            self.completion_stop_candidate & stable,
            self.completion_probability,
            torch.zeros_like(self.completion_probability),
        )
        self.completion_streak.copy_(
            update_completion_streak(
                self.completion_streak,
                streak_probability,
                probability_threshold=float(self.cfg.completion_probability_threshold),
            )
        )
        self.completion_declaration.copy_(self.completion_streak >= int(self.cfg.completion_required_consecutive_steps))

    def _apply_action(self) -> None:
        _, tcp_quaternion, _, _ = self._tcp_error()
        rotation_world_from_camera = self._rotation_world_from_camera(tcp_quaternion)
        twist_camera = self.actions.clone()
        # A high completion probability means "hold" immediately. Playback
        # then waits for the velocity/consecutive-frame gate before declaring
        # the episode complete.
        twist_camera[self.completion_stop_candidate] = 0.0
        twist_camera[:, :3] *= self.cfg.linear_action_scale_m_s
        twist_camera[:, 3:] *= self.cfg.angular_action_scale_rad_s
        twist_world = torch.cat(
            (
                torch.bmm(rotation_world_from_camera, twist_camera[:, :3, None]).squeeze(-1),
                torch.bmm(rotation_world_from_camera, twist_camera[:, 3:, None]).squeeze(-1),
            ),
            dim=-1,
        )

        root_quaternion = self.robot.data.root_quat_w
        rotation_base_from_world = matrix_from_quat(quat_conjugate(root_quaternion))
        twist_base = torch.cat(
            (
                torch.bmm(rotation_base_from_world, twist_world[:, :3, None]).squeeze(-1),
                torch.bmm(rotation_base_from_world, twist_world[:, 3:, None]).squeeze(-1),
            ),
            dim=-1,
        )
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.context.ee_jacobi_body_idx, :, self.arm_ids]
        transpose = jacobian.transpose(1, 2)
        identity = torch.eye(6, device=self.device).expand(self.num_envs, -1, -1)
        q_dot = torch.bmm(
            transpose,
            torch.linalg.solve(
                torch.bmm(jacobian, transpose) + self.cfg.dls_damping**2 * identity,
                twist_base.unsqueeze(-1),
            ),
        ).squeeze(-1)
        q = self.robot.data.joint_pos[:, self.arm_ids]
        q_target = q + q_dot * self.step_dt
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        q_target = torch.max(torch.min(q_target, limits[..., 1]), limits[..., 0])
        self.robot.set_joint_position_target(q_target, joint_ids=self.arm_ids)
        self.context.command_fixed_gripper()
        self.previous_actions.copy_(self.actions)

    def _camera_observation(self, *, randomize_live: bool = True) -> torch.Tensor:
        output = self.wrist_camera.data.output
        rgb = output["rgb"][..., :3].float().div(255.0)
        depth = torch.nan_to_num(output["distance_to_image_plane"].float(), nan=0.50, posinf=0.50, neginf=0.04)
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        observation_size = (
            VISUAL_SERVO_OBSERVATION_HEIGHT,
            VISUAL_SERVO_OBSERVATION_WIDTH,
        )
        rgb = F.interpolate(rgb.permute(0, 3, 1, 2), size=observation_size, mode="area").permute(0, 2, 3, 1)
        depth = F.interpolate(depth.permute(0, 3, 1, 2), size=observation_size, mode="area").permute(0, 2, 3, 1)
        if randomize_live and self.live_observation_randomizer is not None:
            rgb, depth = self.live_observation_randomizer.apply(rgb, depth)
        rgbd = torch.cat((rgb, depth.sub(0.04).div(0.46).clamp(0.0, 1.0)), dim=-1)
        return torch.cat((rgbd, self.goal_rgbd), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        _, tcp_quaternion, position_error, rotation_error = self._tcp_error()
        q = self.robot.data.joint_pos[:, self.arm_ids]
        qd = self.robot.data.joint_vel[:, self.arm_ids]
        state = torch.cat((q, qd, position_error, rotation_error, self.previous_actions), dim=-1)
        visual_observation = self._camera_observation().flatten(start_dim=1)
        # The preceding action is deployment-available policy context. The
        # normalized pose/completion values are privileged labels: RL-Games
        # stores them in the rollout, but the custom actor slices them away
        # before computing any action.
        position_error_camera, rotation_error_camera = world_pose_error_to_camera(
            position_error,
            rotation_error,
            self._rotation_world_from_camera(tcp_quaternion),
        )
        pose_target = torch.cat(
            (
                position_error_camera / self.cfg.auxiliary_position_scale_m,
                rotation_error_camera / self.cfg.auxiliary_rotation_scale_rad,
            ),
            dim=-1,
        )
        position_norm = torch.linalg.norm(position_error, dim=-1)
        rotation_norm = torch.linalg.norm(rotation_error, dim=-1)
        labels = completion_masks(
            position_norm,
            rotation_norm,
            ready_position_m=float(self.cfg.completion_ready_position_m),
            ready_rotation_rad=float(self.cfg.completion_ready_rotation_rad),
            negative_position_m=float(self.cfg.completion_negative_position_m),
            negative_rotation_rad=float(self.cfg.completion_negative_rotation_rad),
            collision_free=~self._gripper_collision(),
        )
        completion_target = torch.stack((labels.ready.float(), labels.supervised.float()), dim=-1)
        policy_observation = torch.cat(
            (
                visual_observation,
                self.previous_actions,
                pose_target,
                completion_target,
            ),
            dim=-1,
        )
        return {"policy": policy_observation, "critic": state}

    def _get_rewards(self) -> torch.Tensor:
        _, _, position_error, rotation_error = self._tcp_error()
        position_norm = torch.linalg.norm(position_error, dim=-1)
        rotation_norm = torch.linalg.norm(rotation_error, dim=-1)
        contact_force = self._gripper_contact_force()
        collision = contact_force >= float(self.cfg.unsafe_contact_force_threshold_n)
        labels = completion_masks(
            position_norm,
            rotation_norm,
            ready_position_m=float(self.cfg.completion_ready_position_m),
            ready_rotation_rad=float(self.cfg.completion_ready_rotation_rad),
            negative_position_m=float(self.cfg.completion_negative_position_m),
            negative_rotation_rad=float(self.cfg.completion_negative_rotation_rad),
            collision_free=~collision,
        )
        correct_completion = self.completion_declaration & labels.ready
        premature_completion = self.completion_declaration & ~labels.ready
        diverged = position_norm > self.cfg.divergence_position_m
        timed_out = self._timed_out()
        failed_timeout = timed_out & ~self.completion_declaration & ~diverged & ~collision

        position_progress = self.previous_position_error - position_norm
        rotation_progress = self.previous_rotation_error - rotation_norm
        previous_position_precision = torch.exp(-self.previous_position_error / self.cfg.position_precision_scale_m)
        position_precision = torch.exp(-position_norm / self.cfg.position_precision_scale_m)
        previous_rotation_precision = torch.exp(-self.previous_rotation_error / self.cfg.rotation_precision_scale_rad)
        rotation_precision = torch.exp(-rotation_norm / self.cfg.rotation_precision_scale_rad)

        # All dense terms are differences of potentials. Holding position no
        # longer accumulates positive reward over a timeout-length episode.
        reward = self.cfg.position_progress_weight * position_progress
        reward += self.cfg.rotation_progress_weight * rotation_progress
        reward += self.cfg.position_precision_weight * (position_precision - previous_position_precision)
        reward += self.cfg.rotation_precision_weight * (rotation_precision - previous_rotation_precision)
        terminal_completion_reward = completion_terminal_reward(
            self.completion_declaration,
            labels.ready,
            correct_reward=float(self.cfg.completion_correct_reward),
            premature_penalty=float(self.cfg.completion_premature_penalty),
        )
        positive_success = correct_completion & self.completion_positive_reset
        terminal_completion_reward = torch.where(
            positive_success,
            terminal_completion_reward * float(self.cfg.completion_positive_terminal_reward_scale),
            terminal_completion_reward,
        )
        reward += terminal_completion_reward
        reward -= self.cfg.timeout_penalty * failed_timeout.float()
        reward -= self.cfg.divergence_penalty * diverged.float()
        reward -= self.cfg.unsafe_collision_penalty * collision.float()
        risk_denominator = max(
            float(self.cfg.unsafe_contact_force_threshold_n) - float(self.cfg.collision_risk_force_threshold_n),
            1.0e-6,
        )
        collision_risk = (
            ((contact_force - float(self.cfg.collision_risk_force_threshold_n)) / risk_denominator)
            .clamp(0.0, 1.0)
            .square()
        )
        reward -= self.cfg.collision_risk_penalty_weight * collision_risk
        reward -= self.cfg.step_penalty
        action_norm = torch.linalg.norm(self.actions, dim=-1)
        reward -= self.cfg.action_penalty_weight * action_norm.square()
        self.previous_position_error.copy_(position_norm)
        self.previous_rotation_error.copy_(rotation_norm)
        terminal = self.completion_declaration | diverged | collision | timed_out
        failure_value = failed_timeout.float()
        failure_value = torch.maximum(failure_value, premature_completion.float())
        failure_value = torch.maximum(failure_value, 1.25 * diverged.float())
        failure_value = torch.maximum(failure_value, 1.50 * collision.float())
        failure_value = torch.where(correct_completion, torch.zeros_like(failure_value), failure_value)
        self.target_failure_scores = update_failure_scores(
            self.target_failure_scores,
            target_indices=self.target_index,
            terminal_mask=terminal,
            failure_values=failure_value,
            decay=float(self.cfg.failure_score_decay),
        )
        if terminal.any():
            self.target_terminal_counts.scatter_add_(
                0,
                self.target_index[terminal],
                torch.ones_like(self.target_index[terminal]),
            )
        log = {
            "position_error_mm": position_norm.mean() * 1000.0,
            "rotation_error_deg": rotation_norm.mean() * 180.0 / torch.pi,
            "success_rate": correct_completion.float().mean(),
            "completion/geometric_ready_rate": labels.ready.float().mean(),
            "completion/declaration_rate": self.completion_declaration.float().mean(),
            "completion/premature_rate": premature_completion.float().mean(),
            "completion/missed_ready_rate": (labels.ready & ~self.completion_declaration).float().mean(),
            # Stochastic PPO rollouts send the Bernoulli draw (0/1), so this
            # is an unbiased batch estimate of mean p(done). Deterministic
            # playback sends the probability itself.
            "completion/stop_signal_mean": self.completion_probability.mean(),
            "collision_rate": collision.float().mean(),
            "collision/contact_force_n": contact_force.mean(),
            "collision/risk_mean": collision_risk.mean(),
            "timeout_rate": failed_timeout.float().mean(),
            "action_norm": action_norm.mean(),
            "reset_progress": self.reset_progress.mean(),
            "reset_noise_rad": self.reset_noise_scale.mean(),
            "reset_rotation_command_deg": self.reset_rotation_command.mean() * 180.0 / torch.pi,
            "reset_position_offset_mm": torch.linalg.norm(self.reset_position_offset, dim=-1).mean() * 1000.0,
            "reset_position_requested_mm": self.reset_position_requested.mean() * 1000.0,
            "reset_position_capped_rate": (
                torch.linalg.norm(self.reset_position_offset, dim=-1) + 1.0e-9 < self.reset_position_requested
            )
            .float()
            .mean(),
            "initial_position_error_mm": self.initial_position_error.mean() * 1000.0,
            "initial_rotation_error_deg": self.initial_rotation_error.mean() * 180.0 / torch.pi,
            "completion/positive_reset_rate": self.completion_positive_reset.float().mean(),
            "completion/exact_reset_rate": self.completion_exact_reset.float().mean(),
            "reset/timeout_s": self.reset_timeout_s.mean(),
            "reset/no_noise_rate": (self.reset_mode == RESET_MODE_NO_NOISE).float().mean(),
            "reset/boundary_rate": (self.reset_mode == RESET_MODE_BOUNDARY).float().mean(),
            "reset/failure_replay_rate": self.reset_failure_replay.float().mean(),
            "curriculum/fraction": position_norm.new_tensor(self.curriculum_fraction),
            "curriculum/progress_min": position_norm.new_tensor(self.curriculum_progress_min),
            "curriculum/perturbation_scale": position_norm.new_tensor(self.curriculum_perturbation_scale),
            "curriculum/visual_strength": position_norm.new_tensor(self.curriculum_visual_strength),
            "failure_sampling/score_mean": self.target_failure_scores.mean(),
            "failure_sampling/score_max": self.target_failure_scores.max(),
            "target_coverage_fraction": position_norm.new_tensor(
                torch.unique(self.target_index).numel() / float(self.target_count)
            ),
        }
        if self.scene_appearance_randomizer is not None and self.scene_appearance_randomizer.current_sample is not None:
            appearance = self.scene_appearance_randomizer.current_sample
            log.update(
                {
                    "appearance/key_yaw_delta_deg": position_norm.new_tensor(appearance.key_yaw_delta_deg),
                    "appearance/key_pitch_delta_deg": position_norm.new_tensor(appearance.key_pitch_delta_deg),
                    "appearance/key_intensity": position_norm.new_tensor(appearance.key_intensity),
                    "appearance/key_angle_deg": position_norm.new_tensor(appearance.key_angle_deg),
                    "appearance/dome_intensity": position_norm.new_tensor(appearance.dome_intensity),
                }
            )
        # Preserve per-environment terminal measurements for batched policy
        # evaluation. DirectRLEnv resets completed environments before it
        # returns from step(), so these must be cloned before _reset_idx()
        # changes their target and initial-state buffers. Keeping this outside
        # ``log`` also prevents 50-element tensors from becoming TensorBoard
        # scalars during training.
        self.extras["evaluation"] = {
            "target_index": self.target_index.clone(),
            "reset_progress": self.reset_progress.clone(),
            "reset_noise_rad": self.reset_noise_scale.clone(),
            "reset_rotation_command_rad": self.reset_rotation_command.clone(),
            "reset_position_offset_w": self.reset_position_offset.clone(),
            "reset_position_requested_m": self.reset_position_requested.clone(),
            "reset_position_safe_cap_m": self.reset_position_safe_cap.clone(),
            "completion_positive_reset": self.completion_positive_reset.clone(),
            "completion_exact_reset": self.completion_exact_reset.clone(),
            "reset_mode": self.reset_mode.clone(),
            "reset_timeout_s": self.reset_timeout_s.clone(),
            "reset_failure_replay": self.reset_failure_replay.clone(),
            "initial_position_error_m": self.initial_position_error.clone(),
            "initial_rotation_error_rad": self.initial_rotation_error.clone(),
            "position_error_m": position_norm.clone(),
            "rotation_error_rad": rotation_norm.clone(),
            "success": correct_completion.clone(),
            "geometric_ready": labels.ready.clone(),
            "completion_probability": self.completion_probability.clone(),
            "completion_declared": self.completion_declaration.clone(),
            "premature_completion": premature_completion.clone(),
            "collision": collision.clone(),
            "contact_force_n": contact_force.clone(),
            "collision_risk": collision_risk.clone(),
            "diverged": diverged.clone(),
            "timed_out": failed_timeout.clone(),
        }
        # Per-orientation plots are useful for the compact five-orientation
        # task but would add 68 mostly noisy curves for the assembly catalog.
        # Multi-part training keeps the smaller per-part breakdown here; the
        # evaluator still writes full per-orientation held-out metrics.
        if self.target_count > 1 and len(self.orientation_names) <= 8:
            selected_orientation = self.target_orientation_indices[self.target_index]
            for orientation_index, orientation_name in enumerate(self.orientation_names):
                mask = selected_orientation == orientation_index
                sample_count = mask.sum().clamp_min(1)
                log[f"orientation/{orientation_name}_position_error_mm"] = (
                    position_norm.masked_fill(~mask, 0.0).sum() / sample_count * 1000.0
                )
                log[f"orientation/{orientation_name}_success_rate"] = (
                    correct_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
                )
        if len(self.part_names) > 1:
            selected_part = self.target_part_indices[self.target_index]
            for part_index, part_name in enumerate(self.part_names):
                mask = selected_part == part_index
                sample_count = mask.sum().clamp_min(1)
                log[f"part/{part_name}_position_error_mm"] = (
                    position_norm.masked_fill(~mask, 0.0).sum() / sample_count * 1000.0
                )
                log[f"part/{part_name}_success_rate"] = (
                    correct_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
                )
        # Aggregate curves can improve merely because a policy solves the
        # close resets. Keep the three approach regions separate so far-range
        # behavior remains visible as the curriculum expands.
        for bucket, mask in approach_progress_bucket_masks(self.reset_progress).items():
            sample_count = mask.sum().clamp_min(1)
            log[f"{bucket}/position_error_mm"] = position_norm.masked_fill(~mask, 0.0).sum() / sample_count * 1000.0
            log[f"{bucket}/rotation_error_deg"] = (
                rotation_norm.masked_fill(~mask, 0.0).sum() / sample_count * 180.0 / torch.pi
            )
            log[f"{bucket}/success_rate"] = correct_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
        self.extras["log"] = log
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, position_error, _ = self._tcp_error()
        position_norm = torch.linalg.norm(position_error, dim=-1)
        diverged = position_norm > self.cfg.divergence_position_m
        collision = self._gripper_collision()
        timed_out = self._timed_out()
        return self.completion_declaration | diverged | collision, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:  # noqa: C901
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if self.fixed_target_index >= 0:
            target_indices = torch.full(
                (count,),
                self.fixed_target_index,
                dtype=torch.long,
                device=self.device,
            )
            unit_progress = (torch.arange(count, dtype=torch.float32, device=self.device) + 0.5) / max(count, 1)
        elif self.cfg.sequential_target_sampling:
            # Full resets assign one environment per catalog target. Automatic
            # partial resets after termination retain that environment's target;
            # the evaluator masks it out, but Isaac still requires a valid reset.
            if count == self.target_count:
                target_indices = torch.arange(count, dtype=torch.long, device=self.device)
            else:
                target_indices = self.target_index[env_ids].clone()
            unit_progress = (torch.arange(count, dtype=torch.float32, device=self.device) + 0.5) / max(count, 1)
        elif self.cfg.random_target_sampling:
            # Playback-only independent draws: every episode gets a newly
            # sampled grasp/object-orientation target, while repeats remain
            # possible. Training deliberately keeps the balanced branches.
            target_indices = torch.randint(self.target_count, (count,), dtype=torch.long, device=self.device)
            unit_progress = torch.rand(count, dtype=torch.float32, device=self.device)
            self.target_sample_counts.scatter_add_(0, target_indices, torch.ones_like(target_indices))
        elif len(self.part_names) > 1:
            target_indices, unit_progress, self.part_cursor = balanced_group_target_progress(
                target_group_indices=self.target_part_indices,
                sample_count=count,
                group_cursor=self.part_cursor,
                group_target_cursors=self.part_target_cursors,
                target_sample_counts=self.target_sample_counts,
            )
        else:
            target_indices, unit_progress, self.target_cursor = balanced_target_progress(
                target_count=self.target_count,
                sample_count=count,
                target_cursor=self.target_cursor,
                target_sample_counts=self.target_sample_counts,
            )
        curriculum = self._curriculum()
        replay_mask = torch.zeros(count, dtype=torch.bool, device=self.device)
        balanced_training = (
            self.fixed_target_index < 0
            and not self.cfg.sequential_target_sampling
            and not self.cfg.random_target_sampling
        )
        if balanced_training and curriculum.failure_replay_fraction > 0.0:
            original_targets = target_indices.clone()
            target_indices, replay_mask = apply_failure_replay(
                target_indices,
                target_group_indices=self.target_part_indices,
                failure_scores=self.target_failure_scores,
                replay_fraction=curriculum.failure_replay_fraction,
                score_floor=float(self.cfg.failure_replay_score_floor),
                score_power=float(self.cfg.failure_replay_score_power),
            )
            if replay_mask.any():
                self.target_sample_counts.add_(
                    torch.bincount(target_indices[replay_mask], minlength=self.target_count)
                    - torch.bincount(original_targets[replay_mask], minlength=self.target_count)
                )
                unit_progress[replay_mask] = torch.rand(int(replay_mask.sum()), device=self.device)

        # Randomize which cloned environment receives each paired target and
        # progress sample without destroying their balanced joint schedule.
        permutation = torch.randperm(count, device=self.device)
        target_indices = target_indices[permutation]
        unit_progress = unit_progress[permutation]
        replay_mask = replay_mask[permutation]
        progress_min = max(
            float(self.cfg.reset_progress_min),
            float(curriculum.progress_min),
        )
        progress = progress_min + unit_progress * (self.cfg.reset_progress_max - progress_min)
        if self.cfg.training_reset_mixture_enabled:
            reset_modes = sample_reset_modes(
                torch.rand(count, device=self.device),
                no_noise_fraction=float(self.cfg.reset_no_noise_fraction),
                ready_fraction=float(self.cfg.reset_ready_fraction),
                boundary_fraction=float(self.cfg.reset_boundary_fraction),
            )
        else:
            positive_fraction = float(self.cfg.completion_positive_reset_fraction)
            if not 0.0 <= positive_fraction <= 1.0:
                raise ValueError("completion_positive_reset_fraction must lie in [0, 1].")
            positive_reset_draw = torch.rand(count, device=self.device) < positive_fraction
            reset_modes = torch.where(
                positive_reset_draw,
                torch.full((count,), RESET_MODE_READY, device=self.device),
                torch.full((count,), RESET_MODE_PATH, device=self.device),
            ).long()
        positive_reset = reset_modes == RESET_MODE_READY
        exact_reset = positive_reset & (
            torch.rand(count, device=self.device) < float(self.cfg.reset_ready_exact_fraction)
        )
        boundary_reset = reset_modes == RESET_MODE_BOUNDARY
        boundary_rotation = boundary_reset & (
            torch.rand(count, device=self.device) < float(self.cfg.reset_boundary_rotation_fraction)
        )
        if positive_reset.any() or boundary_rotation.any():
            goal_progress = self.reset_path_progress_catalog[-1]
            progress = torch.where(
                positive_reset | boundary_rotation,
                goal_progress,
                progress,
            )
        collision_safe_sampling = bool(getattr(self.cfg, "reset_collision_safe_sampling_enabled", False))
        progress_indices: torch.Tensor | None = None
        if collision_safe_sampling:
            if (
                abs(float(self.cfg.reset_joint_noise_far_rad)) > 1.0e-12
                or abs(float(self.cfg.reset_joint_noise_near_rad)) > 1.0e-12
            ):
                raise ValueError(
                    "Collision-safe reset sampling requires zero joint noise; "
                    "only authored and validated states may initialize an episode."
                )
            progress_indices = torch.argmin(
                torch.abs(self.reset_path_progress_catalog.unsqueeze(0) - progress.unsqueeze(1)),
                dim=1,
            )
            final_index = self.reset_path_progress_catalog.numel() - 1
            progress_indices[positive_reset | boundary_rotation] = final_index
            boundary_path = boundary_reset & ~boundary_rotation
            if boundary_path.any():
                boundary_count = min(
                    int(self.cfg.reset_boundary_waypoint_count),
                    max(1, final_index),
                )
                progress_indices[boundary_path] = final_index - torch.randint(
                    1,
                    boundary_count + 1,
                    (int(boundary_path.sum()),),
                    device=self.device,
                )
            progress = self.reset_path_progress_catalog[progress_indices]

        self.target_index[env_ids] = target_indices
        self.goal_tcp_quaternion[env_ids] = self.goal_tcp_quaternions_catalog[target_indices]
        self.goal_rgbd[env_ids] = self.goal_rgbd_catalog[target_indices]

        if collision_safe_sampling:
            q = self.reset_joint_trajectories[target_indices, progress_indices]
        else:
            q = interpolate_joint_trajectory(self.reset_joint_trajectories[target_indices], progress)
        rotation_command = torch.zeros(count, dtype=torch.float32, device=self.device)
        variant_indices: torch.Tensor | None = None
        if self.rotation_reset_joint_trajectories is not None:
            if not (0.0 <= self.cfg.reset_rotation_fraction_min <= self.cfg.reset_rotation_fraction_max <= 1.0):
                raise ValueError("Rotation reset fractions must satisfy 0 <= min <= max <= 1.")
            variant_count = self.rotation_reset_joint_trajectories.shape[1]
            variant_indices = torch.randint(variant_count, (count,), device=self.device)
            rotated_path = self.rotation_reset_joint_trajectories[target_indices, variant_indices]
            if collision_safe_sampling:
                minimum_fraction = float(self.cfg.reset_rotation_fraction_min)
                maximum_fraction = float(self.cfg.reset_rotation_fraction_max)
                fixed_fraction = minimum_fraction
                if (
                    abs(minimum_fraction - maximum_fraction) >= 1.0e-12
                    or min(abs(fixed_fraction), abs(fixed_fraction - 1.0)) >= 1.0e-12
                ):
                    raise ValueError(
                        "Collision-safe reset sampling requires a fixed rotation_fraction "
                        "of 0 or 1; fractional joint interpolation was not collision validated."
                    )
                row_indices = torch.arange(count, device=self.device)
                rotated_q = rotated_path[row_indices, progress_indices]
                rotation_fraction = torch.full((count,), fixed_fraction, device=self.device)
                authored_angle = self.rotation_reset_angle_profile[progress_indices]
                if self.cfg.training_reset_mixture_enabled:
                    rotate_path = (reset_modes == RESET_MODE_PATH) & (
                        torch.rand(count, device=self.device) < curriculum.perturbation_scale
                    )
                    apply_authored_rotation = rotate_path | boundary_rotation
                    rotation_fraction = apply_authored_rotation.float()
            else:
                rotated_q = interpolate_joint_trajectory(rotated_path, progress)
                rotation_fraction = torch.empty(count, device=self.device).uniform_(
                    self.cfg.reset_rotation_fraction_min,
                    self.cfg.reset_rotation_fraction_max,
                )
                authored_angle = interpolate_joint_trajectory(
                    self.rotation_reset_angle_profile.unsqueeze(-1), progress
                ).squeeze(-1)
            # Ready and explicit no-noise states use the nominal robot pose.
            # Boundary rotation states deliberately retain the fully validated
            # final-waypoint rotation (five degrees in the current asset).
            rotation_fraction[positive_reset] = 0.0
            q = torch.lerp(q, rotated_q, rotation_fraction.unsqueeze(-1))
            rotation_command = authored_angle * rotation_fraction

        position_offset = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
        position_requested = torch.zeros(count, dtype=torch.float32, device=self.device)
        position_safe_cap = torch.zeros(count, dtype=torch.float32, device=self.device)
        if self.cfg.reset_position_randomization_enabled:
            if not collision_safe_sampling or progress_indices is None or variant_indices is None:
                raise ValueError(
                    "Position reset randomization requires exact collision-validated "
                    "waypoint and rotation-variant sampling."
                )
            if self.rotation_reset_collision_clearance is None or self.nominal_reset_collision_clearance is None:
                raise RuntimeError("Rotation and nominal reset clearances were not loaded.")
            rotated_clearance = self.rotation_reset_collision_clearance[
                target_indices, variant_indices, progress_indices
            ]
            nominal_clearance = self.nominal_reset_collision_clearance[target_indices, progress_indices]
            selected_clearance = torch.where(
                rotation_fraction > 0.5,
                rotated_clearance,
                nominal_clearance,
            )
            requested_profile = position_offset_profile(
                progress,
                far_offset_m=float(self.cfg.reset_position_far_offset_m),
                near_offset_m=float(self.cfg.reset_position_near_offset_m),
                exponent=float(self.cfg.reset_position_offset_exponent),
            )
            magnitude_samples = torch.empty(count, device=self.device).uniform_(
                float(self.cfg.reset_position_fraction_min),
                float(self.cfg.reset_position_fraction_max),
            )
            zero_offset = positive_reset
            if self.cfg.training_reset_mixture_enabled:
                path_reset = reset_modes == RESET_MODE_PATH
                requested_profile = torch.where(
                    path_reset,
                    requested_profile * curriculum.perturbation_scale,
                    torch.zeros_like(requested_profile),
                )
                requested_profile = torch.where(
                    positive_reset,
                    requested_profile.new_full((), float(self.cfg.reset_ready_position_max_m)),
                    requested_profile,
                )
                magnitude_samples[positive_reset] = torch.rand(int(positive_reset.sum()), device=self.device)
                zero_offset = (reset_modes == RESET_MODE_NO_NOISE) | boundary_reset | exact_reset
            position_offset, position_requested, position_safe_cap = sample_collision_safe_xy_offsets_from_profile(
                requested_profile,
                selected_clearance,
                zero_offset,
                minimum_collision_clearance_m=(self.rotation_reset_minimum_collision_clearance_m),
                clearance_guard_m=float(self.cfg.reset_position_clearance_guard_m),
                magnitude_unit_samples=magnitude_samples,
            )

        # Shift the active object and its final TCP goal together. The robot
        # remains at the validated nominal/rotated waypoint, creating a true
        # off-path Cartesian error while preserving the canonical goal image.
        self.goal_tcp_position[env_ids] = (
            self.goal_tcp_positions_catalog[target_indices] + position_offset + self.scene.env_origins[env_ids]
        )
        object_pose = torch.cat(
            (
                self.object_positions_catalog[target_indices] + position_offset + self.scene.env_origins[env_ids],
                self.object_quaternions_catalog[target_indices],
            ),
            dim=-1,
        )
        selected_part_indices = self.target_part_indices[target_indices]
        zero_velocity = torch.zeros((count, 6), dtype=torch.float32, device=self.device)
        parked_pose = torch.zeros((count, 7), dtype=torch.float32, device=self.device)
        parked_pose[:, :3] = self.scene.env_origins[env_ids]
        parked_pose[:, 2] -= 10.0
        parked_pose[:, 3] = 1.0
        for part_index, part in enumerate(self.parts):
            part_pose = parked_pose.clone()
            active = selected_part_indices == part_index
            part_pose[active] = object_pose[active]
            part.write_root_pose_to_sim(part_pose, env_ids=env_ids)
            if hasattr(part, "write_root_velocity_to_sim"):
                part.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        noise_scale = path_conditioned_noise_scale(
            progress,
            far_scale=self.cfg.reset_joint_noise_far_rad,
            near_scale=self.cfg.reset_joint_noise_near_rad,
            exponent=self.cfg.reset_joint_noise_exponent,
        )
        noise_scale[positive_reset] = 0.0
        if self.cfg.training_reset_mixture_enabled:
            noise_scale[reset_modes != RESET_MODE_PATH] = 0.0
        if not collision_safe_sampling:
            q += torch.empty_like(q).uniform_(-1.0, 1.0) * noise_scale.unsqueeze(-1)
        joint_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_ids]
        q = torch.maximum(torch.minimum(q, joint_limits[..., 1]), joint_limits[..., 0])
        qd = torch.zeros_like(q)
        self.context.set_fixed_gripper_widths(self.approach_gripper_widths_catalog[target_indices], env_ids=env_ids)
        self.context.write_fixed_gripper_state(env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.arm_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.arm_ids, env_ids=env_ids)
        self.previous_actions[env_ids] = 0.0
        self.completion_probability[env_ids] = 0.0
        self.completion_stop_candidate[env_ids] = False
        self.completion_streak[env_ids] = 0
        self.completion_declaration[env_ids] = False
        self.completion_positive_reset[env_ids] = positive_reset
        self.completion_exact_reset[env_ids] = exact_reset
        self.reset_mode[env_ids] = reset_modes
        self.reset_failure_replay[env_ids] = replay_mask
        self.reset_progress[env_ids] = progress
        self.reset_noise_scale[env_ids] = noise_scale
        self.reset_rotation_command[env_ids] = rotation_command
        self.reset_position_offset[env_ids] = position_offset
        self.reset_position_requested[env_ids] = position_requested
        self.reset_position_safe_cap[env_ids] = position_safe_cap
        if self.cfg.variable_reset_timeouts_enabled:
            timeout_s = reset_timeout_seconds(
                progress,
                reset_modes,
                far_seconds=float(self.cfg.reset_timeout_far_s),
                close_seconds=float(self.cfg.reset_timeout_close_s),
                ready_seconds=float(self.cfg.reset_timeout_ready_s),
                boundary_seconds=float(self.cfg.reset_timeout_boundary_s),
                exponent=float(self.cfg.reset_timeout_exponent),
            )
        else:
            timeout_s = progress.new_full((), float(self.cfg.episode_length_s)).expand_as(progress)
        timeout_steps = torch.round(timeout_s / self.step_dt).long().clamp(1, self.max_episode_length)
        self.reset_timeout_s[env_ids] = timeout_steps.float() * self.step_dt
        self.reset_timeout_steps[env_ids] = timeout_steps
        _, _, position_error, rotation_error = self._tcp_error()
        position_norm = torch.linalg.norm(position_error[env_ids], dim=-1)
        rotation_norm = torch.linalg.norm(rotation_error[env_ids], dim=-1)
        self.previous_position_error[env_ids] = position_norm
        self.previous_rotation_error[env_ids] = rotation_norm
        self.initial_position_error[env_ids] = position_norm
        self.initial_rotation_error[env_ids] = rotation_norm
        if self.live_observation_randomizer is not None:
            self.live_observation_randomizer.sample(
                env_ids,
                strength=curriculum.visual_randomization_strength,
            )
