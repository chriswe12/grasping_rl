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
from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    D405WristCameraConfig,
    camera_rotation_in_link7,
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
from grasp_planning.rl.d405_observation import (
    D405ObservationPreprocessCfg,
    pack_policy_rgbd_torch,
    resize_aligned_rgbd_torch,
)
from grasp_planning.rl.goal_live_color import (
    COLOR_RELATIONSHIP_DIFFERENT,
    COLOR_RELATIONSHIP_MATCH,
    COLOR_RELATIONSHIP_SIMILAR,
    sample_goal_live_color_pairs,
)
from grasp_planning.rl.lift_reward import (
    lift_outcome_reward,
    physical_pickup_success,
    retained_lift_quality,
)
from grasp_planning.rl.live_observation_randomization import (
    LiveObservationRandomizationCfg,
    LiveObservationRandomizer,
)
from grasp_planning.rl.policy_context import assemble_policy_context_torch, resolve_policy_context
from grasp_planning.rl.policy_timing import temporal_reward_scale
from grasp_planning.rl.scene_appearance_randomization import (
    SceneAppearanceRandomizationCfg,
    SceneAppearanceRandomizer,
)
from grasp_planning.start_poses import (
    PDZ_GRIPPER_APPROACH_PROFILE,
    PDZ_GRIPPER_CLOSED_WIDTH_M,
    PDZ_GRIPPER_OPEN_WIDTH_M,
)
from grasp_planning.visual_servo_busy_background import spawn_visual_servo_busy_background
from grasp_planning.visual_servo_clutter import spawn_visual_servo_clutter
from grasp_planning.visual_servo_surface_markings import spawn_visual_servo_surface_markings
from grasp_planning.visual_servo_workspace import (
    VISUAL_SERVO_TSLOT_PROFILE,
    LiveWorkspaceAppearanceRandomizer,
    spawn_visual_servo_tslot_surfaces,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul

from .completion import (
    completion_declared,
    completion_masks,
    completion_quality,
    graded_completion_terminal_reward,
    update_completion_streak,
)
from .isaac_rl_env_cfg import GraspVisualServoEnvCfg
from .multigrasp_catalog import (
    load_multigrasp_catalog,
    load_multigrasp_rotation_resets,
    select_catalog_split,
)
from .object_pose_sampling import (
    apply_planar_object_pose_delta,
    sample_collision_safe_yaw_offsets_from_profile,
    yaw_offset_profile,
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

    _LIFT_PHASE_APPROACH = 0
    _LIFT_PHASE_CLOSE = 1
    _LIFT_PHASE_GRAVITY_RELEASE = 2
    _LIFT_PHASE_ASCEND = 3
    _LIFT_PHASE_HOLD = 4
    _LIFT_PHASE_OUTCOME = 5

    def __init__(self, cfg: GraspVisualServoEnvCfg, render_mode: str | None = None, **kwargs):
        if bool(getattr(cfg, "lift_reward_enabled", False)):
            # Lift-aware training needs a dynamic object and stronger gripper
            # drives. Clone the Hydra config so selecting this profile cannot
            # mutate a later baseline environment in the same interpreter.
            cfg = deepcopy(cfg)
            cfg.part_cfg.spawn.rigid_props.kinematic_enabled = False
            cfg.part_cfg.spawn.rigid_props.disable_gravity = True
            cfg.part_cfg.spawn.rigid_props.solver_position_iteration_count = 64
            cfg.part_cfg.spawn.rigid_props.solver_velocity_iteration_count = 4
            cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(cfg.lift_static_friction),
                dynamic_friction=float(cfg.lift_dynamic_friction),
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            )
            # The default GPU patch buffer is too small for 224 lift-aware
            # environments and silently overflows during the initial contact
            # solve.  2**19 covers the measured ~386k-patch peak while keeping
            # substantially more headroom than the default.
            cfg.sim.physx.gpu_max_rigid_patch_count = max(int(cfg.sim.physx.gpu_max_rigid_patch_count), 2**19)
            for actuator_name in ("hand_driver", "hand_follower"):
                actuator = cfg.robot_cfg.actuators.get(actuator_name)
                if actuator is None:
                    raise RuntimeError(f"Lift reward requires PDZ actuator {actuator_name!r}.")
                actuator.effort_limit_sim = float(cfg.lift_hand_effort_limit_n)
                actuator.stiffness = float(cfg.lift_hand_stiffness)
                actuator.damping = float(cfg.lift_hand_damping)
        self.live_observation_randomizer: LiveObservationRandomizer | None = None
        self.scene_appearance_randomizer: SceneAppearanceRandomizer | None = None
        self.live_workspace_appearance_randomizer: LiveWorkspaceAppearanceRandomizer | None = None
        self.visual_light_paths: dict[str, str] = {}
        self.tslot_visual_bindings: dict[str, object] = {}
        self.clutter_visual_bindings: dict[str, object] = {}
        self.busy_background_visual_bindings: dict[str, object] = {}
        self.surface_marking_visual_bindings: dict[str, object] = {}
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
                    gripper_canonical_fraction=float(self.cfg.scene_gripper_canonical_fraction),
                    finger_color_scale=tuple(self.cfg.scene_finger_color_scale),
                    finger_hue_shift_deg=tuple(self.cfg.scene_finger_hue_shift_deg),
                    finger_roughness=tuple(self.cfg.scene_finger_roughness),
                    pad_color_scale=tuple(self.cfg.scene_pad_color_scale),
                    pad_temperature_shift=tuple(self.cfg.scene_pad_temperature_shift),
                    pad_roughness=tuple(self.cfg.scene_pad_roughness),
                    ground_color_scale=tuple(self.cfg.scene_ground_color_scale),
                    ground_hue_shift_deg=tuple(self.cfg.scene_ground_hue_shift_deg),
                    ground_roughness=tuple(self.cfg.scene_ground_roughness),
                ),
                light_paths=self.visual_light_paths,
                material_paths=self.visual_material_bindings["materials"],
                gripper_variant_paths=self.visual_material_bindings["gripper_appearance_variant_roots"],
                device=self.device,
            )
            self.scene_appearance_randomizer.maybe_randomize(
                0,
                force=True,
                strength=0.0 if self.cfg.training_curriculum_enabled else 1.0,
            )
            self.live_workspace_appearance_randomizer = LiveWorkspaceAppearanceRandomizer(
                part_shader_paths_by_env=self.visual_material_bindings["part_shaders_by_env"],
                tslot_aluminum_shader_paths=self.tslot_visual_bindings["aluminum_shader_paths"],
                num_envs=self.num_envs,
                device=self.device,
                part_color_scale=tuple(self.cfg.scene_part_color_scale),
                part_saturation_scale=tuple(self.cfg.scene_part_saturation_scale),
                part_hue_shift_deg=tuple(self.cfg.scene_part_hue_shift_deg),
                part_roughness=tuple(self.cfg.scene_part_roughness),
                part_metallic=tuple(self.cfg.scene_part_metallic),
                tslot_color_scale=tuple(self.cfg.scene_tslot_color_scale),
                tslot_saturation_scale=tuple(self.cfg.scene_tslot_saturation_scale),
                tslot_hue_shift_deg=tuple(self.cfg.scene_tslot_hue_shift_deg),
                tslot_roughness_delta=tuple(self.cfg.scene_tslot_roughness_delta),
            )
        self.context = FR3MotionContext(
            robot=self.robot,
            scene=self.scene,
            sim=self.sim,
            fixed_gripper_width=PDZ_GRIPPER_OPEN_WIDTH_M,
        )
        self.arm_ids = self.context.arm_joint_ids
        self.previous_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.applied_action_delta = torch.zeros_like(self.previous_actions)
        self.filtered_motion_actions = torch.zeros_like(self.previous_actions)
        self.motion_action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_response_scale = torch.ones((self.num_envs, 1), device=self.device)
        self.motion_response_alpha = torch.ones((self.num_envs, 1), device=self.device)
        self.motion_bias = torch.zeros_like(self.previous_actions)
        self.physics_joint_stiffness_scale = torch.ones((self.num_envs, 1), device=self.device)
        self.physics_joint_damping_scale = torch.ones((self.num_envs, 1), device=self.device)
        action_history_length = max(1, int(self.cfg.motion_action_delay_max_steps) + 1)
        self.motion_action_history = torch.zeros((action_history_length, self.num_envs, 6), device=self.device)
        self.live_observation_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        observation_history_length = max(1, int(self.cfg.live_observation_delay_max_steps) + 1)
        self.live_observation_history = torch.zeros(
            (
                observation_history_length,
                self.num_envs,
                VISUAL_SERVO_OBSERVATION_HEIGHT,
                VISUAL_SERVO_OBSERVATION_WIDTH,
                4,
            ),
            dtype=torch.float16,
            device=self.device,
        )
        self.live_observation_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_probability = torch.zeros(self.num_envs, device=self.device)
        self.completion_stop_candidate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.completion_declaration = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_policy_transition = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_commit_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_commit_operational = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_commit_strict = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_commit_quality = torch.zeros(self.num_envs, device=self.device)
        self.lift_commit_position_error = torch.zeros(self.num_envs, device=self.device)
        self.lift_commit_rotation_error = torch.zeros(self.num_envs, device=self.device)
        self.lift_fixture_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_fixture_quaternion = torch.zeros((self.num_envs, 4), device=self.device)
        self.lift_fixture_quaternion[:, 0] = 1.0
        self.lift_arm_target = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.lift_prelift_object_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_prelift_tcp_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_initial_relative_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_peak_object_z = torch.zeros(self.num_envs, device=self.device)
        self.lift_final_height = torch.zeros(self.num_envs, device=self.device)
        self.lift_peak_height = torch.zeros(self.num_envs, device=self.device)
        self.lift_relative_drift = torch.zeros(self.num_envs, device=self.device)
        self.lift_quality = torch.zeros(self.num_envs, device=self.device)
        self.lift_pickup_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_arm_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_operational_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        self.reset_object_yaw_offset = torch.zeros(self.num_envs, device=self.device)
        self.reset_object_yaw_requested = torch.zeros(self.num_envs, device=self.device)
        self.reset_object_yaw_safe_cap = torch.zeros(self.num_envs, device=self.device)
        self.reset_goal_position_delta = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_position_error = torch.zeros(self.num_envs, device=self.device)
        self.initial_rotation_error = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_fraction = 0.0
        self.curriculum_progress_min = float(self.cfg.reset_progress_min)
        self.curriculum_perturbation_scale = 1.0
        self.curriculum_visual_strength = 1.0
        self.camera_config = D405WristCameraConfig()
        self.observation_preprocess_cfg = D405ObservationPreprocessCfg.from_camera(self.camera_config)
        self.rotation_tcp_from_camera = torch.tensor(
            camera_rotation_in_link7(self.camera_config),
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
                correlated_depth_enabled=bool(self.cfg.live_correlated_depth_enabled),
                stereo_focal_length_px=float(self.cfg.live_stereo_focal_length_px),
                stereo_baseline_m=float(self.cfg.live_stereo_baseline_m),
                disparity_bias_px=tuple(self.cfg.live_disparity_bias_px),
                disparity_independent_noise_std_px=tuple(self.cfg.live_disparity_independent_noise_std_px),
                disparity_spatial_noise_std_px=tuple(self.cfg.live_disparity_spatial_noise_std_px),
                disparity_temporal_noise_std_px=tuple(self.cfg.live_disparity_temporal_noise_std_px),
                disparity_temporal_correlation=tuple(self.cfg.live_disparity_temporal_correlation),
                stereo_edge_mismatch_probability=float(self.cfg.live_stereo_edge_mismatch_probability),
                stereo_edge_horizontal_radius_px=int(self.cfg.live_stereo_edge_horizontal_radius_px),
                depth_quantization_m=float(self.cfg.live_depth_quantization_m),
                depth_dropout_probability=tuple(self.cfg.live_depth_dropout_probability),
                depth_edge_dropout_probability=tuple(self.cfg.live_depth_edge_dropout_probability),
                depth_edge_threshold_m=float(self.cfg.live_depth_edge_threshold_m),
                rgb_patch_occlusion_probability=float(self.cfg.live_rgb_patch_occlusion_probability),
                depth_patch_dropout_probability=float(self.cfg.live_depth_patch_dropout_probability),
                depth_structured_dropout_probability=float(self.cfg.live_depth_structured_dropout_probability),
                depth_structured_dropout_seed_probability=tuple(
                    self.cfg.live_depth_structured_dropout_seed_probability
                ),
                patch_area_fraction=tuple(self.cfg.live_patch_area_fraction),
                calibration_warp_enabled=bool(self.cfg.live_calibration_warp_enabled),
                calibration_shift_x_px=tuple(self.cfg.live_calibration_shift_x_px),
                calibration_shift_y_px=tuple(self.cfg.live_calibration_shift_y_px),
                calibration_scale=tuple(self.cfg.live_calibration_scale),
                calibration_roll_deg=tuple(self.cfg.live_calibration_roll_deg),
                clean_episode_fraction=float(self.cfg.live_clean_episode_fraction),
                depth_min_m=float(self.camera_config.reliable_depth_range_m[0]),
                depth_max_m=float(self.camera_config.reliable_depth_range_m[1]),
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
            if approach_profile != PDZ_GRIPPER_APPROACH_PROFILE:
                raise ValueError(
                    "Goal catalog approach-gripper mismatch: "
                    f"catalog='{approach_profile or 'unlabeled'}', "
                    f"environment='{PDZ_GRIPPER_APPROACH_PROFILE}'. Rebuild the "
                    "path asset and re-render every MuJoCo goal image before training "
                    "or playback."
                )
            material_profile = str(np.asarray(complete_catalog.get("visual_material_profile", "")).item())
            if material_profile != VISUAL_SERVO_MATERIAL_PROFILE:
                raise ValueError(
                    "Goal catalog visual material mismatch: "
                    f"catalog='{material_profile or 'unlabeled'}', "
                    f"environment='{VISUAL_SERVO_MATERIAL_PROFILE}'. Re-capture the "
                    "MuJoCo goal RGB-D catalog before training or playback."
                )
            scene_profile = str(np.asarray(complete_catalog.get("visual_scene_profile", "")).item())
            if scene_profile != VISUAL_SERVO_SCENE_PROFILE:
                raise ValueError(
                    "Goal catalog visual scene mismatch: "
                    f"catalog='{scene_profile or 'unlabeled'}', "
                    f"environment='{VISUAL_SERVO_SCENE_PROFILE}'. Re-capture the "
                    "catalog under the canonical lighting and RTX profile."
                )
            tslot_profile = str(np.asarray(complete_catalog.get("visual_tslot_profile", "")).item())
            if tslot_profile != VISUAL_SERVO_TSLOT_PROFILE:
                raise ValueError(
                    "Goal catalog workspace mismatch: "
                    f"catalog='{tslot_profile or 'unlabeled'}', "
                    f"environment='{VISUAL_SERVO_TSLOT_PROFILE}'. Re-capture the "
                    "catalog over the canonical small-pitch T-slot surface."
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
        configured_rotation_radii = tuple(float(value) for value in self.cfg.part_xy_rotation_radii_m)
        if len(configured_rotation_radii) != len(self.part_names) or any(
            value <= 0.0 for value in configured_rotation_radii
        ):
            raise ValueError("part_xy_rotation_radii_m must contain one positive radius for every configured part.")
        self.part_xy_rotation_radii = torch.as_tensor(
            configured_rotation_radii,
            dtype=torch.float32,
            device=self.device,
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
        self.grasp_jaw_widths_catalog = torch.as_tensor(
            catalog.get("grasp_jaw_widths_m", catalog["approach_gripper_widths_m"]),
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
        goal_rgb_t = torch.as_tensor(raw_rgb, device=self.device).float().div_(255.0)
        goal_depth_t = torch.as_tensor(raw_depth, device=self.device).float().unsqueeze(-1)
        goal_rgb_t, goal_depth_t, _goal_valid = resize_aligned_rgbd_torch(
            goal_rgb_t,
            goal_depth_t,
            cfg=self.observation_preprocess_cfg,
        )
        self.goal_rgbd_catalog = pack_policy_rgbd_torch(
            goal_rgb_t,
            goal_depth_t,
            cfg=self.observation_preprocess_cfg,
        )
        self.goal_rgbd = self.goal_rgbd_catalog[0:1].repeat(self.num_envs, 1, 1, 1)
        self.goal_rgb_policy_variants_cpu: torch.Tensor | None = None
        self.goal_variant_palette_indices: torch.Tensor | None = None
        if "goal_rgb_policy_variants" in catalog:
            self.goal_rgb_policy_variants_cpu = torch.from_numpy(
                np.ascontiguousarray(catalog["goal_rgb_policy_variants"])
            )
            self.goal_variant_palette_indices = torch.as_tensor(
                catalog["goal_variant_palette_indices"], dtype=torch.long, device=self.device
            )
        elif bool(self.cfg.goal_live_color_relationship_enabled):
            message = (
                "Goal/live color relationships were requested, but this catalog has no true-rendered "
                "goal_rgb_policy_variants. Re-capture it with --goal-palette-indices."
            )
            if bool(self.cfg.goal_live_color_relationship_required):
                raise ValueError(message)
            print(f"[WARNING] {message} Falling back to independent live colors.", flush=True)
        self.goal_color_relationship_code = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.goal_palette_index = torch.full_like(self.goal_color_relationship_code, -1)
        self.live_palette_index = torch.full_like(self.goal_color_relationship_code, -1)
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
        if (
            self.cfg.reset_position_randomization_enabled or self.cfg.reset_object_yaw_randomization_enabled
        ) and self.rotation_reset_collision_clearance is None:
            raise ValueError(
                "Object-pose reset randomization requires a rotation-reset asset with per-state collision clearances."
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
            "approach_gripper_widths_m": np.asarray([PDZ_GRIPPER_OPEN_WIDTH_M], dtype=np.float32),
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
                prim_path=(
                    "/World/envs/env_.*/Robot/(gripper_base_link|left_finger_link|"
                    "right_finger_link|pdz_gripper_base_link|"
                    "pdz_gripper_left_finger_link|pdz_gripper_right_finger_link)"
                ),
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
        self.tslot_visual_bindings = spawn_visual_servo_tslot_surfaces(
            self.num_envs,
            enabled=bool(self.cfg.scene_tslot_surface_enabled),
            geometry_randomization_enabled=bool(self.cfg.scene_tslot_geometry_randomization_enabled),
            seed=int(self.cfg.seed),
            nominal_fraction=float(self.cfg.scene_tslot_nominal_fraction),
            phase_fraction=float(self.cfg.scene_tslot_phase_fraction),
        )
        self.surface_marking_visual_bindings = spawn_visual_servo_surface_markings(
            self.num_envs,
            enabled=bool(self.cfg.scene_surface_markings_enabled),
            seed=int(self.cfg.seed) + 5_003,
            tslot_variants=self.tslot_visual_bindings["variants"],
            environment_fraction=float(self.cfg.scene_surface_markings_environment_fraction),
            clean_fraction=float(self.cfg.scene_surface_markings_clean_fraction),
            min_markings=int(self.cfg.scene_surface_markings_min_count),
            max_markings=int(self.cfg.scene_surface_markings_max_count),
            target_clearance_radius_m=float(self.cfg.scene_surface_markings_target_clearance_radius_m),
        )
        self.clutter_visual_bindings = spawn_visual_servo_clutter(
            self.num_envs,
            enabled=bool(self.cfg.scene_clutter_enabled),
            seed=int(self.cfg.seed) + 10_003,
            environment_fraction=float(self.cfg.scene_clutter_environment_fraction),
            min_objects=int(self.cfg.scene_clutter_min_objects),
            max_objects=int(self.cfg.scene_clutter_max_objects),
        )
        self.busy_background_visual_bindings = spawn_visual_servo_busy_background(
            self.num_envs,
            enabled=bool(self.cfg.scene_busy_background_enabled),
            seed=int(self.cfg.seed) + 20_003,
            environment_fraction=float(self.cfg.scene_busy_background_environment_fraction),
            min_people=int(self.cfg.scene_busy_background_min_people),
            max_people=int(self.cfg.scene_busy_background_max_people),
        )
        self.scene.articulations["robot"] = self.robot
        for part_index, part in enumerate(self.parts):
            key = "part" if len(self.parts) == 1 else f"part_{part_index}"
            self.scene.rigid_objects[key] = part
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["gripper_contact"] = self.gripper_contact_sensor
        if self.debug_camera is not None:
            self.scene.sensors["debug_camera"] = self.debug_camera
        self.visual_light_paths = spawn_visual_servo_lights()

    def _active_part_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected part pose for every vectorized environment."""

        selected_parts = self.target_part_indices[self.target_index]
        positions = torch.empty((self.num_envs, 3), dtype=torch.float32, device=self.device)
        quaternions = torch.empty((self.num_envs, 4), dtype=torch.float32, device=self.device)
        for part_index, part in enumerate(self.parts):
            mask = selected_parts == part_index
            if mask.any():
                positions[mask] = part.data.root_pos_w[mask]
                quaternions[mask] = part.data.root_quat_w[mask]
        return positions, quaternions

    def _set_part_gravity_disabled(self, env_ids: torch.Tensor, *, disabled: bool) -> None:
        """Toggle gravity only for the selected part clone in each environment."""

        if not bool(self.cfg.lift_reward_enabled) or env_ids.numel() == 0:
            return
        selected_parts = self.target_part_indices[self.target_index[env_ids]]
        for part_index, part in enumerate(self.parts):
            local_mask = selected_parts == part_index
            if not local_mask.any():
                continue
            selected_ids = env_ids[local_mask]
            cpu_ids = selected_ids.detach().to(device="cpu", dtype=torch.int32).reshape(-1)
            # PhysX indexes into a full-view value tensor even when ``indices``
            # selects only a few environments.  A subset-sized tensor happens
            # to work for an all-environment update but fails as soon as a
            # smaller asynchronous reset releases or restores gravity.
            flags = torch.full(
                (int(part.root_physx_view.count), 1),
                int(disabled),
                dtype=torch.uint8,
                device="cpu",
            )
            part.root_physx_view.set_disable_gravities(flags, cpu_ids)
            if not disabled:
                part.root_physx_view.wake_up(cpu_ids)

    def _enable_only_selected_part_simulation(self, env_ids: torch.Tensor) -> None:
        """Simulate one part clone per environment and disable every parked clone.

        The Fabrica task keeps one clone of every supported part type in each
        vectorized environment so reset can select targets without rebuilding
        the scene.  Lift-aware profiles make part rigid bodies dynamic.  If all
        clones remain enabled, PhysX still advances every parked body and the
        cost scales with the number of dataset part types rather than the one
        object visible in the episode.

        Disabling simulation does not remove the prim or its renderable mesh.
        The selected clone remains a normal dynamic, collidable rigid body;
        only the unused clones parked below the scene are excluded from the
        physics solve.
        """

        if not bool(self.cfg.lift_reward_enabled) or env_ids.numel() == 0:
            return
        selected_parts = self.target_part_indices[self.target_index[env_ids]]
        cpu_env_ids = env_ids.detach().to(device="cpu", dtype=torch.int32).reshape(-1)
        for part_index, part in enumerate(self.parts):
            # Like set_disable_gravities(), the raw PhysX tensor API expects a
            # full-view value tensor and separately indexes the rows to apply.
            disabled = torch.ones(
                (int(part.root_physx_view.count), 1),
                dtype=torch.uint8,
                device="cpu",
            )
            selected_ids = env_ids[selected_parts == part_index]
            if selected_ids.numel() > 0:
                selected_cpu_ids = selected_ids.detach().to(device="cpu", dtype=torch.long)
                disabled[selected_cpu_ids] = 0
            part.root_physx_view.set_disable_simulations(disabled, cpu_env_ids)
            if selected_ids.numel() > 0:
                part.root_physx_view.wake_up(selected_cpu_ids.to(dtype=torch.int32))

    def _restore_selected_part_fixture(self, env_ids: torch.Tensor) -> None:
        """Hard-hold selected dynamic parts at their stable reset pose."""

        if env_ids.numel() == 0:
            return
        selected_parts = self.target_part_indices[self.target_index[env_ids]]
        zero_velocity = torch.zeros((env_ids.numel(), 6), dtype=torch.float32, device=self.device)
        for part_index, part in enumerate(self.parts):
            local_mask = selected_parts == part_index
            if not local_mask.any():
                continue
            selected_ids = env_ids[local_mask]
            pose = torch.cat(
                (
                    self.lift_fixture_position[selected_ids],
                    self.lift_fixture_quaternion[selected_ids],
                ),
                dim=-1,
            )
            part.write_root_pose_to_sim(pose, env_ids=selected_ids)
            part.write_root_velocity_to_sim(zero_velocity[local_mask], env_ids=selected_ids)

    def _lift_phase_steps(self, seconds: float) -> int:
        return max(1, round(float(seconds) / float(self.physics_dt)))

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

    def _policy_context(self, tcp_quaternion: torch.Tensor) -> torch.Tensor:
        """Return actor context that can be reproduced from the real pose stream."""

        mode = str(self.cfg.policy_context_mode)
        spec = resolve_policy_context(mode)
        if not spec.uses_tcp_twist:
            return self.previous_actions

        rotation_world_from_camera = self._rotation_world_from_camera(tcp_quaternion)
        rotation_camera_from_world = rotation_world_from_camera.transpose(1, 2)
        body_velocity_world = self.robot.data.body_link_vel_w[:, self.context.ee_body_idx]
        twist_camera = torch.cat(
            (
                torch.bmm(rotation_camera_from_world, body_velocity_world[:, :3, None]).squeeze(-1),
                torch.bmm(rotation_camera_from_world, body_velocity_world[:, 3:, None]).squeeze(-1),
            ),
            dim=-1,
        )
        normalized_twist_camera = torch.cat(
            (
                twist_camera[:, :3] / float(self.cfg.linear_action_scale_m_s),
                twist_camera[:, 3:] / float(self.cfg.angular_action_scale_rad_s),
            ),
            dim=-1,
        ).clamp(-5.0, 5.0)
        rotation_base_from_camera = None
        if spec.uses_camera_rotation:
            rotation_base_from_world = matrix_from_quat(quat_conjugate(self.robot.data.root_quat_w))
            rotation_base_from_camera = torch.bmm(rotation_base_from_world, rotation_world_from_camera)
        return assemble_policy_context_torch(
            mode,
            self.previous_actions,
            normalized_tcp_twist_camera=normalized_twist_camera,
            rotation_base_from_camera=rotation_base_from_camera,
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
        scripted_at_start = bool(self.cfg.lift_reward_enabled) & (self.lift_phase != self._LIFT_PHASE_APPROACH)
        self.lift_policy_transition.copy_(~scripted_at_start)
        self.lift_commit_event.zero_()
        requested_actions = actions[:, :6].clamp(-1.0, 1.0)
        if self.motion_action_history.shape[0] > 1:
            self.motion_action_history[1:] = self.motion_action_history[:-1].clone()
        self.motion_action_history[0].copy_(requested_actions)
        env_indices = torch.arange(self.num_envs, device=self.device)
        delayed_actions = self.motion_action_history[self.motion_action_delay_steps, env_indices]
        response_target = delayed_actions * self.motion_response_scale + self.motion_bias
        self.filtered_motion_actions.mul_(1.0 - self.motion_response_alpha).add_(
            response_target * self.motion_response_alpha
        )
        action_delta = (self.filtered_motion_actions - self.previous_actions).clamp(
            -float(self.cfg.action_delta_limit),
            float(self.cfg.action_delta_limit),
        )
        self.applied_action_delta.copy_(action_delta)
        self.actions = self.previous_actions + action_delta
        if bool(self.cfg.lift_reward_enabled):
            self.actions[scripted_at_start] = 0.0
            self.applied_action_delta[scripted_at_start] = 0.0
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
        eligible = ~scripted_at_start
        streak_probability = torch.where(
            self.completion_stop_candidate & stable & eligible,
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
        self.completion_declaration.copy_(
            eligible & (self.completion_streak >= int(self.cfg.completion_required_consecutive_steps))
        )
        if bool(self.cfg.lift_reward_enabled):
            self._begin_lift_attempts()
            # An approach object has gravity disabled and should only require
            # one fixture correction per policy step.  Writing its pose and
            # velocity from _apply_action() repeated this CPU-to-PhysX work for
            # every decimation substep (eight times at 15 Hz), even though any
            # contact during approach terminates the episode after this step.
            approach_ids = torch.nonzero(
                self.lift_phase == self._LIFT_PHASE_APPROACH,
                as_tuple=False,
            ).flatten()
            self._restore_selected_part_fixture(approach_ids)

    def _begin_lift_attempts(self) -> None:
        """Turn new completion declarations into one scripted physical trial."""

        tcp_position, _, position_error, rotation_error = self._tcp_error()
        position_norm = torch.linalg.norm(position_error, dim=-1)
        rotation_norm = torch.linalg.norm(rotation_error, dim=-1)
        collision = self._gripper_collision()
        start = (
            self.completion_declaration
            & ~collision
            & (position_norm <= float(self.cfg.divergence_position_m))
            & (self.lift_phase == self._LIFT_PHASE_APPROACH)
        )
        if not start.any():
            return
        labels = completion_masks(
            position_norm,
            rotation_norm,
            ready_position_m=float(self.cfg.completion_ready_position_m),
            ready_rotation_rad=float(self.cfg.completion_ready_rotation_rad),
            negative_position_m=float(self.cfg.completion_negative_position_m),
            negative_rotation_rad=float(self.cfg.completion_negative_rotation_rad),
            collision_free=~collision,
        )
        quality = completion_quality(
            position_norm,
            rotation_norm,
            ready_position_m=float(self.cfg.completion_ready_position_m),
            ready_rotation_rad=float(self.cfg.completion_ready_rotation_rad),
            negative_position_m=float(self.cfg.completion_negative_position_m),
            negative_rotation_rad=float(self.cfg.completion_negative_rotation_rad),
            collision_free=~collision,
        )
        strict = (
            (position_norm <= float(self.cfg.strict_success_position_m))
            & (rotation_norm <= float(self.cfg.strict_success_rotation_rad))
            & ~collision
        )
        env_ids = torch.nonzero(start, as_tuple=False).flatten()
        self.lift_commit_event[start] = True
        self.lift_commit_operational[start] = labels.ready[start]
        self.lift_commit_strict[start] = strict[start]
        self.lift_commit_quality[start] = quality[start]
        self.lift_commit_position_error[start] = position_norm[start]
        self.lift_commit_rotation_error[start] = rotation_norm[start]
        self.lift_arm_target[start] = self.robot.data.joint_pos[start][:, self.arm_ids]
        self.lift_phase[start] = self._LIFT_PHASE_CLOSE
        self.lift_phase_step[start] = 0
        self.lift_final_height[start] = 0.0
        self.lift_peak_height[start] = 0.0
        self.lift_relative_drift[start] = 0.0
        self.lift_quality[start] = 0.0
        self.lift_pickup_success[start] = False
        self.lift_arm_ok[start] = False
        self._set_part_gravity_disabled(env_ids, disabled=True)

    def _dls_joint_velocity_world(
        self,
        env_ids: torch.Tensor,
        twist_world: torch.Tensor,
        *,
        damping: float,
    ) -> torch.Tensor:
        """Map selected world-frame TCP twists to arm-joint velocity."""

        root_quaternion = self.robot.data.root_quat_w[env_ids]
        rotation_base_from_world = matrix_from_quat(quat_conjugate(root_quaternion))
        twist_base = torch.cat(
            (
                torch.bmm(rotation_base_from_world, twist_world[:, :3, None]).squeeze(-1),
                torch.bmm(rotation_base_from_world, twist_world[:, 3:, None]).squeeze(-1),
            ),
            dim=-1,
        )
        # Slice environments first, then joints. Indexing both dimensions in
        # one expression makes PyTorch treat env_ids and arm_ids as paired
        # advanced indices, which fails whenever their lengths differ.
        jacobian = self.robot.root_physx_view.get_jacobians()[env_ids, self.context.ee_jacobi_body_idx][
            :, :, self.arm_ids
        ]
        transpose = jacobian.transpose(1, 2)
        identity = torch.eye(6, device=self.device).expand(env_ids.numel(), -1, -1)
        return torch.bmm(
            transpose,
            torch.linalg.solve(
                torch.bmm(jacobian, transpose) + float(damping) ** 2 * identity,
                twist_base.unsqueeze(-1),
            ),
        ).squeeze(-1)

    def _sample_sim2real_dynamics(self, env_ids: torch.Tensor) -> None:
        """Sample training-only timing and controller response per reset."""

        count = len(env_ids)
        if count == 0:
            return
        visual_strength = self.live_observation_randomizer.randomization_strength[env_ids].flatten()
        randomized = visual_strength > 0.0
        observation_max = int(self.cfg.live_observation_delay_max_steps)
        if observation_max > 0:
            sampled_observation_delay = torch.randint(observation_max + 1, (count,), device=self.device)
            self.live_observation_delay_steps[env_ids] = torch.where(
                randomized, sampled_observation_delay, torch.zeros_like(sampled_observation_delay)
            )
        else:
            self.live_observation_delay_steps[env_ids] = 0

        action_max = int(self.cfg.motion_action_delay_max_steps)
        if action_max > 0:
            ordinary_max = min(action_max, 1)
            ordinary_delay = torch.randint(ordinary_max + 1, (count,), device=self.device)
            if action_max >= 2:
                use_two_steps = torch.rand(count, device=self.device) < float(
                    self.cfg.motion_action_two_step_probability
                )
                sampled_action_delay = torch.where(
                    use_two_steps,
                    torch.full_like(ordinary_delay, 2),
                    ordinary_delay,
                )
            else:
                sampled_action_delay = ordinary_delay
            self.motion_action_delay_steps[env_ids] = torch.where(
                randomized, sampled_action_delay, torch.zeros_like(sampled_action_delay)
            )
        else:
            self.motion_action_delay_steps[env_ids] = 0

        def uniform(value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
            lower, upper = value_range
            if lower == upper:
                return torch.full(shape, float(lower), device=self.device)
            return torch.empty(shape, device=self.device).uniform_(float(lower), float(upper))

        response_scale = uniform(tuple(self.cfg.motion_response_scale), (count, 1))
        response_alpha = uniform(tuple(self.cfg.motion_response_alpha), (count, 1))
        motion_bias = uniform(tuple(self.cfg.motion_bias), (count, 6))
        stiffness_scale = uniform(tuple(self.cfg.physics_joint_stiffness_scale), (count, 1))
        damping_scale = uniform(tuple(self.cfg.physics_joint_damping_scale), (count, 1))
        randomized_column = randomized.unsqueeze(-1)
        self.motion_response_scale[env_ids] = torch.where(
            randomized_column, response_scale, torch.ones_like(response_scale)
        )
        self.motion_response_alpha[env_ids] = torch.where(
            randomized_column, response_alpha, torch.ones_like(response_alpha)
        )
        self.motion_bias[env_ids] = torch.where(randomized_column, motion_bias, torch.zeros_like(motion_bias))
        self.physics_joint_stiffness_scale[env_ids] = torch.where(
            randomized_column, stiffness_scale, torch.ones_like(stiffness_scale)
        )
        self.physics_joint_damping_scale[env_ids] = torch.where(
            randomized_column, damping_scale, torch.ones_like(damping_scale)
        )
        stiffness = (
            self.robot.data.default_joint_stiffness[env_ids][:, self.arm_ids]
            * self.physics_joint_stiffness_scale[env_ids]
        )
        damping = (
            self.robot.data.default_joint_damping[env_ids][:, self.arm_ids] * self.physics_joint_damping_scale[env_ids]
        )
        self.robot.write_joint_stiffness_to_sim(stiffness, joint_ids=self.arm_ids, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(damping, joint_ids=self.arm_ids, env_ids=env_ids)

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
        q = self.robot.data.joint_pos[:, self.arm_ids]
        q_target = q.clone()
        normal_mask = self.lift_phase == self._LIFT_PHASE_APPROACH
        normal_ids = torch.nonzero(normal_mask, as_tuple=False).flatten()
        if normal_ids.numel() > 0:
            q_dot = self._dls_joint_velocity_world(
                normal_ids,
                twist_world[normal_ids],
                damping=float(self.cfg.dls_damping),
            )
            q_target[normal_ids] = q[normal_ids] + q_dot * self.step_dt

        if bool(self.cfg.lift_reward_enabled):
            scripted_ids = torch.nonzero(~normal_mask, as_tuple=False).flatten()
            if scripted_ids.numel() > 0:
                q_target[scripted_ids] = self.lift_arm_target[scripted_ids]
            # Closing deliberately retains the hard fixture at physics rate so
            # finger contact cannot push the object away before gravity is
            # released.  Approach correction happens once in
            # _pre_physics_step(), outside the decimation loop.
            close_ids = torch.nonzero(
                self.lift_phase == self._LIFT_PHASE_CLOSE,
                as_tuple=False,
            ).flatten()
            self._restore_selected_part_fixture(close_ids)
            self._advance_lift_phases(q_target)

        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        q_target = torch.max(torch.min(q_target, limits[..., 1]), limits[..., 0])
        if bool(self.cfg.lift_reward_enabled):
            scripted_mask = self.lift_phase != self._LIFT_PHASE_APPROACH
            self.lift_arm_target[scripted_mask] = q_target[scripted_mask]
        self.robot.set_joint_position_target(q_target, joint_ids=self.arm_ids)
        self.context.command_fixed_gripper()
        self.previous_actions.copy_(self.actions)

    def _advance_lift_phases(self, q_target: torch.Tensor) -> None:
        """Advance close, gravity-release, upward-motion, and hold phases."""

        close_steps = self._lift_phase_steps(float(self.cfg.lift_close_duration_s))
        release_steps = self._lift_phase_steps(float(self.cfg.lift_gravity_release_duration_s))
        lift_steps = self._lift_phase_steps(float(self.cfg.lift_height_m) / float(self.cfg.lift_speed_m_s))
        hold_steps = self._lift_phase_steps(float(self.cfg.lift_hold_duration_s))

        phase_at_start = self.lift_phase.clone()
        close_mask = phase_at_start == self._LIFT_PHASE_CLOSE
        if close_mask.any():
            progress = (self.lift_phase_step[close_mask].float() + 1.0) / float(close_steps)
            target_indices = self.target_index[close_mask]
            approach_width = self.approach_gripper_widths_catalog[target_indices]
            close_width = torch.clamp(
                self.grasp_jaw_widths_catalog[target_indices] - float(self.cfg.lift_squeeze_margin_m),
                min=PDZ_GRIPPER_CLOSED_WIDTH_M,
            )
            widths = torch.lerp(approach_width, close_width, progress.clamp(0.0, 1.0))
            close_ids = torch.nonzero(close_mask, as_tuple=False).flatten()
            self.context.set_fixed_gripper_widths(widths, env_ids=close_ids)
            self.lift_phase_step[close_mask] += 1
            finished = close_mask & (self.lift_phase_step >= close_steps)
            self.lift_phase[finished] = self._LIFT_PHASE_GRAVITY_RELEASE
            self.lift_phase_step[finished] = 0

        release_mask = phase_at_start == self._LIFT_PHASE_GRAVITY_RELEASE
        if release_mask.any():
            first_release = release_mask & (self.lift_phase_step == 0)
            if first_release.any():
                self._set_part_gravity_disabled(
                    torch.nonzero(first_release, as_tuple=False).flatten(),
                    disabled=False,
                )
            self.lift_phase_step[release_mask] += 1
            finished = release_mask & (self.lift_phase_step >= release_steps)
            self.lift_phase[finished] = self._LIFT_PHASE_ASCEND
            self.lift_phase_step[finished] = 0

        lift_mask = phase_at_start == self._LIFT_PHASE_ASCEND
        if lift_mask.any():
            lift_ids = torch.nonzero(lift_mask, as_tuple=False).flatten()
            first_lift = lift_mask & (self.lift_phase_step == 0)
            if first_lift.any():
                object_position, _ = self._active_part_pose()
                tcp_position, _ = self.context.get_tcp_pose_w()
                self.lift_prelift_object_position[first_lift] = object_position[first_lift]
                self.lift_prelift_tcp_position[first_lift] = tcp_position[first_lift]
                self.lift_initial_relative_position[first_lift] = object_position[first_lift] - tcp_position[first_lift]
                self.lift_peak_object_z[first_lift] = object_position[first_lift, 2]
            object_position, _ = self._active_part_pose()
            self.lift_peak_object_z[lift_mask] = torch.maximum(
                self.lift_peak_object_z[lift_mask], object_position[lift_mask, 2]
            )
            upward_twist = torch.zeros((lift_ids.numel(), 6), device=self.device)
            upward_twist[:, 2] = float(self.cfg.lift_speed_m_s)
            q_dot = self._dls_joint_velocity_world(
                lift_ids,
                upward_twist,
                damping=float(self.cfg.lift_dls_damping),
            ).clamp(
                -float(self.cfg.lift_maximum_joint_speed_rad_s),
                float(self.cfg.lift_maximum_joint_speed_rad_s),
            )
            self.lift_arm_target[lift_mask] += q_dot * self.physics_dt
            q_target[lift_mask] = self.lift_arm_target[lift_mask]
            self.lift_phase_step[lift_mask] += 1
            finished = lift_mask & (self.lift_phase_step >= lift_steps)
            self.lift_phase[finished] = self._LIFT_PHASE_HOLD
            self.lift_phase_step[finished] = 0

        hold_mask = phase_at_start == self._LIFT_PHASE_HOLD
        if hold_mask.any():
            object_position, _ = self._active_part_pose()
            self.lift_peak_object_z[hold_mask] = torch.maximum(
                self.lift_peak_object_z[hold_mask], object_position[hold_mask, 2]
            )
            q_target[hold_mask] = self.lift_arm_target[hold_mask]
            self.lift_phase_step[hold_mask] += 1
            finished = hold_mask & (self.lift_phase_step >= hold_steps)
            self.lift_phase[finished] = self._LIFT_PHASE_OUTCOME
            self.lift_phase_step[finished] = 0

    def _camera_observation(self, *, randomize_live: bool = True) -> torch.Tensor:
        output = self.wrist_camera.data.output
        rgb = output["rgb"][..., :3].float().div(255.0)
        depth = torch.nan_to_num(output["distance_to_image_plane"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        rgb, depth, _valid = resize_aligned_rgbd_torch(
            rgb,
            depth,
            cfg=self.observation_preprocess_cfg,
        )
        if randomize_live and self.live_observation_randomizer is not None:
            rgb, depth = self.live_observation_randomizer.apply(rgb, depth)
        rgbd = pack_policy_rgbd_torch(rgb, depth, cfg=self.observation_preprocess_cfg)
        timing_enabled = randomize_live and (
            self.live_observation_history.shape[0] > 1 or float(self.cfg.live_observation_repeat_probability) > 0.0
        )
        if timing_enabled:
            previously_valid = self.live_observation_history_valid.clone()
            repeated = previously_valid & (
                torch.rand(self.num_envs, device=self.device) < float(self.cfg.live_observation_repeat_probability)
            )
            newest = torch.where(
                repeated.view(-1, 1, 1, 1),
                self.live_observation_history[0].float(),
                rgbd,
            )
            if self.live_observation_history.shape[0] > 1:
                self.live_observation_history[1:] = self.live_observation_history[:-1].clone()
            self.live_observation_history[0].copy_(newest.to(torch.float16))
            first_frame_envs = torch.nonzero(~previously_valid, as_tuple=False).flatten()
            if first_frame_envs.numel() > 0:
                self.live_observation_history[:, first_frame_envs] = newest[first_frame_envs].to(torch.float16)
            self.live_observation_history_valid[:] = True
            env_indices = torch.arange(self.num_envs, device=self.device)
            rgbd = self.live_observation_history[self.live_observation_delay_steps, env_indices].float()
        return torch.cat((rgbd, self.goal_rgbd), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        _, tcp_quaternion, position_error, rotation_error = self._tcp_error()
        q = self.robot.data.joint_pos[:, self.arm_ids]
        qd = self.robot.data.joint_vel[:, self.arm_ids]
        state = torch.cat((q, qd, position_error, rotation_error, self.previous_actions), dim=-1)
        if bool(self.cfg.lift_reward_enabled):
            phase_features = torch.stack(
                tuple(
                    (self.lift_phase == phase).float()
                    for phase in (
                        self._LIFT_PHASE_CLOSE,
                        self._LIFT_PHASE_GRAVITY_RELEASE,
                        self._LIFT_PHASE_ASCEND,
                        self._LIFT_PHASE_HOLD,
                    )
                ),
                dim=-1,
            )
            object_position, _ = self._active_part_pose()
            measured_phase = self.lift_phase >= self._LIFT_PHASE_ASCEND
            current_lift = torch.where(
                measured_phase,
                object_position[:, 2] - self.lift_prelift_object_position[:, 2],
                torch.zeros_like(object_position[:, 2]),
            )
            lift_state = torch.cat(
                (
                    phase_features,
                    self.lift_commit_operational.float().unsqueeze(-1),
                    (current_lift / float(self.cfg.lift_height_m)).unsqueeze(-1),
                ),
                dim=-1,
            )
            state = torch.cat((state, lift_state), dim=-1)
        visual_observation = self._camera_observation().flatten(start_dim=1)
        # The actor context contains only deployment-available values. The
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
        completion_supervised = labels.supervised
        if bool(self.cfg.lift_reward_enabled):
            if not bool(self.cfg.lift_completion_negative_supervision_enabled):
                # Physical success away from the nominal target is unknown
                # until after the trial. Avoid teaching those states as
                # definite negatives; PPO's failed-lift penalty supplies the
                # negative completion signal instead.
                completion_supervised = labels.ready
            # -1 is a privileged rollout-only sentinel. The visual network
            # masks auxiliary losses and PPO masks actor updates on scripted
            # close/lift transitions; the actor never consumes this value.
            completion_supervised = completion_supervised.float()
            completion_supervised = torch.where(
                self.lift_phase == self._LIFT_PHASE_APPROACH,
                completion_supervised,
                -torch.ones_like(completion_supervised),
            )
        completion_target = torch.stack((labels.ready.float(), completion_supervised.float()), dim=-1)
        policy_context = self._policy_context(tcp_quaternion)
        policy_observation = torch.cat(
            (
                visual_observation,
                policy_context,
                pose_target,
                completion_target,
            ),
            dim=-1,
        )
        return {"policy": policy_observation, "critic": state}

    def _finalize_lift_outcomes(self) -> torch.Tensor:
        """Measure terminal retained-lift outcomes from the pre-lift baseline."""

        terminal = self.lift_phase == self._LIFT_PHASE_OUTCOME
        if not terminal.any():
            return terminal
        object_position, _ = self._active_part_pose()
        tcp_position, _ = self.context.get_tcp_pose_w()
        self.lift_peak_object_z[terminal] = torch.maximum(
            self.lift_peak_object_z[terminal], object_position[terminal, 2]
        )
        final_height = object_position[:, 2] - self.lift_prelift_object_position[:, 2]
        peak_height = self.lift_peak_object_z - self.lift_prelift_object_position[:, 2]
        final_relative = object_position - tcp_position
        relative_drift = torch.linalg.norm(
            final_relative - self.lift_initial_relative_position,
            dim=-1,
        )
        tcp_lift = tcp_position[:, 2] - self.lift_prelift_tcp_position[:, 2]
        arm_ok = tcp_lift >= (float(self.cfg.lift_minimum_arm_fraction) * float(self.cfg.lift_height_m))
        quality = retained_lift_quality(
            final_height,
            peak_height,
            relative_drift,
            arm_ok,
            minimum_credit_lift_m=float(self.cfg.lift_minimum_credit_m),
            full_credit_lift_m=float(self.cfg.lift_full_credit_m),
            drift_scale_m=float(self.cfg.lift_drift_scale_m),
            drop_scale_m=float(self.cfg.lift_drop_scale_m),
        )
        pickup = physical_pickup_success(
            final_height,
            peak_height,
            relative_drift,
            arm_ok,
            minimum_final_lift_m=float(self.cfg.lift_minimum_pickup_m),
            maximum_relative_drift_m=float(self.cfg.lift_maximum_relative_drift_m),
            maximum_peak_drop_m=float(self.cfg.lift_maximum_peak_drop_m),
        )
        self.lift_final_height[terminal] = final_height[terminal]
        self.lift_peak_height[terminal] = peak_height[terminal]
        self.lift_relative_drift[terminal] = relative_drift[terminal]
        self.lift_arm_ok[terminal] = arm_ok[terminal]
        self.lift_quality[terminal] = quality[terminal]
        self.lift_pickup_success[terminal] = pickup[terminal]
        return terminal

    def _get_rewards(self) -> torch.Tensor:
        _, _, position_error, rotation_error = self._tcp_error()
        position_norm = torch.linalg.norm(position_error, dim=-1)
        rotation_norm = torch.linalg.norm(rotation_error, dim=-1)
        lift_enabled = bool(self.cfg.lift_reward_enabled)
        lift_terminal = (
            self._finalize_lift_outcomes() if lift_enabled else torch.zeros_like(self.completion_declaration)
        )
        policy_transition = (
            self.lift_policy_transition if lift_enabled else torch.ones_like(self.completion_declaration)
        )
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
        operational_quality = completion_quality(
            position_norm,
            rotation_norm,
            ready_position_m=float(self.cfg.completion_ready_position_m),
            ready_rotation_rad=float(self.cfg.completion_ready_rotation_rad),
            negative_position_m=float(self.cfg.completion_negative_position_m),
            negative_rotation_rad=float(self.cfg.completion_negative_rotation_rad),
            collision_free=~collision,
        )
        strict_ready = (
            (position_norm <= float(self.cfg.strict_success_position_m))
            & (rotation_norm <= float(self.cfg.strict_success_rotation_rad))
            & ~collision
        )
        diverged = position_norm > self.cfg.divergence_position_m
        approach_now = (
            self.lift_phase == self._LIFT_PHASE_APPROACH
            if lift_enabled
            else torch.ones_like(self.completion_declaration)
        )
        # Contact during the controller-owned close/lift sequence is intended.
        # Safety termination uses the contact state captured while the policy
        # still owns motion; _begin_lift_attempts already refuses a declaration
        # that was colliding before close began.
        collision_terminal = collision & approach_now
        diverged_terminal = diverged & approach_now & ~collision_terminal
        if lift_enabled:
            completion_terminal = lift_terminal
            operational_completion = completion_terminal & self.lift_commit_operational
            strict_completion = completion_terminal & self.lift_commit_strict
            physical_completion = completion_terminal & self.lift_pickup_success
            task_completion = operational_completion | physical_completion
            premature_completion = completion_terminal & ~task_completion
            borderline_completion = premature_completion & (self.lift_commit_quality > 0.0)
        else:
            completion_terminal = self.completion_declaration & ~collision_terminal & ~diverged_terminal
            operational_completion = completion_terminal & labels.ready
            strict_completion = operational_completion & strict_ready
            physical_completion = torch.zeros_like(completion_terminal)
            task_completion = operational_completion
            premature_completion = completion_terminal & ~labels.ready
            borderline_completion = premature_completion & (operational_quality > 0.0)
        self.ever_operational_ready.logical_or_(labels.ready)
        timed_out = self._timed_out() & approach_now
        failed_timeout = timed_out & ~self.completion_declaration & ~diverged_terminal & ~collision_terminal
        missed_operational_timeout = failed_timeout & self.ever_operational_ready

        position_progress = self.previous_position_error - position_norm
        rotation_progress = self.previous_rotation_error - rotation_norm
        previous_position_precision = torch.exp(-self.previous_position_error / self.cfg.position_precision_scale_m)
        position_precision = torch.exp(-position_norm / self.cfg.position_precision_scale_m)
        previous_rotation_precision = torch.exp(-self.previous_rotation_error / self.cfg.rotation_precision_scale_rad)
        rotation_precision = torch.exp(-rotation_norm / self.cfg.rotation_precision_scale_rad)

        # All dense terms are differences of potentials. Holding position no
        # longer accumulates positive reward over a timeout-length episode.
        active = policy_transition.float()
        reward = active * self.cfg.position_progress_weight * position_progress
        reward += active * self.cfg.rotation_progress_weight * rotation_progress
        reward += active * self.cfg.position_precision_weight * (position_precision - previous_position_precision)
        reward += active * self.cfg.rotation_precision_weight * (rotation_precision - previous_rotation_precision)
        if lift_enabled:
            commit_reward = (
                self.lift_commit_event.float()
                * self.lift_commit_operational.float()
                * float(self.cfg.lift_commit_geometric_reward)
            )
            positive_reset_commit = self.lift_commit_event & self.completion_positive_reset
            commit_reward = torch.where(
                positive_reset_commit,
                commit_reward * float(self.cfg.completion_positive_terminal_reward_scale),
                commit_reward,
            )
            reward += commit_reward
            reward += lift_terminal.float() * lift_outcome_reward(
                self.lift_commit_operational,
                self.lift_pickup_success,
                self.lift_quality,
                lift_quality_reward=float(self.cfg.lift_quality_reward),
                geometric_lift_bonus=float(self.cfg.lift_geometric_lift_bonus),
                neither_penalty=float(self.cfg.lift_neither_penalty),
            )
        else:
            terminal_completion_reward = graded_completion_terminal_reward(
                completion_terminal,
                operational_quality,
                correct_reward=float(self.cfg.completion_correct_reward),
                premature_penalty=float(self.cfg.completion_premature_penalty),
            )
            # Ready-pose reset episodes exist to teach declaration, so keep their
            # terminal bonus smaller even if the policy first drifts into the gray
            # band.  Scaling only exact operational successes would reward that
            # drift (a gray-band declaration could otherwise earn more than an
            # exact declaration).  Do not soften negative premature penalties.
            positive_reset_bonus = (
                completion_terminal & self.completion_positive_reset & (terminal_completion_reward > 0.0)
            )
            terminal_completion_reward = torch.where(
                positive_reset_bonus,
                terminal_completion_reward * float(self.cfg.completion_positive_terminal_reward_scale),
                terminal_completion_reward,
            )
            reward += terminal_completion_reward
        timeout_cost = torch.where(
            missed_operational_timeout,
            position_norm.new_full((), float(self.cfg.missed_operational_timeout_penalty)),
            position_norm.new_full((), float(self.cfg.timeout_penalty)),
        )
        reward -= timeout_cost * failed_timeout.float()
        reward -= self.cfg.divergence_penalty * diverged_terminal.float()
        reward -= self.cfg.unsafe_collision_penalty * collision_terminal.float()
        risk_denominator = max(
            float(self.cfg.unsafe_contact_force_threshold_n) - float(self.cfg.collision_risk_force_threshold_n),
            1.0e-6,
        )
        collision_risk = (
            ((contact_force - float(self.cfg.collision_risk_force_threshold_n)) / risk_denominator)
            .clamp(0.0, 1.0)
            .square()
        )
        # These are costs per unit time rather than potentials. Scale them by
        # the actual policy period so changing 30 Hz to 15 Hz does not silently
        # halve the objective's action/hold/contact cost per simulated second.
        time_cost_scale = temporal_reward_scale(self.step_dt)
        reward -= active * time_cost_scale * self.cfg.collision_risk_penalty_weight * collision_risk
        reward -= active * time_cost_scale * self.cfg.step_penalty
        action_norm = torch.linalg.norm(self.actions, dim=-1)
        reward -= active * time_cost_scale * self.cfg.action_penalty_weight * action_norm.square()
        near_goal = (
            (position_norm <= float(self.cfg.completion_negative_position_m))
            & (rotation_norm <= float(self.cfg.completion_negative_rotation_rad))
            & ~collision
        )
        action_delta_norm = torch.linalg.norm(self.applied_action_delta, dim=-1)
        near_goal_action_cost = near_goal.float() * action_norm.square()
        action_delta_cost = action_delta_norm.square()
        near_goal_regression = near_goal.float() * (
            torch.relu(position_norm - self.previous_position_error) / float(self.cfg.position_precision_scale_m)
            + torch.relu(rotation_norm - self.previous_rotation_error) / float(self.cfg.rotation_precision_scale_rad)
        )
        reward -= active * time_cost_scale * float(self.cfg.near_goal_action_penalty_weight) * near_goal_action_cost
        reward -= active * time_cost_scale * float(self.cfg.action_delta_penalty_weight) * action_delta_cost
        reward -= active * float(self.cfg.near_goal_regression_penalty_weight) * near_goal_regression
        linear_speed, angular_speed = self._tcp_speed()
        excess_linear_speed = torch.relu(linear_speed - float(self.cfg.completion_max_linear_speed_m_s)) / float(
            self.cfg.linear_action_scale_m_s
        )
        excess_angular_speed = torch.relu(angular_speed - float(self.cfg.completion_max_angular_speed_rad_s)) / float(
            self.cfg.angular_action_scale_rad_s
        )
        near_goal_excess_speed = near_goal.float() * (excess_linear_speed.square() + excess_angular_speed.square())
        reward -= (
            active * time_cost_scale * float(self.cfg.near_goal_excess_speed_penalty_weight) * near_goal_excess_speed
        )
        self.previous_position_error.copy_(position_norm)
        self.previous_rotation_error.copy_(rotation_norm)
        terminal = completion_terminal | diverged_terminal | collision_terminal | timed_out
        failure_value = failed_timeout.float()
        failure_value = torch.maximum(
            failure_value,
            premature_completion.float() * (1.0 - operational_quality),
        )
        failure_value = torch.maximum(failure_value, 1.25 * diverged_terminal.float())
        failure_value = torch.maximum(failure_value, 1.50 * collision_terminal.float())
        failure_value = torch.where(task_completion, torch.zeros_like(failure_value), failure_value)
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
        scripted_or_outcome = lift_enabled & (self.lift_phase != self._LIFT_PHASE_APPROACH)
        reported_position_norm = torch.where(
            scripted_or_outcome,
            self.lift_commit_position_error,
            position_norm,
        )
        reported_rotation_norm = torch.where(
            scripted_or_outcome,
            self.lift_commit_rotation_error,
            rotation_norm,
        )
        declaration_event = self.lift_commit_event if lift_enabled else self.completion_declaration
        log = {
            "position_error_mm": reported_position_norm.mean() * 1000.0,
            "rotation_error_deg": reported_rotation_norm.mean() * 180.0 / torch.pi,
            # Preserve strict_success_rate for historical benchmark curves;
            # operational_success_rate is the outcome PPO is now rewarded for.
            "success_rate": strict_completion.float().mean(),
            "strict_success_rate": strict_completion.float().mean(),
            "operational_success_rate": operational_completion.float().mean(),
            "lift/pickup_success_rate": physical_completion.float().mean(),
            "lift/task_success_rate": task_completion.float().mean(),
            "lift/phase_active_rate": (~policy_transition).float().mean(),
            "lift/commit_operational_rate": (self.lift_commit_event & self.lift_commit_operational).float().mean(),
            "lift/final_height_mm": (self.lift_final_height * lift_terminal.float()).sum()
            / lift_terminal.float().sum().clamp_min(1.0)
            * 1000.0,
            "lift/quality": (self.lift_quality * lift_terminal.float()).sum()
            / lift_terminal.float().sum().clamp_min(1.0),
            "lift/relative_drift_mm": (self.lift_relative_drift * lift_terminal.float()).sum()
            / lift_terminal.float().sum().clamp_min(1.0)
            * 1000.0,
            "completion/strict_ready_rate": strict_ready.float().mean(),
            "completion/geometric_ready_rate": labels.ready.float().mean(),
            "completion/operational_ready_rate": labels.ready.float().mean(),
            "completion/operational_quality_mean": operational_quality.mean(),
            "completion/ever_operational_ready_rate": self.ever_operational_ready.float().mean(),
            "completion/declaration_rate": declaration_event.float().mean(),
            "completion/premature_rate": premature_completion.float().mean(),
            "completion/borderline_declaration_rate": borderline_completion.float().mean(),
            "completion/missed_ready_rate": (labels.ready & ~declaration_event & policy_transition).float().mean(),
            "completion/missed_operational_timeout_rate": missed_operational_timeout.float().mean(),
            # Stochastic PPO rollouts send the Bernoulli draw (0/1), so this
            # is an unbiased batch estimate of mean p(done). Deterministic
            # playback sends the probability itself.
            "completion/stop_signal_mean": self.completion_probability.mean(),
            "collision_rate": collision_terminal.float().mean(),
            "collision/contact_force_n": contact_force.mean(),
            "collision/risk_mean": collision_risk.mean(),
            "timeout_rate": failed_timeout.float().mean(),
            "action_norm": action_norm.mean(),
            "reward_cost/near_goal_action": near_goal_action_cost.mean(),
            "reward_cost/action_delta": action_delta_cost.mean(),
            "reward_cost/near_goal_regression": near_goal_regression.mean(),
            "reward_cost/near_goal_excess_speed": near_goal_excess_speed.mean(),
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
            "reset/object_yaw_deg": self.reset_object_yaw_offset.abs().mean() * 180.0 / torch.pi,
            "reset/object_yaw_requested_deg": (self.reset_object_yaw_requested.mean() * 180.0 / torch.pi),
            "reset/object_yaw_capped_rate": (
                self.reset_object_yaw_offset.abs() + 1.0e-9 < self.reset_object_yaw_requested
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
            "sim2real/observation_delay_steps": self.live_observation_delay_steps.float().mean(),
            "sim2real/motion_action_delay_steps": self.motion_action_delay_steps.float().mean(),
            "sim2real/motion_response_scale": self.motion_response_scale.mean(),
            "sim2real/motion_response_alpha": self.motion_response_alpha.mean(),
            "sim2real/motion_bias_abs_mean": self.motion_bias.abs().mean(),
            "sim2real/joint_stiffness_scale": self.physics_joint_stiffness_scale.mean(),
            "sim2real/joint_damping_scale": self.physics_joint_damping_scale.mean(),
        }
        if self.live_observation_randomizer is not None:
            log.update(
                {
                    "sim2real/clean_episode_rate": (
                        self.live_observation_randomizer.randomization_strength.flatten() <= 0.0
                    )
                    .float()
                    .mean(),
                    "sim2real/disparity_error_abs_mean_px": (
                        self.live_observation_randomizer.last_disparity_error_abs_mean_px
                    ),
                    "sim2real/depth_invalid_fraction": (self.live_observation_randomizer.last_depth_invalid_fraction),
                }
            )
        if self.scene_appearance_randomizer is not None and self.scene_appearance_randomizer.current_sample is not None:
            appearance = self.scene_appearance_randomizer.current_sample
            log.update(
                {
                    "appearance/key_yaw_delta_deg": position_norm.new_tensor(appearance.key_yaw_delta_deg),
                    "appearance/key_pitch_delta_deg": position_norm.new_tensor(appearance.key_pitch_delta_deg),
                    "appearance/key_intensity": position_norm.new_tensor(appearance.key_intensity),
                    "appearance/key_angle_deg": position_norm.new_tensor(appearance.key_angle_deg),
                    "appearance/dome_intensity": position_norm.new_tensor(appearance.dome_intensity),
                    "appearance/gripper_canonical": position_norm.new_tensor(float(appearance.gripper_canonical)),
                    "appearance/finger_roughness": position_norm.new_tensor(appearance.finger_roughness),
                    "appearance/pad_roughness": position_norm.new_tensor(appearance.pad_roughness),
                }
            )
        if self.live_workspace_appearance_randomizer is not None:
            log.update(
                {
                    "appearance/canonical_part_fraction": (
                        self.live_workspace_appearance_randomizer.part_palette_index == 0
                    )
                    .float()
                    .mean(),
                    "appearance/nominal_tslot_fraction": (
                        self.live_workspace_appearance_randomizer.background_index == 0
                    )
                    .float()
                    .mean(),
                    "appearance/clutter_environment_fraction": position_norm.new_tensor(
                        float(self.clutter_visual_bindings.get("active_environment_count", 0)) / float(self.num_envs)
                    ),
                    "appearance/busy_background_environment_fraction": position_norm.new_tensor(
                        float(self.busy_background_visual_bindings.get("active_environment_count", 0))
                        / float(self.num_envs)
                    ),
                    "appearance/busy_background_people_per_environment": position_norm.new_tensor(
                        float(self.busy_background_visual_bindings.get("people_count", 0)) / float(self.num_envs)
                    ),
                    "appearance/surface_marking_environment_fraction": position_norm.new_tensor(
                        float(self.surface_marking_visual_bindings.get("active_environment_count", 0))
                        / float(self.num_envs)
                    ),
                    "appearance/goal_live_color_match_rate": (
                        self.goal_color_relationship_code == COLOR_RELATIONSHIP_MATCH
                    )
                    .float()
                    .mean(),
                    "appearance/goal_live_color_similar_rate": (
                        self.goal_color_relationship_code == COLOR_RELATIONSHIP_SIMILAR
                    )
                    .float()
                    .mean(),
                    "appearance/goal_live_color_different_rate": (
                        self.goal_color_relationship_code == COLOR_RELATIONSHIP_DIFFERENT
                    )
                    .float()
                    .mean(),
                    "appearance/busy_background_worker_reaches_per_environment": (
                        position_norm.new_tensor(
                            float(self.busy_background_visual_bindings.get("worker_reach_count", 0))
                            / float(self.num_envs)
                        )
                    ),
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
            "reset_object_yaw_offset_rad": self.reset_object_yaw_offset.clone(),
            "reset_object_yaw_requested_rad": self.reset_object_yaw_requested.clone(),
            "reset_object_yaw_safe_cap_rad": self.reset_object_yaw_safe_cap.clone(),
            "reset_goal_position_delta_w": self.reset_goal_position_delta.clone(),
            "completion_positive_reset": self.completion_positive_reset.clone(),
            "completion_exact_reset": self.completion_exact_reset.clone(),
            "reset_mode": self.reset_mode.clone(),
            "reset_timeout_s": self.reset_timeout_s.clone(),
            "reset_failure_replay": self.reset_failure_replay.clone(),
            "goal_color_relationship_code": self.goal_color_relationship_code.clone(),
            "goal_palette_index": self.goal_palette_index.clone(),
            "live_palette_index": self.live_palette_index.clone(),
            "initial_position_error_m": self.initial_position_error.clone(),
            "initial_rotation_error_rad": self.initial_rotation_error.clone(),
            "position_error_m": reported_position_norm.clone(),
            "rotation_error_rad": reported_rotation_norm.clone(),
            # `success` intentionally remains the historical strict benchmark.
            "success": strict_completion.clone(),
            "strict_success": strict_completion.clone(),
            "operational_success": operational_completion.clone(),
            "physical_pickup_success": physical_completion.clone(),
            "task_success": task_completion.clone(),
            "lift_phase": self.lift_phase.clone(),
            "lift_commit_event": self.lift_commit_event.clone(),
            "lift_commit_operational": self.lift_commit_operational.clone(),
            "lift_final_height_m": self.lift_final_height.clone(),
            "lift_peak_height_m": self.lift_peak_height.clone(),
            "lift_relative_drift_m": self.lift_relative_drift.clone(),
            "lift_quality": self.lift_quality.clone(),
            "lift_arm_ok": self.lift_arm_ok.clone(),
            "strict_ready": strict_ready.clone(),
            "operational_ready": labels.ready.clone(),
            "operational_quality": operational_quality.clone(),
            "ever_operational_ready": self.ever_operational_ready.clone(),
            "geometric_ready": labels.ready.clone(),
            "completion_supervised": labels.supervised.clone(),
            "completion_probability": self.completion_probability.clone(),
            "completion_declared": declaration_event.clone(),
            "premature_completion": premature_completion.clone(),
            "borderline_completion": borderline_completion.clone(),
            "collision": collision_terminal.clone(),
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
                    reported_position_norm.masked_fill(~mask, 0.0).sum() / sample_count * 1000.0
                )
                log[f"orientation/{orientation_name}_success_rate"] = (
                    strict_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
                )
                log[f"orientation/{orientation_name}_operational_success_rate"] = (
                    operational_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
                )
        if len(self.part_names) > 1:
            selected_part = self.target_part_indices[self.target_index]
            for part_index, part_name in enumerate(self.part_names):
                mask = selected_part == part_index
                sample_count = mask.sum().clamp_min(1)
                log[f"part/{part_name}_position_error_mm"] = (
                    reported_position_norm.masked_fill(~mask, 0.0).sum() / sample_count * 1000.0
                )
                log[f"part/{part_name}_success_rate"] = (
                    strict_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
                )
                log[f"part/{part_name}_operational_success_rate"] = (
                    operational_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
                )
        # Aggregate curves can improve merely because a policy solves the
        # close resets. Keep the three approach regions separate so far-range
        # behavior remains visible as the curriculum expands.
        for bucket, mask in approach_progress_bucket_masks(self.reset_progress).items():
            sample_count = mask.sum().clamp_min(1)
            log[f"{bucket}/position_error_mm"] = (
                reported_position_norm.masked_fill(~mask, 0.0).sum() / sample_count * 1000.0
            )
            log[f"{bucket}/rotation_error_deg"] = (
                reported_rotation_norm.masked_fill(~mask, 0.0).sum() / sample_count * 180.0 / torch.pi
            )
            log[f"{bucket}/success_rate"] = strict_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
            log[f"{bucket}/operational_success_rate"] = (
                operational_completion.float().masked_fill(~mask, 0.0).sum() / sample_count
            )
        self.extras["log"] = log
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, position_error, _ = self._tcp_error()
        position_norm = torch.linalg.norm(position_error, dim=-1)
        diverged = position_norm > self.cfg.divergence_position_m
        collision = self._gripper_collision()
        timed_out = self._timed_out()
        if bool(self.cfg.lift_reward_enabled):
            approach = self.lift_phase == self._LIFT_PHASE_APPROACH
            return (
                (self.lift_phase == self._LIFT_PHASE_OUTCOME) | (diverged & approach) | (collision & approach),
                timed_out & approach,
            )
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
        forced_live_palette_indices: torch.Tensor | None = None
        self.goal_color_relationship_code[env_ids] = -1
        self.goal_palette_index[env_ids] = -1
        self.live_palette_index[env_ids] = -1
        if (
            bool(self.cfg.goal_live_color_relationship_enabled)
            and self.goal_rgb_policy_variants_cpu is not None
            and self.goal_variant_palette_indices is not None
        ):
            color_pairs = sample_goal_live_color_pairs(
                count,
                self.goal_variant_palette_indices,
                match_fraction=float(self.cfg.goal_live_color_match_fraction),
                similar_fraction=float(self.cfg.goal_live_color_similar_fraction),
                device=self.device,
            )
            selected_variant_rgb = (
                self.goal_rgb_policy_variants_cpu[
                    target_indices.detach().cpu(), color_pairs.goal_variant_slots.detach().cpu()
                ]
                .to(device=self.device, dtype=torch.float32)
                .div_(255.0)
            )
            self.goal_rgbd[env_ids, ..., :3] = selected_variant_rgb
            forced_live_palette_indices = color_pairs.live_palette_indices
            self.goal_color_relationship_code[env_ids] = color_pairs.relationship_codes
            self.goal_palette_index[env_ids] = color_pairs.goal_palette_indices
            self.live_palette_index[env_ids] = color_pairs.live_palette_indices

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
        object_yaw_offset = torch.zeros(count, dtype=torch.float32, device=self.device)
        object_yaw_requested = torch.zeros(count, dtype=torch.float32, device=self.device)
        object_yaw_safe_cap = torch.zeros(count, dtype=torch.float32, device=self.device)
        selected_part_indices = self.target_part_indices[target_indices]
        selected_clearance: torch.Tensor | None = None
        object_pose_randomization_enabled = bool(
            self.cfg.reset_position_randomization_enabled or self.cfg.reset_object_yaw_randomization_enabled
        )
        if object_pose_randomization_enabled:
            if not collision_safe_sampling or progress_indices is None or variant_indices is None:
                raise ValueError(
                    "Object-pose reset randomization requires exact collision-validated "
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

        if self.cfg.reset_position_randomization_enabled:
            if selected_clearance is None:
                raise RuntimeError("Collision clearance was not selected for object translation.")
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

        if self.cfg.reset_object_yaw_randomization_enabled:
            if selected_clearance is None:
                raise RuntimeError("Collision clearance was not selected for object yaw.")
            if not (
                0.0
                <= float(self.cfg.reset_object_yaw_fraction_min)
                <= float(self.cfg.reset_object_yaw_fraction_max)
                <= 1.0
            ):
                raise ValueError("Object-yaw reset fractions must satisfy 0 <= min <= max <= 1.")
            requested_yaw_profile = yaw_offset_profile(
                progress,
                far_yaw_rad=float(self.cfg.reset_object_yaw_far_rad),
                near_yaw_rad=float(self.cfg.reset_object_yaw_near_rad),
                exponent=float(self.cfg.reset_object_yaw_exponent),
            )
            yaw_fraction_samples = torch.empty(count, device=self.device).uniform_(
                float(self.cfg.reset_object_yaw_fraction_min),
                float(self.cfg.reset_object_yaw_fraction_max),
            )
            yaw_zero_offset = positive_reset
            if self.cfg.training_reset_mixture_enabled:
                path_reset = reset_modes == RESET_MODE_PATH
                requested_yaw_profile = torch.where(
                    path_reset,
                    requested_yaw_profile * curriculum.perturbation_scale,
                    torch.zeros_like(requested_yaw_profile),
                )
                yaw_zero_offset = ~path_reset
            object_yaw_offset, object_yaw_requested, object_yaw_safe_cap = (
                sample_collision_safe_yaw_offsets_from_profile(
                    requested_yaw_profile,
                    selected_clearance,
                    torch.linalg.norm(position_offset, dim=-1),
                    self.part_xy_rotation_radii[selected_part_indices],
                    yaw_zero_offset,
                    minimum_collision_clearance_m=(self.rotation_reset_minimum_collision_clearance_m),
                    clearance_guard_m=float(self.cfg.reset_position_clearance_guard_m),
                    magnitude_unit_samples=yaw_fraction_samples,
                )
            )

        # Apply one rigid planar delta to the physical object and the final TCP
        # target expressed in its part frame. The robot remains at the exact
        # validated waypoint and the canonical goal RGB-D remains unchanged.
        nominal_object_position = self.object_positions_catalog[target_indices]
        nominal_object_quaternion = self.object_quaternions_catalog[target_indices]
        nominal_goal_position = self.goal_tcp_positions_catalog[target_indices]
        nominal_goal_quaternion = self.goal_tcp_quaternions_catalog[target_indices]
        (
            moved_object_position,
            moved_object_quaternion,
            moved_goal_position,
            moved_goal_quaternion,
        ) = apply_planar_object_pose_delta(
            nominal_object_position,
            nominal_object_quaternion,
            nominal_goal_position,
            nominal_goal_quaternion,
            position_offset,
            object_yaw_offset,
        )
        self.goal_tcp_position[env_ids] = moved_goal_position + self.scene.env_origins[env_ids]
        self.goal_tcp_quaternion[env_ids] = moved_goal_quaternion
        object_pose = torch.cat(
            (
                moved_object_position + self.scene.env_origins[env_ids],
                moved_object_quaternion,
            ),
            dim=-1,
        )
        if bool(self.cfg.lift_reward_enabled):
            self.lift_fixture_position[env_ids] = object_pose[:, :3]
            self.lift_fixture_quaternion[env_ids] = object_pose[:, 3:7]
        zero_velocity = torch.zeros((count, 6), dtype=torch.float32, device=self.device)
        parked_pose = torch.zeros((count, 7), dtype=torch.float32, device=self.device)
        parked_pose[:, :3] = self.scene.env_origins[env_ids]
        parked_pose[:, 2] -= 10.0
        parked_pose[:, 3] = 1.0
        self._enable_only_selected_part_simulation(env_ids)
        for part_index, part in enumerate(self.parts):
            part_pose = parked_pose.clone()
            active = selected_part_indices == part_index
            part_pose[active] = object_pose[active]
            part.write_root_pose_to_sim(part_pose, env_ids=env_ids)
            if hasattr(part, "write_root_velocity_to_sim"):
                part.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        self._set_part_gravity_disabled(env_ids, disabled=True)

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
        self.applied_action_delta[env_ids] = 0.0
        self.filtered_motion_actions[env_ids] = 0.0
        self.motion_action_history[:, env_ids] = 0.0
        self.live_observation_history_valid[env_ids] = False
        self.completion_probability[env_ids] = 0.0
        self.completion_stop_candidate[env_ids] = False
        self.completion_streak[env_ids] = 0
        self.completion_declaration[env_ids] = False
        self.lift_phase[env_ids] = self._LIFT_PHASE_APPROACH
        self.lift_phase_step[env_ids] = 0
        self.lift_policy_transition[env_ids] = True
        self.lift_commit_event[env_ids] = False
        self.lift_commit_operational[env_ids] = False
        self.lift_commit_strict[env_ids] = False
        self.lift_commit_quality[env_ids] = 0.0
        self.lift_commit_position_error[env_ids] = 0.0
        self.lift_commit_rotation_error[env_ids] = 0.0
        self.lift_arm_target[env_ids] = q
        self.lift_prelift_object_position[env_ids] = object_pose[:, :3]
        self.lift_prelift_tcp_position[env_ids] = 0.0
        self.lift_initial_relative_position[env_ids] = 0.0
        self.lift_peak_object_z[env_ids] = object_pose[:, 2]
        self.lift_final_height[env_ids] = 0.0
        self.lift_peak_height[env_ids] = 0.0
        self.lift_relative_drift[env_ids] = 0.0
        self.lift_quality[env_ids] = 0.0
        self.lift_pickup_success[env_ids] = False
        self.lift_arm_ok[env_ids] = False
        self.ever_operational_ready[env_ids] = False
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
        self.reset_object_yaw_offset[env_ids] = object_yaw_offset
        self.reset_object_yaw_requested[env_ids] = object_yaw_requested
        self.reset_object_yaw_safe_cap[env_ids] = object_yaw_safe_cap
        self.reset_goal_position_delta[env_ids] = moved_goal_position - nominal_goal_position
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
            if self.live_workspace_appearance_randomizer is not None:
                self.live_workspace_appearance_randomizer.sample(
                    env_ids,
                    strength=self.live_observation_randomizer.randomization_strength[env_ids],
                    palette_indices=forced_live_palette_indices,
                )
            self._sample_sim2real_dynamics(env_ids)
