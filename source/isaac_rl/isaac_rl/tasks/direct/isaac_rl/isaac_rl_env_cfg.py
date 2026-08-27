"""Configuration for the KUKA wrist-camera grasp alignment RL task."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[7]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.d405_wrist_camera import (
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
    D405WristCameraConfig,
    camera_pose_in_link7,
)
from grasp_planning.envs.fr3_part_env import make_fr3_part_scene_cfg
from grasp_planning.isaac_visual_scene import make_visual_servo_render_cfg
from grasp_planning.rl.policy_context import POLICY_CONTEXT_ACTION
from grasp_planning.rl.policy_timing import PHYSICS_RATE_HZ, POLICY_DECIMATION

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

ROBOT_USD = REPO_ROOT / "assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd"
PART_USD = REPO_ROOT / "artifacts/isaac_bundle_assets/pipeline_stage2_ground_feasible_bundle_local.usd"
LEGACY_GOAL_RESET_DATA = REPO_ROOT / "isaac_rl/data/fixed_goal_reset.npz"
MULTIGRASP_CATALOG_DATA = REPO_ROOT / "isaac_rl/data/multigrasp_50_catalog.npz"
MULTIGRASP_ROTATION_RESET_DATA = REPO_ROOT / "isaac_rl/data/multigrasp_50_rotation_resets.npz"
MULTIPART_DATA_ROOT = REPO_ROOT / "isaac_rl/data/plumbers_block"
MULTIPART_CATALOG_DATA = MULTIPART_DATA_ROOT / "goal_catalog.npz"
MULTIPART_ROTATION_RESET_DATA = MULTIPART_DATA_ROOT / "rotation_resets.npz"
MULTIPART_PART_IDS = ("0", "1", "2", "3", "4")
MULTIPART_PART_USDS = tuple(
    MULTIPART_DATA_ROOT / "usd" / f"part_{part_id}_bundle_local.usd" for part_id in MULTIPART_PART_IDS
)

OBJECT_POSITION_W = (0.4252643585205078, 0.05988234281539917, 0.02475000304188643)
OBJECT_ORIENTATION_XYZW = (-0.0, 0.0, -0.02874958897323683, 0.9995866451358131)
GOAL_GRASP_POSITION_W = (0.4272937326504249, 0.09513242649798231, 0.03501698252776764)
GOAL_GRASP_ORIENTATION_XYZW = (
    0.715167442648851,
    0.6977595362616698,
    0.028513895341512947,
    0.029225268235817444,
)
_scene_assets = make_fr3_part_scene_cfg(
    fr3_asset_path=str(ROBOT_USD),
    part_usd_path=str(PART_USD),
    part_position=OBJECT_POSITION_W,
    part_orientation_xyzw=OBJECT_ORIENTATION_XYZW,
    part_density_kg_m3=1240.0,
)
_scene_assets.robot.prim_path = "/World/envs/env_.*/Robot"
_scene_assets.robot.spawn.activate_contact_sensors = True
_scene_assets.part.prim_path = "/World/envs/env_.*/Part"
_scene_assets.part.spawn.rigid_props.kinematic_enabled = True
_camera_cfg = D405WristCameraConfig(enabled=True)
_camera_position, _camera_orientation_wxyz = camera_pose_in_link7(_camera_cfg)


@configclass
class GraspVisualServoEnvCfg(DirectRLEnvCfg):
    # One policy action and one wrist RGB-D observation every 1/15 second.
    decimation = POLICY_DECIMATION
    # Maximum training horizon. Individual resets receive shorter timeouts
    # according to path progress and whether they train completion/boundaries.
    episode_length_s = 12.0
    # Six normalized camera-frame velocities plus one Bernoulli completion
    # decision. The custom RL-Games model treats only the first six values as
    # Gaussian actions.
    action_space = 7
    # Flattened 72x128x8 visual input, deployment-measurable actor context, six
    # privileged pose targets, one completion label, and one supervision mask.
    # The network excludes the final eight labels from its action path; the
    # preceding action is deployment-available temporal context.
    observation_space = VISUAL_SERVO_OBSERVATION_HEIGHT * VISUAL_SERVO_OBSERVATION_WIDTH * 8 + 14
    # Training entrypoints update the observation/network sizes together when
    # selecting a larger ablation context.
    policy_context_mode = POLICY_CONTEXT_ACTION
    # The critic remains q, qd, privileged pose error, and the preceding six
    # motion actions. The completion decision is deliberately omitted.
    state_space = 26

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / PHYSICS_RATE_HZ,
        render_interval=decimation,
        render=make_visual_servo_render_cfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)
    viewer = ViewerCfg(eye=(1.25, 1.25, 1.0), lookat=(0.4, 0.05, 0.15))

    robot_cfg = _scene_assets.robot
    part_cfg = _scene_assets.part
    part_names = ("0",)
    part_usd_paths = (str(PART_USD),)
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/link7/D405LeftCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=_camera_position,
            rot=_camera_orientation_wxyz,
            convention="ros",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=_camera_cfg.intrinsic_matrix_row_major,
            width=_camera_cfg.width,
            height=_camera_cfg.height,
            clipping_range=_camera_cfg.clipping_range_m,
        ),
        width=VISUAL_SERVO_RENDER_WIDTH,
        height=VISUAL_SERVO_RENDER_HEIGHT,
    )
    # Optional fixed external camera used only by the composite debug recorder.
    # It is not instantiated during training or ordinary playback, so it adds
    # no rendering or VRAM cost to those paths.
    debug_camera_enabled = False
    debug_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera",
        offset=TiledCameraCfg.OffsetCfg(
            # OpenGL camera looking from the robot's negative-Y side toward
            # the workspace center. Authoring this pose avoids a Fabric/USD
            # synchronization delay when recording immediately after reset.
            pos=(0.35, -0.90, 0.65),
            rot=(0.81386772, 0.57953072, 0.02435819, 0.03420758),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 10.0),
        ),
        width=640,
        height=360,
    )

    goal_catalog_data_path = str(MULTIGRASP_CATALOG_DATA)
    rotation_reset_data_path = str(MULTIGRASP_ROTATION_RESET_DATA)
    catalog_split = "all"
    legacy_goal_reset_data_path = str(LEGACY_GOAL_RESET_DATA)
    # Training must fail loudly until the 50 targets have both a MoveIt path
    # and an actual Isaac goal rendering. This prevents accidentally launching
    # another overnight single-goal run after requesting multi-grasp training.
    require_multigrasp_catalog = True
    require_rotation_reset_data = True
    fixed_target_index = -1
    fixed_target_id = ""
    # Training uses balanced coverage. Interactive playback may opt into
    # independent random target draws so consecutive resets need not follow
    # catalog order and repeats remain possible.
    random_target_sampling = False
    # Evaluation-only exact catalog sweep. Requires num_envs == target_count.
    sequential_target_sampling = False
    object_position_w = OBJECT_POSITION_W
    object_orientation_xyzw = OBJECT_ORIENTATION_XYZW
    goal_grasp_position_w = GOAL_GRASP_POSITION_W
    goal_grasp_orientation_xyzw = GOAL_GRASP_ORIENTATION_XYZW
    # Sample nearly the complete nominal pregrasp-to-grasp path continuously.
    # The exact goal is excluded, while a small number of perturbed near-goal
    # samples can still enter tolerance and bootstrap terminal behavior.
    reset_progress_min = 0.0
    reset_progress_max = 0.94
    reset_joint_noise_far_rad = 0.040
    reset_joint_noise_near_rad = 0.003
    reset_joint_noise_exponent = 1.5
    # Position-preserving TCP rotation paths provide substantially stronger
    # orientation variation than raw joint noise. At pregrasp they span
    # 7.5--15 degrees; their authored angle tapers to five degrees near the
    # grasp. Sampling 0.5--1.0 of that magnitude makes close states span
    # 2.5--5 degrees, straddling the four-degree clear-negative boundary and
    # explicitly teaching the final rotational correction.
    reset_rotation_randomization_enabled = True
    reset_rotation_fraction_min = 0.5
    reset_rotation_fraction_max = 1.0
    reset_rotation_far_rad = 15.0 * 3.141592653589793 / 180.0
    # Horizontal translation moves the physical object and its part-relative
    # target together, while the robot stays at its nominal/rotated waypoint.
    # Runtime sampling caps every displacement by the collision clearance
    # stored for that exact target, rotation variant, and waypoint.
    reset_position_randomization_enabled = False
    reset_position_far_offset_m = 0.010
    reset_position_near_offset_m = 0.003
    reset_position_offset_exponent = 1.5
    reset_position_fraction_min = 0.5
    reset_position_fraction_max = 1.0
    reset_position_clearance_guard_m = 0.0001
    # In-plane object yaw models error between the nominal perceived part frame
    # and the stable actual part pose rather than merely rotating the gripper.
    # Actual poses stay on the catalog support manifold: world Z and roll/pitch
    # never change. The target TCP follows the same rigid transform and the
    # canonical goal RGB-D remains unchanged. The sampler conservatively
    # budgets maximum surface displacement against validated clearance.
    reset_object_yaw_randomization_enabled = False
    reset_object_yaw_far_rad = 10.0 * 3.141592653589793 / 180.0
    reset_object_yaw_near_rad = 3.0 * 3.141592653589793 / 180.0
    reset_object_yaw_exponent = 1.5
    reset_object_yaw_fraction_min = 0.0
    reset_object_yaw_fraction_max = 1.0
    # Maximum XY vertex radius about the USD/part-frame root, rounded upward
    # from the scaled source OBJ. Multipart overrides this tuple per part.
    part_xy_rotation_radii_m = (0.046,)
    # Continuous path/error resets remain the majority, while explicit
    # unperturbed, ready-region, and boundary cases prevent the completion
    # classifier from being starved by a high-dimensional uniform sampler.
    training_reset_mixture_enabled = False
    reset_no_noise_fraction = 0.15
    reset_ready_fraction = 0.15
    reset_boundary_fraction = 0.15
    reset_ready_exact_fraction = 0.25
    reset_ready_position_max_m = 0.0035
    reset_boundary_waypoint_count = 3
    reset_boundary_rotation_fraction = 0.50
    # Expand close nominal resets into the full path, perturbation, appearance,
    # and failure-replay distribution over roughly 3k 64-step PPO epochs.
    training_curriculum_enabled = False
    curriculum_warmup_steps = 16_000
    curriculum_full_steps = 192_000
    curriculum_initial_progress_min = 0.70
    curriculum_final_progress_min = 0.0
    # Per-reset time budgets at 15 Hz. Completion-focused resets are short;
    # far resets are long enough to exhibit the 7--8 s successes seen in eval.
    reset_timeout_far_s = 12.0
    reset_timeout_close_s = 4.0
    reset_timeout_ready_s = 1.5
    reset_timeout_boundary_s = 2.5
    reset_timeout_exponent = 1.0
    variable_reset_timeouts_enabled = False
    # Part-balanced hard-target replay is learned online from terminal outcomes.
    failure_replay_fraction = 0.25
    failure_replay_score_floor = 0.10
    failure_replay_score_power = 1.5
    failure_score_decay = 0.90
    # Training randomizes only the live RGB-D observation. Goal catalog images
    # remain deterministic canonical references, matching deployment where a
    # real D405 live frame is compared with a synthetic catalog goal.
    live_observation_randomization_enabled = True
    live_rgb_exposure_stops = (-0.30, 0.30)
    live_rgb_contrast = (0.85, 1.15)
    live_rgb_gamma = (0.90, 1.10)
    live_rgb_white_balance_gain = (0.93, 1.07)
    live_rgb_vignette_strength = (0.0, 0.14)
    live_rgb_noise_std = (0.0, 0.015)
    live_rgb_blur_probability = 0.12
    live_rgb_blur_mix = (0.25, 0.60)
    live_depth_scale = (0.99, 1.01)
    live_depth_bias_m = (-0.002, 0.002)
    live_depth_noise_std_m = (0.0, 0.0002)
    # Provisional documented D405/D400 profile. Device-specific plane captures
    # will replace these ranges without changing the observation contract.
    sim2real_randomization_profile = "d405_documented_provisional_v6_15hz:combined_sim2real"
    live_correlated_depth_enabled = True
    live_stereo_focal_length_px = _camera_cfg.fx
    live_stereo_baseline_m = _camera_cfg.stereo_baseline_m
    live_disparity_bias_px = (-0.04, 0.04)
    live_disparity_independent_noise_std_px = (0.01, 0.03)
    live_disparity_spatial_noise_std_px = (0.02, 0.07)
    live_disparity_temporal_noise_std_px = (0.01, 0.04)
    live_disparity_temporal_correlation = (0.60, 0.92)
    live_stereo_edge_mismatch_probability = 0.12
    live_stereo_edge_horizontal_radius_px = 2
    live_depth_quantization_m = _camera_cfg.depth_unit_m
    live_depth_dropout_probability = (0.0, 0.004)
    live_depth_edge_dropout_probability = (0.0, 0.035)
    live_depth_edge_threshold_m = 0.008
    live_rgb_patch_occlusion_probability = 0.06
    live_depth_patch_dropout_probability = 0.04
    live_patch_area_fraction = (0.005, 0.03)
    live_calibration_warp_enabled = True
    live_calibration_shift_x_px = (-1.5, 1.5)
    live_calibration_shift_y_px = (-1.0, 1.0)
    live_calibration_scale = (0.99, 1.01)
    live_calibration_roll_deg = (-1.0, 1.0)
    live_clean_episode_fraction = 0.15
    # Camera-frame and controller timing at 15 Hz. The completion hold is
    # immediate; only live observations and six motion components are delayed.
    # One 15 Hz step preserves the former maximum ~67 ms delay.
    live_observation_delay_max_steps = 1
    live_observation_repeat_probability = 0.02
    motion_action_delay_max_steps = 1
    motion_action_two_step_probability = 0.0
    motion_response_scale = (0.88, 1.12)
    # 0.91 at 15 Hz has the same time response as 0.70 at 30 Hz.
    motion_response_alpha = (0.91, 1.0)
    motion_bias = (-0.015, 0.015)
    physics_joint_stiffness_scale = (0.90, 1.10)
    physics_joint_damping_scale = (0.90, 1.10)
    # Change the physical live scene as well as applying sensor-space noise.
    # Rotating the distant key light changes the cast-shadow direction; the
    # slow cadence keeps each appearance stable for four seconds at 15 Hz.
    scene_appearance_randomization_enabled = True
    scene_appearance_randomization_interval_steps = 60
    # A half-scale T-slot is canonical render/depth geometry. The exact
    # collision surface remains the unchanged flat z=0 plane.
    scene_tslot_surface_enabled = True
    scene_tslot_geometry_randomization_enabled = True
    scene_tslot_nominal_fraction = 0.60
    scene_tslot_phase_fraction = 0.20
    # Optional render/depth-only peripheral props. They deliberately carry no
    # collision schema and sit outside the nominal target/approach corridor;
    # PhysX continues to use the same flat z=0 workspace plane.
    scene_clutter_enabled = False
    scene_clutter_environment_fraction = 0.0
    scene_clutter_min_objects = 1
    scene_clutter_max_objects = 3
    # Optional larger render/depth-only office/factory layer behind the task.
    # It contains low-poly walls, storage, screens, safety frames, tabletop
    # cables, and multiple people, while collision remains the flat plane.
    scene_busy_background_enabled = False
    scene_busy_background_environment_fraction = 0.0
    scene_busy_background_min_people = 2
    scene_busy_background_max_people = 4
    scene_key_yaw_delta_deg = (-35.0, 35.0)
    scene_key_pitch_delta_deg = (-15.0, 15.0)
    scene_key_intensity_scale = (0.70, 1.30)
    scene_key_angle_deg = (5.0, 12.0)
    scene_dome_intensity_scale = (0.75, 1.25)
    scene_light_temperature_shift = (-0.08, 0.08)
    scene_part_color_scale = (0.90, 1.10)
    scene_part_saturation_scale = (0.90, 1.10)
    scene_part_hue_shift_deg = (-5.0, 5.0)
    scene_part_roughness = (0.65, 0.90)
    scene_tslot_color_scale = (0.88, 1.12)
    scene_tslot_saturation_scale = (0.90, 1.10)
    scene_tslot_hue_shift_deg = (-5.0, 5.0)
    scene_tslot_roughness_delta = (-0.08, 0.08)
    scene_finger_color_scale = (0.80, 1.20)
    scene_ground_color_scale = (0.75, 1.25)
    scene_ground_hue_shift_deg = (-12.0, 12.0)
    scene_ground_roughness = (0.75, 1.0)
    linear_action_scale_m_s = 0.04
    angular_action_scale_rad_s = 0.24
    # Preserve the former 7.5 normalized-units/s slew limit at 15 Hz.
    action_delta_limit = 0.50
    dls_damping = 0.08
    # Privileged completion ground truth is used only for training and
    # evaluation. The strict positive and clear negative thresholds leave an
    # ignored ambiguity band, preventing contradictory labels near the edge.
    completion_ready_position_m = 0.004
    completion_ready_rotation_rad = 3.0 * 3.141592653589793 / 180.0
    completion_negative_position_m = 0.006
    completion_negative_rotation_rad = 4.0 * 3.141592653589793 / 180.0
    # Training and deployment use the same high-confidence, multi-frame,
    # low-speed gate. The auxiliary classifier supplies dense probability
    # supervision while PPO retains control of the actual stop decision.
    completion_probability_threshold = 0.95
    completion_required_consecutive_steps = 4
    completion_max_linear_speed_m_s = 0.005
    completion_max_angular_speed_rad_s = 0.03
    completion_positive_reset_fraction = 0.15
    unsafe_contact_force_threshold_n = 1.0
    collision_risk_force_threshold_n = 0.05
    collision_risk_penalty_weight = 2.0
    divergence_position_m = 0.18
    position_progress_weight = 300.0
    rotation_progress_weight = 30.0
    position_precision_weight = 4.0
    position_precision_scale_m = 0.020
    rotation_precision_weight = 2.0
    rotation_precision_scale_rad = 0.12
    completion_correct_reward = 50.0
    completion_positive_terminal_reward_scale = 0.20
    completion_premature_penalty = 50.0
    unsafe_collision_penalty = 50.0
    timeout_penalty = 15.0
    divergence_penalty = 25.0
    step_penalty = 0.02
    action_penalty_weight = 0.002
    auxiliary_position_scale_m = 0.10
    auxiliary_rotation_scale_rad = 0.35


@configclass
class GraspVisualServoEnvCfg_PLAY(GraspVisualServoEnvCfg):
    episode_length_s = 15.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    # Playback remains a full-distance stress test rather than starting from a
    # randomly selected easy point in the approach curriculum.
    reset_progress_min = 0.0
    reset_progress_max = 0.0
    reset_joint_noise_far_rad = 0.08
    # Make playback an explicit maximum-rotation test. CLI overrides can set a
    # smaller fixed magnitude without rebuilding the reset asset.
    reset_rotation_fraction_min = 1.0
    reset_rotation_fraction_max = 1.0
    # Preserve playback of the existing single-goal checkpoints while the
    # validated 50-goal catalog is being generated. Once the catalog exists it
    # is still loaded automatically.
    require_multigrasp_catalog = False
    require_rotation_reset_data = False
    live_observation_randomization_enabled = False
    scene_appearance_randomization_enabled = False
    scene_tslot_geometry_randomization_enabled = False
    scene_tslot_surface_enabled = True
    live_observation_delay_max_steps = 0
    live_observation_repeat_probability = 0.0
    motion_action_delay_max_steps = 0
    motion_action_two_step_probability = 0.0
    motion_response_scale = (1.0, 1.0)
    motion_response_alpha = (1.0, 1.0)
    motion_bias = (0.0, 0.0)
    physics_joint_stiffness_scale = (1.0, 1.0)
    physics_joint_damping_scale = (1.0, 1.0)
    completion_positive_reset_fraction = 0.0
    completion_probability_threshold = 0.95
    completion_required_consecutive_steps = 4
    completion_max_linear_speed_m_s = 0.005
    completion_max_angular_speed_rad_s = 0.03


@configclass
class GraspVisualServoMultiPartEnvCfg(GraspVisualServoEnvCfg):
    """Training configuration for all five plumbers-block parts."""

    goal_catalog_data_path = str(MULTIPART_CATALOG_DATA)
    rotation_reset_data_path = str(MULTIPART_ROTATION_RESET_DATA)
    catalog_split = "train"
    part_names = MULTIPART_PART_IDS
    part_usd_paths = tuple(str(path) for path in MULTIPART_PART_USDS)
    require_multigrasp_catalog = True
    require_rotation_reset_data = True
    # The narrowed, object-specific approach aperture makes arbitrary joint
    # interpolation unsafe. Sample only exact reset waypoints/rotation variants
    # that the rotation-asset builder validated against the part and ground.
    reset_collision_safe_sampling_enabled = True
    reset_joint_noise_far_rad = 0.0
    reset_joint_noise_near_rad = 0.0
    reset_rotation_fraction_min = 1.0
    reset_rotation_fraction_max = 1.0
    reset_position_randomization_enabled = True
    reset_position_fraction_min = 0.0
    reset_position_fraction_max = 1.0
    reset_object_yaw_randomization_enabled = True
    part_xy_rotation_radii_m = (0.046, 0.055, 0.084, 0.055, 0.055)
    training_reset_mixture_enabled = True
    training_curriculum_enabled = True
    variable_reset_timeouts_enabled = True


@configclass
class GraspVisualServoMultiPartEnvCfg_PLAY(GraspVisualServoMultiPartEnvCfg):
    """Held-out-test playback configuration for the multi-part policy."""

    episode_length_s = 15.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    catalog_split = "test"
    reset_progress_min = 0.0
    reset_progress_max = 0.0
    reset_joint_noise_far_rad = 0.0
    reset_joint_noise_near_rad = 0.0
    reset_rotation_fraction_min = 1.0
    reset_rotation_fraction_max = 1.0
    reset_position_fraction_min = 1.0
    reset_position_fraction_max = 1.0
    reset_object_yaw_fraction_min = 1.0
    reset_object_yaw_fraction_max = 1.0
    live_observation_randomization_enabled = False
    scene_appearance_randomization_enabled = False
    scene_tslot_geometry_randomization_enabled = False
    scene_tslot_surface_enabled = True
    live_observation_delay_max_steps = 0
    live_observation_repeat_probability = 0.0
    motion_action_delay_max_steps = 0
    motion_action_two_step_probability = 0.0
    motion_response_scale = (1.0, 1.0)
    motion_response_alpha = (1.0, 1.0)
    motion_bias = (0.0, 0.0)
    physics_joint_stiffness_scale = (1.0, 1.0)
    physics_joint_damping_scale = (1.0, 1.0)
    completion_positive_reset_fraction = 0.0
    training_reset_mixture_enabled = False
    training_curriculum_enabled = False
    variable_reset_timeouts_enabled = False
    failure_replay_fraction = 0.0
    completion_probability_threshold = 0.95
    completion_required_consecutive_steps = 4
    completion_max_linear_speed_m_s = 0.005
    completion_max_angular_speed_rad_s = 0.03
