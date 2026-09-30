"""Clutter-v5 training semantics adapted to validated Panda pose resets.

All perturbations are selected from measured collision-tested bank entries.
No unchecked joint interpolation or zero-error fallback is used for pose resets.
"""

import math
import os

import torch

from .completion import completion_terminal_reward
from .training_curriculum import (
    apply_failure_replay,
    curriculum_state,
    reset_timeout_seconds,
    sample_reset_modes,
    update_failure_scores,
)


def boundary_pool(valid, kinds, progress, rotation):
    """Separate near-goal position and orientation classification examples."""
    position = valid & (kinds[None, :] == 2)
    orientation = valid & (kinds[None, :] == 3)
    fallback = valid & (kinds[None, :] == 0) & ((progress[None, :] - 0.94).abs() < 1e-5)
    position = torch.where(position.any(-1)[:, None], position, fallback)
    orientation = torch.where(orientation.any(-1)[:, None], orientation, fallback)
    selected = torch.where(rotation[:, None], orientation, position)
    if not selected.any(-1).all():
        raise RuntimeError("Missing validated position/orientation boundary starts")
    return selected


def ready_pool(valid, kinds, exact):
    """Exact starts are allowed only after their own physics/contact validation."""
    ready = valid & (kinds[None, :] == 1)
    goal = valid & (kinds[None, :] == 4)
    ready = torch.where(ready.any(-1)[:, None], ready, goal)
    selected = torch.where((exact & goal.any(-1))[:, None], goal, ready)
    if not selected.any(-1).all():
        raise RuntimeError("Target lacks a collision-validated positive reset")
    return selected


def nominal_pool(valid, kinds, progress, minimum):
    allowed = valid & (kinds[None, :] == 5) & (progress[None, :] <= 0.94)
    close = allowed & (progress[None, :] >= minimum)
    selected = torch.where(close.any(-1)[:, None], close, allowed)
    if not selected.any(-1).all():
        raise RuntimeError("Missing collision-validated nominal approach states")
    return selected


class FrankaPoseRecipeMixin:
    def _init_pose_recipe(self, data, selected):
        self.recipe = dict(self.cfg.training_recipe["source_settings"])
        # Preserve source physical-time behavior at the requested policy rate.
        self.time_ratio = self.step_dt / (1 / 30)
        self.recipe["completion_required_consecutive_steps"] = math.ceil(
            self.recipe["completion_required_consecutive_steps"] / self.time_ratio
        )
        self.recipe["action_delta_limit"] *= self.time_ratio
        self.recipe["motion_response_alpha"] = [
            1 - (1 - x) ** self.time_ratio for x in self.recipe["motion_response_alpha"]
        ]
        for name in ("joints", "valid", "position_error_m", "rotation_error_rad", "lateral_m", "contact_n"):
            key = "pose_reset_" + name
            value = data[key][selected]
            setattr(
                self,
                key,
                torch.as_tensor(value, device=self.device, dtype=torch.bool if name == "valid" else torch.float32),
            )
        self.placement_object_poses = self.placement_goal_poses = None
        if self.cfg.placement_randomization:
            from grasp_planning.rl.franka_placement import PLACEMENT_PROFILE, validate_placement_poses

            if self.cfg.placement_randomization != PLACEMENT_PROFILE:
                raise ValueError("Unsupported independent-placement profile")
            for name in ("object_poses", "goal_poses"):
                value = data["placement_" + name][selected]
                validate_placement_poses(value, data["pose_reset_valid"][selected])
                setattr(self, "placement_" + name, torch.as_tensor(value, device=self.device))
        self.pose_reset_progress = torch.as_tensor(data["pose_reset_progress"], device=self.device)
        if "pose_reset_kind" not in data:
            raise ValueError("Rebuild the v1 pose bank: its boundary pool contains only positive starts")
        self.pose_reset_kind = torch.as_tensor(data["pose_reset_kind"], device=self.device)
        assert (self.pose_reset_kind == 5).any(), "Rebuild the bank with nominal-path physics validation"
        assert (self.pose_reset_valid & (self.pose_reset_kind[None, :] == 0)).any(-1).all(), (
            "Missing perturbed reset coverage"
        )
        self.reset_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_progress = torch.zeros(self.num_envs, device=self.device)
        self.reset_bank_index = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.initial_position_error = torch.zeros(self.num_envs, device=self.device)
        self.initial_rotation_error = torch.zeros_like(self.initial_position_error)
        self.previous_position_error = torch.zeros_like(self.initial_position_error)
        self.previous_rotation_error = torch.zeros_like(self.initial_position_error)
        self.pose_timeout_steps = torch.full(
            (self.num_envs,), int(12 / self.step_dt), device=self.device, dtype=torch.long
        )
        self.completion_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.completion_declaration = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.target_failure_scores = torch.zeros(len(selected), device=self.device)
        self.target_groups = torch.as_tensor(data["target_part_indices"][selected], device=self.device)
        self.motion_history = torch.zeros((3, self.num_envs, 6), device=self.device)
        self.motion_filter = torch.zeros((self.num_envs, 6), device=self.device)
        self.motion_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_scale = torch.ones((self.num_envs, 1), device=self.device)
        self.motion_alpha = torch.ones((self.num_envs, 1), device=self.device)
        self.motion_bias = torch.zeros((self.num_envs, 6), device=self.device)
        self.observation_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.live_history = torch.zeros(
            (3, self.num_envs, self.camera_profile["observation_height"], self.camera_profile["observation_width"], 4),
            device=self.device,
        )
        self.live_history_empty = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.pose_curriculum_offset = 0
        from dataclasses import fields

        from grasp_planning.rl.live_observation_randomization import (
            LiveObservationRandomizationCfg,
            LiveObservationRandomizer,
        )

        sensor = {}
        aliases = {
            "exposure_stops": "live_rgb_exposure_stops",
            "contrast": "live_rgb_contrast",
            "gamma": "live_rgb_gamma",
            "white_balance_gain": "live_rgb_white_balance_gain",
            "vignette_strength": "live_rgb_vignette_strength",
            "rgb_noise_std": "live_rgb_noise_std",
            "blur_probability": "live_rgb_blur_probability",
            "blur_mix": "live_rgb_blur_mix",
        }
        for f in fields(LiveObservationRandomizationCfg):
            key = aliases.get(f.name, "live_" + f.name)
            if key in self.recipe:
                sensor[f.name] = self.recipe[key]
        # Preserve generic corruptions, not D405-specific disparity statistics.
        sensor.update(
            correlated_depth_enabled=False,
            depth_structured_dropout_probability=0.0,
            stereo_focal_length_px=self.camera_profile["fx"]
            * self.camera_profile["observation_width"]
            / self.camera_profile["source_width"],
            stereo_baseline_m=self.camera_profile["stereo_baseline_m"],
            depth_min_m=self.camera_profile["depth_min_m"],
            depth_max_m=self.camera_profile["depth_max_m"],
        )
        self.live_randomizer = LiveObservationRandomizer(
            LiveObservationRandomizationCfg(**sensor), num_envs=self.num_envs, device=self.device
        )

    def _pose_curriculum(self):
        r = self.recipe
        return curriculum_state(
            int(
                (self.common_step_counter + self.pose_curriculum_offset)
                * self.num_envs
                * int(os.environ.get("WORLD_SIZE", "1"))
                / 512
            ),
            enabled=not self.cfg.pose_evaluation,
            warmup_steps=int(r["curriculum_warmup_steps"]),
            full_steps=int(r["curriculum_full_steps"]),
            initial_progress_min=r["curriculum_initial_progress_min"],
            final_progress_min=0.0,
            final_failure_replay_fraction=r["failure_replay_fraction"],
        )

    def _pose_reset(self, env_ids, target):
        r = self.recipe
        c = self._pose_curriculum()
        n = len(env_ids)
        if not self.cfg.pose_evaluation:
            target, _ = apply_failure_replay(
                target,
                target_group_indices=self.target_groups,
                failure_scores=self.target_failure_scores,
                replay_fraction=c.failure_replay_fraction,
                score_floor=r["failure_replay_score_floor"],
                score_power=r["failure_replay_score_power"],
            )
        modes = sample_reset_modes(
            torch.rand(n, device=self.device),
            no_noise_fraction=r["reset_no_noise_fraction"],
            ready_fraction=r["reset_ready_fraction"],
            boundary_fraction=r["reset_boundary_fraction"],
        )
        if self.cfg.pose_evaluation:
            modes.zero_()
        eligible = self.pose_reset_valid[target].clone()
        if self.cfg.pose_evaluation:
            distances = (self.pose_reset_progress - float(self.cfg.pose_evaluation_progress)).abs()
            eligible &= (distances[None, :] < 1e-5) & (self.pose_reset_kind[None, :] == 0)
        else:
            path_mask = (
                (self.pose_reset_progress[None, :] >= c.progress_min)
                & (self.pose_reset_progress[None, :] <= 0.94)
                & (self.pose_reset_kind[None, :] == 0)
            )
            # When a close perturbation is infeasible, use a farther valid perturbation, never a nominal substitute.
            close = self.pose_reset_valid[target] & path_mask
            use_close = close.any(-1)
            eligible = torch.where(use_close[:, None], close, eligible & (self.pose_reset_kind[None, :] == 0))
            ready = ready_pool(
                self.pose_reset_valid[target],
                self.pose_reset_kind,
                torch.rand(n, device=self.device) < r["reset_ready_exact_fraction"],
            )
            eligible = torch.where((modes == 2)[:, None], ready, eligible)
            boundary = boundary_pool(
                self.pose_reset_valid[target],
                self.pose_reset_kind,
                self.pose_reset_progress,
                torch.rand(n, device=self.device) < r["reset_boundary_rotation_fraction"],
            )
            eligible = torch.where((modes == 3)[:, None], boundary, eligible)
        rotated = (
            (modes == 3) | ((modes == 0) & (torch.rand(n, device=self.device) < c.perturbation_scale)) | (modes == 2)
        )
        if self.cfg.pose_evaluation:
            rotated[:] = True
        else:
            nominal = nominal_pool(
                self.pose_reset_valid[target], self.pose_reset_kind, self.pose_reset_progress, c.progress_min
            )
            eligible = torch.where(rotated[:, None], eligible, nominal)
        if (rotated & ~eligible.any(-1)).any():
            raise RuntimeError("Requested pose condition lacks safe resets; refusing nominal fallback")
        selected = torch.multinomial(eligible.float(), 1).squeeze(-1)
        q = self.pose_reset_joints[target, selected]
        bank = selected
        progress = self.pose_reset_progress[selected]
        self.target_index[env_ids] = target
        goal = self.catalog["goal_poses"][target]
        obj = self.catalog["object_poses"][target]
        if self.placement_object_poses is not None:
            goal = self.placement_goal_poses[target, selected]
            obj = self.placement_object_poses[target, selected]
        self.goal_pose[env_ids] = goal
        self.goal_pose[env_ids, :3] += self.scene.env_origins[env_ids]
        # Canonical image never follows the live object's randomized world pose.
        self._sample_goal_images(env_ids, target)
        self.write_state(q, obj, env_ids, open_width=self.catalog["open_widths"][target])
        self.scene.reset(env_ids)
        self.sim.forward()
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.completion_streak[env_ids] = 0
        self.completion_declaration[env_ids] = False
        self.reset_mode[env_ids] = modes
        self.reset_progress[env_ids] = progress
        self.reset_bank_index[env_ids] = bank
        pe, re = self.pose_errors()
        pn = pe.norm(dim=-1)
        rn = re.norm(dim=-1)
        for key, value in [
            ("initial_position_error", pn),
            ("previous_position_error", pn),
            ("initial_rotation_error", rn),
            ("previous_rotation_error", rn),
        ]:
            getattr(self, key)[env_ids] = value[env_ids]
        if self.cfg.symmetry_training and self.cfg.symmetry_objective == "orbit_potential_pose_set_v2":
            self.previous_potential[env_ids] = self.symmetry_potential()[env_ids]
        seconds = reset_timeout_seconds(
            progress,
            modes,
            far_seconds=r["reset_timeout_far_s"],
            close_seconds=r["reset_timeout_close_s"],
            ready_seconds=r["reset_timeout_ready_s"],
            boundary_seconds=r["reset_timeout_boundary_s"],
            exponent=r["reset_timeout_exponent"],
        )
        if self.cfg.pose_evaluation:
            seconds[:] = 15.0
        self.pose_timeout_steps[env_ids] = (seconds / self.step_dt).ceil().long()
        self.rgb_gain[env_ids] = 1
        if self.appearance:
            self.appearance.apply_many(
                env_ids.cpu().tolist(), torch.randint(2**31 - 1, (n,), device=self.device).cpu().tolist()
            )
        self.live_randomizer.sample(
            env_ids, strength=0.0 if self.cfg.pose_evaluation else c.visual_randomization_strength
        )
        self.motion_history[:, env_ids] = 0
        self.motion_filter[env_ids] = 0
        self.live_history_empty[env_ids] = True
        strength = 0.0 if self.cfg.pose_evaluation else c.visual_randomization_strength
        self.motion_delay[env_ids] = torch.randint(2, (n,), device=self.device)
        two = torch.rand(n, device=self.device) < r["motion_action_two_step_probability"]
        self.motion_delay[env_ids[two]] = 2
        self.motion_delay[env_ids] = (self.motion_delay[env_ids].float() / self.time_ratio).round().long()
        self.observation_delay[env_ids] = (
            (torch.randint(3, (n,), device=self.device).float() / self.time_ratio).round().long()
        )
        if strength == 0:
            self.motion_delay[env_ids] = 0
            self.observation_delay[env_ids] = 0
        for name, key, center in [
            ("motion_scale", "motion_response_scale", 1.0),
            ("motion_alpha", "motion_response_alpha", 1.0),
            ("motion_bias", "motion_bias", 0.0),
        ]:
            out = getattr(self, name)
            lo, hi = r[key]
            out[env_ids] = center + strength * (torch.rand_like(out[env_ids]) * (hi - lo) + lo - center)
        # Gain ranges from source recipe, centered on Panda's stable actuator gains.
        if not hasattr(self, "nominal_stiffness"):
            self.nominal_stiffness = self.robot.data.joint_stiffness.clone()
            self.nominal_damping = self.robot.data.joint_damping.clone()
        for kind in ["stiffness", "damping"]:
            lo, hi = r["physics_joint_" + kind + "_scale"]
            values = getattr(self, "nominal_" + kind)[env_ids] * (
                1 + strength * (torch.rand((n, 1), device=self.device) * (hi - lo) + lo - 1)
            )
            getattr(self.robot, "write_joint_" + kind + "_to_sim")(values, env_ids=env_ids)
        self.extras["log"] = {
            "curriculum/fraction": pn.new_tensor(c.fraction),
            "reset/rotation_deg": rn[env_ids].mean() * 180 / torch.pi,
            "reset/position_mm": pn[env_ids].mean() * 1000,
            "reset/perturbed_fraction": rotated.float().mean(),
        }

    def _pose_actions(self, actions):
        r = self.recipe
        self.motion_history[1:] = self.motion_history[:-1].clone()
        self.motion_history[0] = actions[:, :6].clamp(-1, 1)
        delayed = self.motion_history[self.motion_delay, torch.arange(self.num_envs, device=self.device)]
        self.motion_filter.mul_(1 - self.motion_alpha).add_(
            (delayed * self.motion_scale + self.motion_bias) * self.motion_alpha
        )
        motion = self.previous_actions + (self.motion_filter - self.previous_actions).clamp(
            -r["action_delta_limit"], r["action_delta_limit"]
        )
        self.actions = torch.cat((motion, actions[:, 6:7].clamp(0, 1)), -1)
        twist = (self.tcp_jacobian() @ self.robot.data.joint_vel[:, self.arm_ids, None]).squeeze(-1)
        stable = (twist[:, :3].norm(dim=-1) <= r["completion_max_linear_speed_m_s"]) & (
            twist[:, 3:].norm(dim=-1) <= r["completion_max_angular_speed_rad_s"]
        )
        eligible = (actions[:, 6] >= r["completion_probability_threshold"]) & stable
        self.completion_streak = torch.where(
            eligible, self.completion_streak + 1, torch.zeros_like(self.completion_streak)
        )
        self.completion_declaration = self.completion_streak >= r["completion_required_consecutive_steps"]

    def _pose_live(self, live):
        lo = self.camera_profile["depth_min_m"]
        hi = self.camera_profile["depth_max_m"]
        rgb, depth = self.live_randomizer.apply(live[..., :3], live[..., 3:] * (hi - lo) + lo)
        current = torch.cat((rgb, ((depth - lo) / (hi - lo)).clamp(0, 1)), -1)
        empty = self.live_history_empty
        self.live_history[:, empty] = current[empty]
        self.live_history_empty[:] = False
        self.live_history[1:] = self.live_history[:-1].clone()
        self.live_history[0] = current
        return self.live_history[self.observation_delay, torch.arange(self.num_envs, device=self.device)]

    def _pose_reward(self, pn, rn):
        r = self.recipe
        if self.cfg.symmetry_training and self.cfg.symmetry_objective == "orbit_potential_pose_set_v2":
            current_potential = self.symmetry_potential()
            reward = current_potential - self.previous_potential
            self.previous_potential.copy_(current_potential)
        else:
            reward = r["position_progress_weight"] * (self.previous_position_error - pn) + r[
                "rotation_progress_weight"
            ] * (self.previous_rotation_error - rn)
            for current, previous, prefix in [
                (pn, self.previous_position_error, "position"),
                (rn, self.previous_rotation_error, "rotation"),
            ]:
                scale = r[prefix + "_precision_scale_" + ("m" if prefix == "position" else "rad")]
                reward += r[prefix + "_precision_weight"] * (torch.exp(-current / scale) - torch.exp(-previous / scale))
        declaration = self.completion_declaration
        ready = self._labels(pn, rn).ready
        terminal = completion_terminal_reward(
            declaration,
            ready,
            correct_reward=r["completion_correct_reward"],
            premature_penalty=r["completion_premature_penalty"],
        )
        terminal = torch.where(
            (self.reset_mode == 2) & (terminal > 0), terminal * r["completion_positive_terminal_reward_scale"], terminal
        )
        timeout = self.episode_length_buf >= self.pose_timeout_steps - 1
        reward += (
            terminal
            - r["unsafe_collision_penalty"] * self.terminal_collision.float()
            - r["divergence_penalty"] * self.terminal_divergence.float()
            - r["timeout_penalty"]
            * (timeout & ~declaration & ~self.terminal_collision & ~self.terminal_divergence).float()
        )
        risk = (
            (
                (self.contact_force() - r["collision_risk_force_threshold_n"])
                / (r["unsafe_contact_force_threshold_n"] - r["collision_risk_force_threshold_n"])
            )
            .clamp(0, 1)
            .square()
        )
        reward -= self.time_ratio * (
            r["step_penalty"]
            + r["action_penalty_weight"] * self.actions[:, :6].square().sum(-1)
            + r["collision_risk_penalty_weight"] * risk
        )
        terminal_mask = declaration | timeout | self.terminal_collision | self.terminal_divergence
        failure = torch.maximum(
            (~self.terminal_success).float(),
            torch.maximum(1.5 * self.terminal_collision.float(), 1.25 * self.terminal_divergence.float()),
        )
        self.target_failure_scores = update_failure_scores(
            self.target_failure_scores,
            target_indices=self.target_index,
            terminal_mask=terminal_mask,
            failure_values=failure,
            decay=r["failure_score_decay"],
        )
        self.previous_position_error.copy_(pn)
        self.previous_rotation_error.copy_(rn)
        return reward
