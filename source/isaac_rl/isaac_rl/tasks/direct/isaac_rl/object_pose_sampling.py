"""Collision-bounded planar object-pose perturbations for visual servoing."""

from __future__ import annotations

import math

import torch


def yaw_offset_profile(
    progress: torch.Tensor,
    *,
    far_yaw_rad: float,
    near_yaw_rad: float,
    exponent: float,
) -> torch.Tensor:
    """Return the maximum absolute object-yaw error along the approach path."""

    if far_yaw_rad < 0.0 or near_yaw_rad < 0.0:
        raise ValueError("Object-yaw reset limits must be non-negative.")
    if near_yaw_rad > far_yaw_rad:
        raise ValueError("The near object-yaw limit cannot exceed the far limit.")
    if far_yaw_rad > math.pi:
        raise ValueError("The object-yaw limit cannot exceed pi radians.")
    if exponent <= 0.0:
        raise ValueError("The object-yaw exponent must be positive.")
    return near_yaw_rad + (far_yaw_rad - near_yaw_rad) * torch.pow(
        1.0 - progress.clamp(0.0, 1.0),
        exponent,
    )


def sample_collision_safe_yaw_offsets_from_profile(
    requested_profile_rad: torch.Tensor,
    collision_clearance_m: torch.Tensor,
    translation_magnitude_m: torch.Tensor,
    object_xy_radius_m: torch.Tensor,
    zero_offset_mask: torch.Tensor,
    *,
    minimum_collision_clearance_m: float,
    clearance_guard_m: float,
    magnitude_unit_samples: torch.Tensor | None = None,
    sign_unit_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample signed yaw while conservatively preserving validated clearance.

    A point at planar radius ``r`` moves by at most
    ``2*r*sin(abs(yaw)/2)`` under a yaw rotation around the object root.  The
    triangle inequality combines that displacement with the already sampled
    object translation.  The resulting angular cap therefore cannot consume
    the minimum robot/object clearance validated for the nominal reset state.

    Returns ``(signed_yaw, requested_abs_yaw, safe_abs_yaw_cap)``.
    """

    tensors = (
        collision_clearance_m,
        translation_magnitude_m,
        object_xy_radius_m,
        zero_offset_mask,
    )
    if any(value.shape != requested_profile_rad.shape for value in tensors):
        raise ValueError("Yaw profiles, clearances, translations, radii, and zero masks must match.")
    if torch.any(requested_profile_rad < 0.0) or torch.any(requested_profile_rad > math.pi):
        raise ValueError("Requested absolute yaw must lie in [0, pi].")
    if torch.any(translation_magnitude_m < 0.0):
        raise ValueError("Translation magnitudes must be non-negative.")
    if torch.any(object_xy_radius_m <= 0.0):
        raise ValueError("Object XY radii must be positive.")
    if minimum_collision_clearance_m < 0.0 or clearance_guard_m < 0.0:
        raise ValueError("Collision clearance and guard must be non-negative.")
    if magnitude_unit_samples is None:
        magnitude_unit_samples = torch.rand_like(requested_profile_rad)
    if sign_unit_samples is None:
        sign_unit_samples = torch.rand_like(requested_profile_rad)
    if (
        magnitude_unit_samples.shape != requested_profile_rad.shape
        or sign_unit_samples.shape != requested_profile_rad.shape
    ):
        raise ValueError("Object-yaw random samples must match the profile shape.")

    requested_abs_yaw = requested_profile_rad * magnitude_unit_samples.clamp(0.0, 1.0)
    remaining_displacement_m = (
        collision_clearance_m
        - float(minimum_collision_clearance_m)
        - float(clearance_guard_m)
        - translation_magnitude_m
    ).clamp_min(0.0)
    sine_half_cap = (remaining_displacement_m / (2.0 * object_xy_radius_m)).clamp(0.0, 1.0)
    safe_abs_yaw_cap = 2.0 * torch.asin(sine_half_cap)
    abs_yaw = torch.minimum(requested_abs_yaw, safe_abs_yaw_cap)
    abs_yaw = torch.where(zero_offset_mask, torch.zeros_like(abs_yaw), abs_yaw)
    requested_abs_yaw = torch.where(
        zero_offset_mask,
        torch.zeros_like(requested_abs_yaw),
        requested_abs_yaw,
    )
    sign = torch.where(sign_unit_samples < 0.5, -torch.ones_like(abs_yaw), torch.ones_like(abs_yaw))
    return sign * abs_yaw, requested_abs_yaw, safe_abs_yaw_cap


def _quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def apply_planar_object_pose_delta(
    object_position_w: torch.Tensor,
    object_quaternion_wxyz: torch.Tensor,
    target_position_w: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
    translation_offset_w: torch.Tensor,
    yaw_offset_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a stable object and part-relative target on the support plane.

    Only world-XY translation and world-Z yaw are accepted. The validated
    resting height and roll/pitch therefore remain unchanged; no unstable or
    floor-penetrating physical pose can be synthesized by this helper.
    """

    count = yaw_offset_rad.shape[0]
    if object_position_w.shape != (count, 3) or target_position_w.shape != (count, 3):
        raise ValueError("Object and target positions must have shape (N, 3).")
    if object_quaternion_wxyz.shape != (count, 4) or target_quaternion_wxyz.shape != (count, 4):
        raise ValueError("Object and target quaternions must have shape (N, 4).")
    if translation_offset_w.shape != (count, 3):
        raise ValueError("Object translation offsets must have shape (N, 3).")
    if torch.any(translation_offset_w[:, 2].abs() > 1.0e-9):
        raise ValueError("Stable planar object-pose perturbations cannot change world Z.")

    cosine = torch.cos(yaw_offset_rad)
    sine = torch.sin(yaw_offset_rad)
    relative_target = target_position_w - object_position_w
    rotated_relative = relative_target.clone()
    rotated_relative[:, 0] = cosine * relative_target[:, 0] - sine * relative_target[:, 1]
    rotated_relative[:, 1] = sine * relative_target[:, 0] + cosine * relative_target[:, 1]
    moved_object_position = object_position_w + translation_offset_w
    moved_target_position = moved_object_position + rotated_relative

    half_yaw = 0.5 * yaw_offset_rad
    yaw_quaternion = torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ),
        dim=-1,
    )
    moved_object_quaternion = _quat_mul_wxyz(yaw_quaternion, object_quaternion_wxyz)
    moved_target_quaternion = _quat_mul_wxyz(yaw_quaternion, target_quaternion_wxyz)
    return (
        moved_object_position,
        moved_object_quaternion,
        moved_target_position,
        moved_target_quaternion,
    )


__all__ = [
    "apply_planar_object_pose_delta",
    "sample_collision_safe_yaw_offsets_from_profile",
    "yaw_offset_profile",
]
