"""Collision-safe translational reset perturbations for visual servoing."""

from __future__ import annotations

import math

import torch


def position_offset_profile(
    progress: torch.Tensor,
    *,
    far_offset_m: float,
    near_offset_m: float,
    exponent: float,
) -> torch.Tensor:
    """Return the requested off-path position error at each path progress."""

    if far_offset_m < 0.0 or near_offset_m < 0.0:
        raise ValueError("Position-reset offsets must be non-negative.")
    if near_offset_m > far_offset_m:
        raise ValueError("The near position-reset offset cannot exceed the far offset.")
    if exponent <= 0.0:
        raise ValueError("The position-reset exponent must be positive.")
    return near_offset_m + (far_offset_m - near_offset_m) * torch.pow(1.0 - progress.clamp(0.0, 1.0), exponent)


def sample_collision_safe_xy_offsets(
    progress: torch.Tensor,
    collision_clearance_m: torch.Tensor,
    zero_offset_mask: torch.Tensor,
    *,
    minimum_collision_clearance_m: float,
    clearance_guard_m: float,
    far_offset_m: float,
    near_offset_m: float,
    exponent: float,
    fraction_min: float,
    fraction_max: float,
    magnitude_unit_samples: torch.Tensor | None = None,
    direction_unit_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample horizontal object/goal offsets bounded by validated clearance.

    Translating the object by ``delta`` is equivalent, for object/gripper
    separation, to translating the validated gripper pose by ``-delta``.
    Euclidean separation can decrease by at most ``||delta||``.  Capping the
    requested offset by the stored clearance margin therefore preserves the
    authored minimum clearance for every direction without runtime IK or an
    optimistic collision approximation.

    Returns ``(offset_xyz, requested_magnitude, safe_cap)``. Entries selected
    by ``zero_offset_mask`` always receive a zero offset.
    """

    if progress.shape != collision_clearance_m.shape or progress.shape != zero_offset_mask.shape:
        raise ValueError("Progress, clearance, and zero-mask tensors must have matching shapes.")
    if minimum_collision_clearance_m < 0.0 or clearance_guard_m < 0.0:
        raise ValueError("Collision clearance and guard must be non-negative.")
    if not 0.0 <= fraction_min <= fraction_max <= 1.0:
        raise ValueError("Position-reset fractions must satisfy 0 <= min <= max <= 1.")

    requested_profile = position_offset_profile(
        progress,
        far_offset_m=far_offset_m,
        near_offset_m=near_offset_m,
        exponent=exponent,
    )
    if magnitude_unit_samples is None:
        magnitude_unit_samples = torch.rand_like(progress)
    if direction_unit_samples is None:
        direction_unit_samples = torch.rand_like(progress)
    if magnitude_unit_samples.shape != progress.shape or direction_unit_samples.shape != progress.shape:
        raise ValueError("Position-reset random samples must match the progress shape.")
    fraction = fraction_min + (fraction_max - fraction_min) * magnitude_unit_samples.clamp(0.0, 1.0)
    requested_magnitude = requested_profile * fraction
    safe_cap = (collision_clearance_m - float(minimum_collision_clearance_m) - float(clearance_guard_m)).clamp_min(0.0)
    magnitude = torch.minimum(requested_magnitude, safe_cap)
    magnitude = torch.where(zero_offset_mask, torch.zeros_like(magnitude), magnitude)
    requested_magnitude = torch.where(zero_offset_mask, torch.zeros_like(requested_magnitude), requested_magnitude)
    angle = direction_unit_samples.clamp(0.0, 1.0) * (2.0 * math.pi)
    offsets = torch.stack(
        (
            magnitude * torch.cos(angle),
            magnitude * torch.sin(angle),
            torch.zeros_like(magnitude),
        ),
        dim=-1,
    )
    return offsets, requested_magnitude, safe_cap


def sample_collision_safe_xy_offsets_from_profile(
    requested_profile_m: torch.Tensor,
    collision_clearance_m: torch.Tensor,
    zero_offset_mask: torch.Tensor,
    *,
    minimum_collision_clearance_m: float,
    clearance_guard_m: float,
    magnitude_unit_samples: torch.Tensor | None = None,
    direction_unit_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample offsets from a per-environment maximum requested magnitude."""

    if requested_profile_m.shape != collision_clearance_m.shape or requested_profile_m.shape != zero_offset_mask.shape:
        raise ValueError("Position profiles, clearances, and zero masks must match.")
    if torch.any(requested_profile_m < 0.0):
        raise ValueError("Requested position profiles must be non-negative.")
    if minimum_collision_clearance_m < 0.0 or clearance_guard_m < 0.0:
        raise ValueError("Collision clearance and guard must be non-negative.")
    if magnitude_unit_samples is None:
        magnitude_unit_samples = torch.rand_like(requested_profile_m)
    if direction_unit_samples is None:
        direction_unit_samples = torch.rand_like(requested_profile_m)
    if (
        magnitude_unit_samples.shape != requested_profile_m.shape
        or direction_unit_samples.shape != requested_profile_m.shape
    ):
        raise ValueError("Position-reset random samples must match the profile shape.")

    requested_magnitude = requested_profile_m * magnitude_unit_samples.clamp(0.0, 1.0)
    safe_cap = (collision_clearance_m - float(minimum_collision_clearance_m) - float(clearance_guard_m)).clamp_min(0.0)
    magnitude = torch.minimum(requested_magnitude, safe_cap)
    magnitude = torch.where(zero_offset_mask, torch.zeros_like(magnitude), magnitude)
    requested_magnitude = torch.where(
        zero_offset_mask,
        torch.zeros_like(requested_magnitude),
        requested_magnitude,
    )
    angle = direction_unit_samples.clamp(0.0, 1.0) * (2.0 * math.pi)
    offsets = torch.stack(
        (
            magnitude * torch.cos(angle),
            magnitude * torch.sin(angle),
            torch.zeros_like(magnitude),
        ),
        dim=-1,
    )
    return offsets, requested_magnitude, safe_cap


__all__ = [
    "position_offset_profile",
    "sample_collision_safe_xy_offsets",
    "sample_collision_safe_xy_offsets_from_profile",
]
