"""Frame transforms shared by visual-servo observations and control."""

from __future__ import annotations

import torch


def world_pose_error_to_camera(
    position_error_world: torch.Tensor,
    rotation_error_world: torch.Tensor,
    rotation_world_from_camera: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express world-frame translation and rotation-vector errors in the camera frame."""

    rotation_camera_from_world = rotation_world_from_camera.transpose(-1, -2)
    position_error_camera = torch.matmul(
        rotation_camera_from_world, position_error_world.unsqueeze(-1)
    ).squeeze(-1)
    rotation_error_camera = torch.matmul(
        rotation_camera_from_world, rotation_error_world.unsqueeze(-1)
    ).squeeze(-1)
    return position_error_camera, rotation_error_camera


def interpolate_joint_trajectory(
    joint_trajectory: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    """Interpolate a batched set of normalized progress values along a joint path."""

    if joint_trajectory.ndim not in (2, 3):
        raise ValueError("joint_trajectory must have shape (N, J) or (B, N, J).")
    if progress.ndim != 1:
        raise ValueError("progress must be a one-dimensional tensor.")
    waypoint_count = joint_trajectory.shape[-2]
    if waypoint_count < 2:
        raise ValueError("joint_trajectory must contain at least two waypoints.")
    if joint_trajectory.ndim == 3 and joint_trajectory.shape[0] != progress.shape[0]:
        raise ValueError(
            "Batched joint trajectories and progress must have the same batch size."
        )
    scaled = progress.clamp(0.0, 1.0) * (waypoint_count - 1)
    lower = scaled.floor().to(dtype=torch.long)
    upper = (lower + 1).clamp(max=waypoint_count - 1)
    blend = (scaled - lower).unsqueeze(-1)
    if joint_trajectory.ndim == 2:
        lower_values = joint_trajectory[lower]
        upper_values = joint_trajectory[upper]
    else:
        batch = torch.arange(progress.shape[0], device=progress.device)
        lower_values = joint_trajectory[batch, lower]
        upper_values = joint_trajectory[batch, upper]
    return torch.lerp(lower_values, upper_values, blend)


def balanced_target_progress(
    *,
    target_count: int,
    sample_count: int,
    target_cursor: int,
    target_sample_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Assign balanced targets and low-discrepancy path progress values.

    The persistent per-target counters prevent frequent/easy resets from
    repeatedly receiving the same approach region. Returned progress is in
    [0, 1); callers apply their configured curriculum interval afterwards.
    """

    if target_count <= 0 or sample_count < 0:
        raise ValueError("target_count must be positive and sample_count non-negative.")
    if target_sample_counts.shape != (target_count,):
        raise ValueError(
            f"target_sample_counts must have shape ({target_count},), got "
            f"{tuple(target_sample_counts.shape)}."
        )
    if sample_count == 0:
        empty_targets = torch.empty(0, dtype=torch.long, device=target_sample_counts.device)
        empty_progress = torch.empty(0, dtype=torch.float32, device=target_sample_counts.device)
        return empty_targets, empty_progress, target_cursor % target_count

    device = target_sample_counts.device
    targets = (
        torch.arange(sample_count, dtype=torch.long, device=device) + int(target_cursor)
    ) % target_count
    occurrence = torch.zeros(sample_count, dtype=torch.long, device=device)
    for target in range(target_count):
        positions = torch.nonzero(targets == target, as_tuple=False).flatten()
        if positions.numel():
            occurrence[positions] = torch.arange(
                positions.numel(), dtype=torch.long, device=device
            )
    sequence_index = target_sample_counts[targets] + occurrence
    golden_conjugate = 0.6180339887498949
    # Hash target identity into phase rather than using its catalog index
    # directly. Catalog entries are grouped by object orientation; a linear
    # phase would therefore correlate one orientation with near resets and
    # another with far resets during the first synchronized rollout.
    target_phase = torch.remainder(targets.to(torch.float32) * 0.754877666, 1.0)
    progress = torch.remainder(
        (sequence_index.to(torch.float32) + 0.5) * golden_conjugate + target_phase,
        1.0,
    )
    target_sample_counts.add_(
        torch.bincount(targets, minlength=target_count).to(target_sample_counts.dtype)
    )
    return targets, progress, (int(target_cursor) + sample_count) % target_count


def balanced_group_target_progress(
    *,
    target_group_indices: torch.Tensor,
    sample_count: int,
    group_cursor: int,
    group_target_cursors: torch.Tensor,
    target_sample_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sample groups uniformly, then targets uniformly within each group.

    This prevents a part with hundreds of valid grasps from dominating a part
    with only tens of valid grasps. The target counters retain the same
    low-discrepancy approach-progress schedule as ``balanced_target_progress``.
    ``group_target_cursors`` and ``target_sample_counts`` are updated in place.
    """

    if target_group_indices.ndim != 1 or target_group_indices.numel() == 0:
        raise ValueError("target_group_indices must be a non-empty vector.")
    target_count = int(target_group_indices.numel())
    if target_sample_counts.shape != (target_count,):
        raise ValueError(
            f"target_sample_counts must have shape ({target_count},), got "
            f"{tuple(target_sample_counts.shape)}."
        )
    group_values = torch.unique(target_group_indices, sorted=True)
    group_count = int(group_values.numel())
    if group_target_cursors.shape != (group_count,):
        raise ValueError(
            f"group_target_cursors must have shape ({group_count},), got "
            f"{tuple(group_target_cursors.shape)}."
        )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    if sample_count == 0:
        empty_targets = torch.empty(
            0, dtype=torch.long, device=target_group_indices.device
        )
        empty_progress = torch.empty(
            0, dtype=torch.float32, device=target_group_indices.device
        )
        return empty_targets, empty_progress, group_cursor % group_count

    device = target_group_indices.device
    group_ordinals = (
        torch.arange(sample_count, dtype=torch.long, device=device) + int(group_cursor)
    ) % group_count
    sampled_groups = group_values[group_ordinals]
    targets = torch.empty(sample_count, dtype=torch.long, device=device)
    for ordinal, group_value in enumerate(group_values):
        positions = torch.nonzero(sampled_groups == group_value, as_tuple=False).flatten()
        if not positions.numel():
            continue
        candidates = torch.nonzero(
            target_group_indices == group_value, as_tuple=False
        ).flatten()
        local_indices = (
            torch.arange(positions.numel(), dtype=torch.long, device=device)
            + group_target_cursors[ordinal]
        ) % candidates.numel()
        targets[positions] = candidates[local_indices]
        group_target_cursors[ordinal] = (
            group_target_cursors[ordinal] + positions.numel()
        ) % candidates.numel()

    occurrence = torch.zeros(sample_count, dtype=torch.long, device=device)
    for target in torch.unique(targets):
        positions = torch.nonzero(targets == target, as_tuple=False).flatten()
        occurrence[positions] = torch.arange(
            positions.numel(), dtype=torch.long, device=device
        )
    sequence_index = target_sample_counts[targets] + occurrence
    target_phase = torch.remainder(targets.to(torch.float32) * 0.754877666, 1.0)
    progress = torch.remainder(
        (sequence_index.to(torch.float32) + 0.5) * 0.6180339887498949
        + target_phase,
        1.0,
    )
    target_sample_counts.add_(
        torch.bincount(targets, minlength=target_count).to(target_sample_counts.dtype)
    )
    return targets, progress, (int(group_cursor) + sample_count) % group_count


def path_conditioned_noise_scale(
    progress: torch.Tensor,
    *,
    far_scale: float,
    near_scale: float,
    exponent: float,
) -> torch.Tensor:
    """Return perturbation scales that monotonically shrink toward the grasp."""

    if far_scale < near_scale or near_scale < 0.0:
        raise ValueError("Noise scales must satisfy far_scale >= near_scale >= 0.")
    if exponent <= 0.0:
        raise ValueError("Noise exponent must be positive.")
    remaining = 1.0 - progress.clamp(0.0, 1.0)
    return near_scale + (far_scale - near_scale) * remaining.pow(exponent)


def approach_progress_bucket_masks(
    progress: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split normalized pregrasp-to-grasp progress into far, middle, and close thirds."""

    if progress.ndim != 1:
        raise ValueError("progress must be a one-dimensional tensor.")
    return {
        "far": progress < 1.0 / 3.0,
        "mid": (progress >= 1.0 / 3.0) & (progress < 2.0 / 3.0),
        "close": progress >= 2.0 / 3.0,
    }
