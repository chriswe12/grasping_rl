"""Pure helpers for visual-servo reset curriculum and hard-target replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch

RESET_MODE_PATH = 0
RESET_MODE_NO_NOISE = 1
RESET_MODE_READY = 2
RESET_MODE_BOUNDARY = 3


@dataclass(frozen=True)
class CurriculumState:
    """Resolved curriculum controls at one simulator step."""

    fraction: float
    progress_min: float
    perturbation_scale: float
    visual_randomization_strength: float
    failure_replay_fraction: float


def curriculum_state(
    step: int,
    *,
    enabled: bool,
    warmup_steps: int,
    full_steps: int,
    initial_progress_min: float,
    final_progress_min: float,
    final_failure_replay_fraction: float,
) -> CurriculumState:
    """Expand close/easy resets into the full task with a linear ramp."""

    if warmup_steps < 0 or full_steps <= warmup_steps:
        raise ValueError("Curriculum steps must satisfy 0 <= warmup < full.")
    if not 0.0 <= final_progress_min <= initial_progress_min <= 1.0:
        raise ValueError("Curriculum progress bounds must lie in [0, 1] and decrease.")
    if not 0.0 <= final_failure_replay_fraction <= 1.0:
        raise ValueError("Failure replay fraction must lie in [0, 1].")
    if not enabled:
        return CurriculumState(
            fraction=1.0,
            progress_min=final_progress_min,
            perturbation_scale=1.0,
            visual_randomization_strength=1.0,
            failure_replay_fraction=final_failure_replay_fraction,
        )
    fraction = min(1.0, max(0.0, (int(step) - warmup_steps) / (full_steps - warmup_steps)))
    return CurriculumState(
        fraction=fraction,
        progress_min=initial_progress_min + fraction * (final_progress_min - initial_progress_min),
        perturbation_scale=fraction,
        visual_randomization_strength=fraction,
        failure_replay_fraction=fraction * final_failure_replay_fraction,
    )


def sample_reset_modes(
    unit_samples: torch.Tensor,
    *,
    no_noise_fraction: float,
    ready_fraction: float,
    boundary_fraction: float,
) -> torch.Tensor:
    """Assign path, unperturbed, completion-positive, and boundary resets."""

    if unit_samples.ndim != 1:
        raise ValueError("Reset-mode samples must be one-dimensional.")
    fractions = (no_noise_fraction, ready_fraction, boundary_fraction)
    if any(value < 0.0 for value in fractions) or sum(fractions) > 1.0:
        raise ValueError("Reset-mode fractions must be non-negative and sum to at most one.")
    values = unit_samples.clamp(0.0, 1.0)
    modes = torch.full_like(values, RESET_MODE_PATH, dtype=torch.long)
    no_noise_end = no_noise_fraction
    ready_end = no_noise_end + ready_fraction
    boundary_end = ready_end + boundary_fraction
    modes[values < no_noise_end] = RESET_MODE_NO_NOISE
    modes[(values >= no_noise_end) & (values < ready_end)] = RESET_MODE_READY
    modes[(values >= ready_end) & (values < boundary_end)] = RESET_MODE_BOUNDARY
    return modes


def reset_timeout_seconds(
    progress: torch.Tensor,
    reset_modes: torch.Tensor,
    *,
    far_seconds: float,
    close_seconds: float,
    ready_seconds: float,
    boundary_seconds: float,
    exponent: float,
) -> torch.Tensor:
    """Return a per-episode timeout matched to reset distance and purpose."""

    if progress.shape != reset_modes.shape:
        raise ValueError("Progress and reset modes must have matching shapes.")
    if min(far_seconds, close_seconds, ready_seconds, boundary_seconds) <= 0.0:
        raise ValueError("Every reset timeout must be positive.")
    if far_seconds < close_seconds:
        raise ValueError("Far timeout cannot be shorter than close timeout.")
    if exponent <= 0.0:
        raise ValueError("Timeout exponent must be positive.")
    remaining = (1.0 - progress.clamp(0.0, 1.0)).pow(exponent)
    result = close_seconds + (far_seconds - close_seconds) * remaining
    result = torch.where(
        reset_modes == RESET_MODE_READY,
        result.new_full((), ready_seconds),
        result,
    )
    return torch.where(
        reset_modes == RESET_MODE_BOUNDARY,
        result.new_full((), boundary_seconds),
        result,
    )


def apply_failure_replay(
    targets: torch.Tensor,
    *,
    target_group_indices: torch.Tensor,
    failure_scores: torch.Tensor,
    replay_fraction: float,
    score_floor: float,
    score_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of balanced targets with hard targets from the same group."""

    if targets.ndim != 1:
        raise ValueError("Targets must be one-dimensional.")
    target_count = int(target_group_indices.numel())
    if failure_scores.shape != (target_count,):
        raise ValueError("Failure scores must contain one value per target.")
    if not 0.0 <= replay_fraction <= 1.0:
        raise ValueError("Replay fraction must lie in [0, 1].")
    if score_floor <= 0.0 or score_power <= 0.0:
        raise ValueError("Replay score floor and power must be positive.")
    replay_count = min(len(targets), int(round(len(targets) * replay_fraction)))
    replay_mask = torch.zeros_like(targets, dtype=torch.bool)
    if replay_count == 0:
        return targets.clone(), replay_mask

    replay_positions = torch.randperm(len(targets), device=targets.device)[:replay_count]
    replay_mask[replay_positions] = True
    result = targets.clone()
    slot_groups = target_group_indices[targets[replay_positions]]
    for group in torch.unique(slot_groups):
        group_positions = replay_positions[slot_groups == group]
        candidates = torch.nonzero(target_group_indices == group, as_tuple=False).flatten()
        weights = (failure_scores[candidates].clamp_min(0.0) + score_floor).pow(score_power)
        chosen = torch.multinomial(weights, len(group_positions), replacement=True)
        result[group_positions] = candidates[chosen]
    return result, replay_mask


def update_failure_scores(
    failure_scores: torch.Tensor,
    *,
    target_indices: torch.Tensor,
    terminal_mask: torch.Tensor,
    failure_values: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """Apply one per-target EMA update from newly terminated episodes."""

    if target_indices.shape != terminal_mask.shape or target_indices.shape != failure_values.shape:
        raise ValueError("Failure update tensors must have matching shapes.")
    if not 0.0 <= decay < 1.0:
        raise ValueError("Failure-score decay must lie in [0, 1).")
    result = failure_scores.clone()
    if not terminal_mask.any():
        return result
    terminals = target_indices[terminal_mask]
    values = failure_values[terminal_mask]
    sums = torch.zeros_like(result)
    counts = torch.zeros_like(result)
    sums.scatter_add_(0, terminals, values)
    counts.scatter_add_(0, terminals, torch.ones_like(values))
    updated = counts > 0
    means = sums / counts.clamp_min(1.0)
    result[updated] = decay * result[updated] + (1.0 - decay) * means[updated]
    return result


__all__ = [
    "CurriculumState",
    "RESET_MODE_BOUNDARY",
    "RESET_MODE_NO_NOISE",
    "RESET_MODE_PATH",
    "RESET_MODE_READY",
    "apply_failure_replay",
    "curriculum_state",
    "reset_timeout_seconds",
    "sample_reset_modes",
    "update_failure_scores",
]
