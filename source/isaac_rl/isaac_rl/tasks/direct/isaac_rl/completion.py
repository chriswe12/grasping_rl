"""Completion labels, decisions, and rewards for autonomous visual stopping."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompletionMasks:
    """Privileged training/evaluation masks; never actor inputs at deployment."""

    ready: torch.Tensor
    definitely_not_ready: torch.Tensor
    supervised: torch.Tensor


def completion_masks(
    position_error_m: torch.Tensor,
    rotation_error_rad: torch.Tensor,
    *,
    ready_position_m: float,
    ready_rotation_rad: float,
    negative_position_m: float,
    negative_rotation_rad: float,
    collision_free: torch.Tensor | None = None,
) -> CompletionMasks:
    """Build strict-positive, clear-negative, and ambiguity-ignore masks."""

    if position_error_m.shape != rotation_error_rad.shape:
        raise ValueError("Position and rotation errors must have the same shape.")
    if not 0.0 < ready_position_m < negative_position_m:
        raise ValueError("Completion position thresholds must satisfy 0 < ready < negative.")
    if not 0.0 < ready_rotation_rad < negative_rotation_rad:
        raise ValueError("Completion rotation thresholds must satisfy 0 < ready < negative.")
    if collision_free is None:
        collision_free = torch.ones_like(position_error_m, dtype=torch.bool)
    elif collision_free.shape != position_error_m.shape:
        raise ValueError("collision_free must have the same shape as the error tensors.")
    collision_free = collision_free.bool()
    ready = (position_error_m <= ready_position_m) & (rotation_error_rad <= ready_rotation_rad) & collision_free
    definitely_not_ready = (
        (position_error_m >= negative_position_m) | (rotation_error_rad >= negative_rotation_rad) | ~collision_free
    )
    return CompletionMasks(
        ready=ready,
        definitely_not_ready=definitely_not_ready,
        supervised=ready | definitely_not_ready,
    )


def completion_declared(stop_action: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Interpret the final normalized action as a Bernoulli stop decision."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("Stop-action threshold must lie strictly between zero and one.")
    return stop_action >= threshold


def completion_terminal_reward(
    declared: torch.Tensor,
    ready: torch.Tensor,
    *,
    correct_reward: float,
    premature_penalty: float,
) -> torch.Tensor:
    """Return the one-time reward attached to an explicit stop declaration."""

    if declared.shape != ready.shape:
        raise ValueError("declared and ready must have the same shape.")
    reward = declared.new_zeros(declared.shape, dtype=torch.float32)
    reward += float(correct_reward) * (declared & ready).float()
    reward -= float(premature_penalty) * (declared & ~ready).float()
    return reward


def update_completion_streak(
    previous_streak: torch.Tensor,
    completion_probability: torch.Tensor,
    *,
    probability_threshold: float,
) -> torch.Tensor:
    """Count consecutive high-confidence frames for deployment debouncing."""

    if previous_streak.shape != completion_probability.shape:
        raise ValueError("Streak and completion probability must have the same shape.")
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("Probability threshold must lie strictly between zero and one.")
    high_confidence = completion_probability >= probability_threshold
    return torch.where(high_confidence, previous_streak + 1, torch.zeros_like(previous_streak))


__all__ = [
    "CompletionMasks",
    "completion_declared",
    "completion_masks",
    "completion_terminal_reward",
    "update_completion_streak",
]
