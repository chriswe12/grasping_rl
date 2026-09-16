"""PPO integration for the Gaussian-motion plus Bernoulli-stop policy."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.players import PpoPlayerContinuous
from torch import nn

from .completion_model import bernoulli_kl_from_probabilities


class _ReusableGradientAllReduce:
    """All-reduce model gradients through one persistent flat buffer."""

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None
        self._signature: tuple[tuple[int, int, torch.dtype, torch.device], ...] = ()

    @property
    def size_mib(self) -> float:
        if self.buffer is None:
            return 0.0
        return self.buffer.numel() * self.buffer.element_size() / (1024.0 * 1024.0)

    def reduce(self, parameters, *, world_size: int) -> None:
        parameters_with_grad = tuple(parameter for parameter in parameters if parameter.grad is not None)
        if not parameters_with_grad:
            return
        signature = tuple(
            (id(parameter), parameter.numel(), parameter.grad.dtype, parameter.grad.device)
            for parameter in parameters_with_grad
        )
        first_gradient = parameters_with_grad[0].grad
        if self.buffer is None or signature != self._signature:
            self.buffer = torch.empty(
                sum(parameter.numel() for parameter in parameters_with_grad),
                dtype=first_gradient.dtype,
                device=first_gradient.device,
            )
            self._signature = signature

        with torch.no_grad():
            offset = 0
            for parameter in parameters_with_grad:
                count = parameter.numel()
                self.buffer[offset : offset + count].copy_(parameter.grad.reshape(-1))
                offset += count
            dist.all_reduce(self.buffer, op=dist.ReduceOp.SUM)
            self.buffer.mul_(1.0 / float(world_size))
            offset = 0
            for parameter in parameters_with_grad:
                count = parameter.numel()
                parameter.grad.copy_(self.buffer[offset : offset + count].view_as(parameter.grad))
                offset += count


def _central_value_calc_gradients_with_reusable_buffer(self, batch):
    """Pinned RL-Games central-value update with persistent gradient storage."""

    from rl_games.common import common_losses

    obs_batch = self._preproc_obs(batch["obs"])
    value_preds_batch = batch["old_values"]
    returns_batch = batch["returns"]
    actions_batch = batch["actions"]
    dones_batch = batch["dones"]
    rnn_masks_batch = batch.get("rnn_masks")
    batch_dict = {
        "obs": obs_batch,
        "actions": actions_batch,
        "seq_length": self.seq_length,
        "dones": dones_batch,
    }
    if self.is_rnn:
        batch_dict["rnn_states"] = batch["rnn_states"]

    result = self.model(batch_dict)
    values = result["values"]
    loss = common_losses.critic_loss(
        self.model,
        value_preds_batch,
        values,
        self.e_clip,
        returns_batch,
        self.clip_value,
    )
    losses, _ = torch_ext.apply_masks([loss], rnn_masks_batch)
    loss = losses[0]
    if self.multi_gpu:
        self.optimizer.zero_grad()
    else:
        for parameter in self.model.parameters():
            parameter.grad = None
    loss.backward()

    if self.multi_gpu:
        self._grasp_gradient_reducer.reduce(self.model.parameters(), world_size=self.world_size)
    if self.truncate_grads:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
    self.optimizer.step()
    return loss


class GraspCompletionPpoAgent(A2CAgent):
    """Use the stock continuous PPO loop with a hybrid distribution KL."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._grasp_gradient_reducer = _ReusableGradientAllReduce()
        self._memory_log_warning_emitted = False
        if self.has_central_value:
            self.central_value_net._grasp_gradient_reducer = _ReusableGradientAllReduce()
            self.central_value_net.calc_gradients = MethodType(
                _central_value_calc_gradients_with_reusable_buffer,
                self.central_value_net,
            )

    @staticmethod
    def _policy_train_mask(observations: torch.Tensor) -> torch.Tensor:
        """Return one for policy-controlled transitions and zero for lift phases."""

        # The final privileged completion-supervision value is -1 only while
        # the environment owns the scripted close/lift option. It is sliced
        # away before the actor trunk and therefore is never a policy input.
        return (observations[..., -1] >= 0.0).to(dtype=torch.float32)

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        """Use undiscounted GAE across the controller-owned lift option.

        The declaration transition and subsequent scripted transitions form
        one semi-MDP option. Normal motion retains the configured gamma/tau;
        option transitions use gamma=lambda=1 so the measured terminal pickup
        outcome reaches the declaration action without exponential decay.
        """

        observations = self.experience_buffer.tensor_dict["obses"]
        if not isinstance(observations, torch.Tensor):
            return super().discount_values(fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards)
        scripted = self._policy_train_mask(observations) < 0.5
        final_observation = self.obs["obs"] if isinstance(self.obs, dict) else self.obs
        final_scripted = self._policy_train_mask(final_observation) < 0.5
        next_scripted = torch.cat((scripted[1:], final_scripted.unsqueeze(0)), dim=0)
        option_transition = scripted | next_scripted
        if not option_transition.any():
            return super().discount_values(fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards)

        last_gae = torch.zeros_like(last_extrinsic_values)
        advantages = torch.zeros_like(mb_rewards)
        for step in reversed(range(self.horizon_length)):
            if step == self.horizon_length - 1:
                next_nonterminal = 1.0 - fdones.float()
                next_values = last_extrinsic_values
            else:
                next_nonterminal = 1.0 - mb_fdones[step + 1].float()
                next_values = mb_extrinsic_values[step + 1]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            option = option_transition[step].unsqueeze(1)
            gamma = torch.where(option, mb_rewards.new_ones(()), mb_rewards.new_tensor(self.gamma))
            gae_lambda = torch.where(option, mb_rewards.new_ones(()), mb_rewards.new_tensor(self.tau))
            delta = mb_rewards[step] + gamma * next_values * next_nonterminal - mb_extrinsic_values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        return advantages

    def prepare_dataset(self, batch_dict):
        """Normalize actor advantages over policy-controlled samples only."""

        raw_advantages = torch.sum(batch_dict["returns"] - batch_dict["values"], dim=1)
        train_mask = self._policy_train_mask(batch_dict["obses"]).bool()
        actor_advantages = torch.zeros_like(raw_advantages)
        if train_mask.any():
            active = raw_advantages[train_mask]
            if self.normalize_advantage:
                active = (active - active.mean()) / (active.std(unbiased=False) + 1.0e-8)
            actor_advantages[train_mask] = active
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["advantages"] = actor_advantages

    @staticmethod
    def _hybrid_kl(
        p0_mu: torch.Tensor,
        p0_sigma: torch.Tensor,
        p1_mu: torch.Tensor,
        p1_sigma: torch.Tensor,
        *,
        reduce: bool,
    ) -> torch.Tensor:
        motion_kl = torch_ext.policy_kl(
            p0_mu[..., :-1],
            p0_sigma[..., :-1],
            p1_mu[..., :-1],
            p1_sigma[..., :-1],
            reduce=False,
        )
        completion_kl = bernoulli_kl_from_probabilities(p0_mu[..., -1], p1_mu[..., -1])
        combined = motion_kl + completion_kl
        return combined.mean() if reduce else combined

    def calc_gradients(self, input_dict):  # noqa: C901 - mirrors installed RL-Games PPO core
        """Stock PPO gradient step with hybrid KL instead of all-Gaussian KL."""

        from rl_games.algos_torch import torch_ext
        from rl_games.common import common_losses

        value_predictions = input_dict["old_values"]
        old_action_log_probs = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu = input_dict["mu"]
        old_sigma = input_dict["sigma"]
        returns = input_dict["returns"]
        actions = input_dict["actions"]
        observations = self._preproc_obs(input_dict["obs"])
        policy_train_mask = self._policy_train_mask(observations)
        active_fraction = policy_train_mask.mean()
        policy_loss_scale = policy_train_mask / active_fraction.clamp_min(1.0e-6)
        current_clip = self.e_clip
        batch_dict = {
            "is_train": True,
            "prev_actions": actions,
            "obs": observations,
        }
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.amp.autocast(device_type="cuda", enabled=self.mixed_precision):
            result = self.model(batch_dict)
            action_log_probs = result["prev_neglogp"]
            values = result["values"]
            entropy = result["entropy"]
            mu = result["mus"]
            sigma = result["sigmas"]
            completion_probability_mean = (
                result["completion_probability"].squeeze(-1) * policy_train_mask
            ).sum() / policy_train_mask.sum().clamp_min(1.0)
            actor_loss = self.actor_loss_func(
                old_action_log_probs,
                action_log_probs,
                advantage,
                self.ppo,
                current_clip,
            )
            if self.has_value_loss:
                critic_loss = common_losses.critic_loss(
                    self.model,
                    value_predictions,
                    values,
                    current_clip,
                    returns,
                    self.clip_value,
                )
            else:
                critic_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                bounds_loss = self.reg_loss(mu[..., :-1])
            elif self.bound_loss_type == "bound":
                bounds_loss = self.bound_loss(mu[..., :-1])
            else:
                bounds_loss = torch.zeros(1, device=self.ppo_device)
            actor_loss = actor_loss * policy_loss_scale
            entropy = entropy * policy_loss_scale
            if bounds_loss.ndim > 0 and bounds_loss.numel() == policy_loss_scale.numel():
                bounds_loss = bounds_loss * policy_loss_scale
            losses, _ = torch_ext.apply_masks(
                [
                    actor_loss.unsqueeze(1),
                    critic_loss,
                    entropy.unsqueeze(1),
                    bounds_loss.unsqueeze(1),
                ],
                rnn_masks,
            )
            actor_loss, critic_loss, entropy, bounds_loss = losses
            loss = (
                actor_loss
                + 0.5 * critic_loss * self.critic_coef
                - entropy * self.entropy_coef
                + bounds_loss * self.bounds_loss_coef
            )
            auxiliary_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if auxiliary_loss is not None:
                for name, value in auxiliary_loss.items():
                    loss += value
                    self.aux_loss_dict.setdefault(name, []).append(value.detach())
            # RL-Games writes auxiliary entries under losses/. This value is a
            # detached diagnostic and is deliberately added only after the
            # loss terms have been accumulated.
            self.aux_loss_dict["completion_probability_mean"] = [completion_probability_mean.detach()]
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for parameter in self.model.parameters():
                    parameter.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()
        with torch.no_grad():
            kl_distance = self._hybrid_kl(
                mu.detach(),
                sigma.detach(),
                old_mu.detach(),
                old_sigma.detach(),
                reduce=False,
            )
            kl_mask = policy_train_mask
            if rnn_masks is not None:
                kl_mask = kl_mask * rnn_masks
            kl_distance = (kl_distance * kl_mask).sum() / kl_mask.sum().clamp_min(1.0)

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_predictions,
                "returns": returns,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs,
                "masks": rnn_masks,
            },
            current_clip,
            0,
        )
        self.train_result = (
            actor_loss,
            critic_loss,
            entropy,
            kl_distance,
            self.last_lr,
            1.0,
            mu.detach(),
            sigma.detach(),
            bounds_loss,
        )

    def trancate_gradients_and_step(self) -> None:
        """Step PPO without allocating a new flattened NCCL buffer per minibatch."""

        if self.multi_gpu:
            self._grasp_gradient_reducer.reduce(self.model.parameters(), world_size=self.world_size)
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_epoch(self):
        result = super().train_epoch()
        self._record_cuda_memory()
        return result

    def _record_cuda_memory(self) -> None:
        """Record synchronized per-epoch allocator and whole-device VRAM use."""

        if not torch.cuda.is_available():
            return
        device_index = torch.cuda.current_device()
        torch.cuda.synchronize(device_index)
        mib = 1024.0 * 1024.0
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        # Use the CUDA device selected by this task. NVML indices can differ
        # from CUDA ordinals under Slurm's per-task GPU visibility.
        device_used_bytes = total_bytes - free_bytes
        central_reducer = getattr(getattr(self, "central_value_net", None), "_grasp_gradient_reducer", None)
        row = {
            "epoch": int(self.epoch_num),
            "timestamp_unix": time.time(),
            "allocated_mib": torch.cuda.memory_allocated(device_index) / mib,
            "reserved_mib": torch.cuda.memory_reserved(device_index) / mib,
            "epoch_peak_allocated_mib": torch.cuda.max_memory_allocated(device_index) / mib,
            "epoch_peak_reserved_mib": torch.cuda.max_memory_reserved(device_index) / mib,
            "device_used_mib": device_used_bytes / mib,
            "device_free_mib": free_bytes / mib,
            "device_total_mib": total_bytes / mib,
            "actor_gradient_buffer_mib": self._grasp_gradient_reducer.size_mib,
            "central_gradient_buffer_mib": central_reducer.size_mib if central_reducer is not None else 0.0,
        }
        try:
            status_lines = Path("/proc/self/status").read_text().splitlines()
            row["cpu_rss_mib"] = (
                float(next(line.split()[1] for line in status_lines if line.startswith("VmRSS:"))) / 1024.0
            )
        except (OSError, StopIteration, ValueError):
            row["cpu_rss_mib"] = 0.0
        path = Path(self.experiment_dir) / f"gpu_memory_rank_{self.global_rank}.csv"
        try:
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:  # diagnostics must not destroy a long training run
            if not self._memory_log_warning_emitted:
                print(f"[GPU_MEMORY][WARNING] rank={self.global_rank} could not write {path}: {exc}", flush=True)
                self._memory_log_warning_emitted = True
        print(
            f"[GPU_MEMORY] rank={self.global_rank} epoch={self.epoch_num} "
            f"allocated_mib={row['allocated_mib']:.1f} reserved_mib={row['reserved_mib']:.1f} "
            f"device_used_mib={row['device_used_mib']:.1f} free_mib={row['device_free_mib']:.1f} "
            f"gradient_buffers_mib="
            f"{row['actor_gradient_buffer_mib'] + row['central_gradient_buffer_mib']:.1f}",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats(device_index)


class GraspCompletionPpoPlayer(PpoPlayerContinuous):
    """Continuous player whose seventh deterministic action is p(done)."""


def register_grasp_completion_runner(runner) -> None:
    """Install the custom PPO algorithm/player on one RL-Games runner."""

    runner.algo_factory.register_builder(
        "grasp_completion_ppo",
        lambda **kwargs: GraspCompletionPpoAgent(**kwargs),
    )
    runner.player_factory.register_builder(
        "grasp_completion_ppo",
        lambda **kwargs: GraspCompletionPpoPlayer(**kwargs),
    )


__all__ = [
    "GraspCompletionPpoAgent",
    "GraspCompletionPpoPlayer",
    "register_grasp_completion_runner",
]
