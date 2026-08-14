"""PPO integration for the Gaussian-motion plus Bernoulli-stop policy."""

from __future__ import annotations

import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.players import PpoPlayerContinuous

from .completion_model import bernoulli_kl_from_probabilities


class GraspCompletionPpoAgent(A2CAgent):
    """Use the stock continuous PPO loop with a hybrid distribution KL."""

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
            completion_probability_mean = result["completion_probability"].mean()
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
            reduce_kl = rnn_masks is None
            kl_distance = self._hybrid_kl(
                mu.detach(),
                sigma.detach(),
                old_mu.detach(),
                old_sigma.detach(),
                reduce=reduce_kl,
            )
            if rnn_masks is not None:
                kl_distance = (kl_distance * rnn_masks).sum() / rnn_masks.numel()

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
