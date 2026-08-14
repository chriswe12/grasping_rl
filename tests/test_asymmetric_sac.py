"""CPU tests for the standalone asymmetric-SAC components."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = Path(__file__).parents[1] / "source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/asymmetric_sac.py"
MODULE_NAME = "isaac_rl_asymmetric_sac_test_module"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
try:
    SPEC.loader.exec_module(MODULE)
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"asymmetric SAC dependencies unavailable: {exc}", allow_module_level=True)


def _policy_observation(batch_size: int, spec) -> torch.Tensor:
    image = torch.rand(batch_size, spec.image_height, spec.image_width, spec.image_channels)
    context = torch.rand(batch_size, spec.context_size) * 2.0 - 1.0
    pose = torch.rand(batch_size, spec.pose_target_size) * 2.0 - 1.0
    completion = torch.tensor([[0.0, 1.0], [1.0, 1.0]])[:batch_size]
    return torch.cat((image.flatten(start_dim=1), context, pose, completion), dim=-1)


def test_actor_outputs_six_motion_values_and_does_not_read_labels():
    spec = MODULE.VisualObservationSpec()
    actor = MODULE.AsymmetricSacActor(observation_spec=spec, pretrained=False).eval()
    observation = _policy_observation(1, spec)
    changed_labels = observation.clone()
    changed_labels[:, -(spec.pose_target_size + spec.completion_target_size) :] = 123.0

    with torch.inference_mode():
        output = actor.from_observation(observation, deterministic=True)
        changed = actor.from_observation(changed_labels, deterministic=True)

    assert output.action.shape == (1, 6)
    assert output.log_probability.shape == (1, 1)
    assert output.pose_prediction.shape == (1, 6)
    assert output.completion_logits.shape == (1, 1)
    torch.testing.assert_close(output.action, changed.action)
    torch.testing.assert_close(output.completion_logits, changed.completion_logits)


def test_twin_critics_use_privileged_state_and_motion_action():
    critic = MODULE.TwinQCritic(state_size=26, action_size=6)
    q1, q2 = critic(torch.randn(3, 26), torch.randn(3, 6))
    assert q1.shape == (3, 1)
    assert q2.shape == (3, 1)
    assert critic.q1 is not critic.q2


def test_compressed_replay_round_trip_and_memory_bound():
    spec = MODULE.VisualObservationSpec()
    replay = MODULE.CompressedVisualReplayBuffer(4, observation_spec=spec)
    observation = _policy_observation(2, spec)
    next_observation = _policy_observation(2, spec)
    goal_catalog = torch.rand(3, spec.image_height, spec.image_width, 4)
    replay.add(
        observation=observation,
        critic_state=torch.randn(2, 26),
        goal_index=torch.tensor([0, 2]),
        action=torch.randn(2, 6).clamp(-1.0, 1.0),
        reward=torch.tensor([1.0, -1.0]),
        done=torch.tensor([False, True]),
        next_observation=next_observation,
        next_critic_state=torch.randn(2, 26),
        next_goal_index=torch.tensor([1, 0]),
    )
    batch = replay.sample(2, device="cpu", goal_rgbd_catalog=goal_catalog)

    assert len(replay) == 2
    assert batch.image.shape == (2, 8, 72, 128)
    assert batch.next_image.shape == (2, 8, 72, 128)
    assert batch.critic_state.shape == (2, 26)
    assert batch.action.shape == (2, 6)
    assert batch.done.min() >= 0.0 and batch.done.max() <= 1.0
    # Two compressed live frames plus low-dimensional data stay below 100 KiB
    # per transition; full current/next float32 observations need ~576 KiB.
    assert replay.allocated_bytes / replay.capacity < 100 * 1024


def test_supervision_mask_removes_ambiguous_completion_examples():
    output = MODULE.ActorOutput(
        action=torch.zeros(2, 6),
        log_probability=torch.zeros(2, 1),
        mean_action=torch.zeros(2, 6),
        completion_logits=torch.tensor([[0.0], [100.0]], requires_grad=True),
        pose_prediction=torch.zeros(2, 6, requires_grad=True),
    )
    pose_target = torch.zeros(2, 6)
    completion_target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss, metrics = MODULE.AsymmetricSacActor.auxiliary_loss(
        output,
        pose_target,
        completion_target,
        pose_weight=0.2,
        completion_weight=0.2,
        completion_positive_weight=3.0,
    )
    assert torch.isfinite(loss)
    assert float(metrics["completion_aux_loss"]) == pytest.approx(0.2 * 3.0 * 0.693147, rel=1.0e-5)


def test_agent_update_optimizes_actor_critics_and_temperature():
    class TinyActor(torch.nn.Module):
        action_size = 6

        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Sequential(torch.nn.Linear(14, 16), torch.nn.ELU())
            self.mean = torch.nn.Linear(16, 6)
            self.log_std = torch.nn.Linear(16, 6)
            self.pose = torch.nn.Linear(16, 6)
            self.completion = torch.nn.Linear(16, 1)

        def forward(self, image, context, *, deterministic=False, with_log_probability=True):
            features = self.trunk(torch.cat((image.mean(dim=(-2, -1)), context), dim=-1))
            mean = self.mean(features)
            distribution = torch.distributions.Normal(mean, self.log_std(features).clamp(-5.0, 2.0).exp())
            pre_tanh = mean if deterministic else distribution.rsample()
            action = torch.tanh(pre_tanh)
            log_probability = None
            if with_log_probability:
                log_probability = (distribution.log_prob(pre_tanh) - torch.log(1.0 - action.square() + 1.0e-6)).sum(
                    dim=-1, keepdim=True
                )
            return MODULE.ActorOutput(
                action=action,
                log_probability=log_probability,
                mean_action=torch.tanh(mean),
                completion_logits=self.completion(features),
                pose_prediction=self.pose(features),
            )

        auxiliary_loss = staticmethod(MODULE.AsymmetricSacActor.auxiliary_loss)

    actor = TinyActor()
    critic = MODULE.TwinQCritic()
    agent = MODULE.AsymmetricSacAgent(actor, critic, device="cpu")
    batch = MODULE.SacBatch(
        image=torch.rand(4, 8, 3, 3),
        context=torch.rand(4, 6),
        pose_target=torch.rand(4, 6),
        completion_target=torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]]),
        critic_state=torch.rand(4, 26),
        action=torch.rand(4, 6) * 2.0 - 1.0,
        reward=torch.rand(4, 1),
        done=torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        next_image=torch.rand(4, 8, 3, 3),
        next_context=torch.rand(4, 6),
        next_critic_state=torch.rand(4, 26),
    )
    before = actor.mean.weight.detach().clone()
    metrics = agent.update(batch)

    assert agent.update_count == 1
    assert not torch.equal(before, actor.mean.weight)
    assert all(math.isfinite(value) for value in metrics.values())
