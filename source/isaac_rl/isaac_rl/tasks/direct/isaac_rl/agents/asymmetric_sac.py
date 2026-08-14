"""Asymmetric SAC for goal-conditioned RGB-D visual servoing.

The deployable actor sees only the live/goal RGB-D pair and the previous motion
command.  Pose and completion values at the tail of the policy observation are
training labels, never actor inputs.  Twin critics consume the environment's
privileged 26D state and the actor's six-dimensional motion action.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

if os.environ.get("ISAAC_RL_DISABLE_CUDNN") == "1":
    torch.backends.cudnn.enabled = False


@dataclass(frozen=True)
class VisualObservationSpec:
    """Shape contract shared by the actor and compressed replay."""

    image_height: int = 72
    image_width: int = 128
    image_channels: int = 8
    context_size: int = 6
    pose_target_size: int = 6
    completion_target_size: int = 2

    @property
    def image_values(self) -> int:
        return self.image_height * self.image_width * self.image_channels

    @property
    def policy_values(self) -> int:
        return self.image_values + self.context_size + self.pose_target_size + self.completion_target_size


@dataclass
class ActorOutput:
    """One actor forward pass, including auxiliary predictions."""

    action: torch.Tensor
    log_probability: torch.Tensor | None
    mean_action: torch.Tensor
    completion_logits: torch.Tensor
    pose_prediction: torch.Tensor


@dataclass
class SacBatch:
    """Decoded replay minibatch on the training device."""

    image: torch.Tensor
    context: torch.Tensor
    pose_target: torch.Tensor
    completion_target: torch.Tensor
    critic_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_image: torch.Tensor
    next_context: torch.Tensor
    next_critic_state: torch.Tensor


def split_policy_observation(
    observation: torch.Tensor,
    spec: VisualObservationSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the flattened environment observation without leaking labels."""

    if observation.ndim != 2 or observation.shape[1] != spec.policy_values:
        raise ValueError(f"Expected policy observation [B, {spec.policy_values}], received {tuple(observation.shape)}.")
    image_end = spec.image_values
    context_end = image_end + spec.context_size
    pose_end = context_end + spec.pose_target_size
    image = observation[:, :image_end].view(
        -1,
        spec.image_height,
        spec.image_width,
        spec.image_channels,
    )
    image = image.permute(0, 3, 1, 2).contiguous()
    return (
        image,
        observation[:, image_end:context_end],
        observation[:, context_end:pose_end],
        observation[:, pose_end:],
    )


class AsymmetricSacActor(nn.Module):
    """Siamese RGB-D actor with supervised geometry and completion heads."""

    def __init__(
        self,
        *,
        observation_spec: VisualObservationSpec | None = None,
        action_size: int = 6,
        geometry_feature_size: int = 128,
        pretrained: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.observation_spec = observation_spec or VisualObservationSpec()
        self.action_size = int(action_size)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.rgb_stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.rgb_layer1 = backbone.layer1
        self.rgb_layer2 = backbone.layer2
        self.rgb_layer3 = backbone.layer3
        for module in (self.rgb_stem, self.rgb_layer1, self.rgb_layer2):
            module.requires_grad_(False)
        self.register_buffer(
            "rgb_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        rgbd_channels = 256 + 128
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(5 * rgbd_channels, 256, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.policy_trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 8, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
        )
        self.geometry_trunk = nn.Sequential(nn.Linear(256, geometry_feature_size), nn.ELU())
        shared_size = 256 + geometry_feature_size
        self.motion_trunk = nn.Sequential(
            nn.Linear(shared_size + self.observation_spec.context_size, 256),
            nn.ELU(),
        )
        self.motion_mean = nn.Linear(256, self.action_size)
        self.motion_log_std = nn.Linear(256, self.action_size)
        self.pose_head = nn.Sequential(
            nn.Linear(geometry_feature_size, 128),
            nn.ELU(),
            nn.Linear(128, self.observation_spec.pose_target_size),
        )
        self.completion_head = nn.Sequential(nn.Linear(shared_size, 128), nn.ELU(), nn.Linear(128, 1))

        for module in (
            self.depth_encoder,
            self.spatial_fusion,
            self.policy_trunk,
            self.geometry_trunk,
            self.motion_trunk,
            self.motion_mean,
            self.motion_log_std,
            self.pose_head,
            self.completion_head,
        ):
            for layer in module.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.motion_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.motion_log_std.weight, gain=0.01)
        nn.init.constant_(self.motion_log_std.bias, -1.5)
        nn.init.orthogonal_(self.completion_head[-1].weight, gain=0.01)
        nn.init.constant_(self.completion_head[-1].bias, -3.0)

    def train(self, mode: bool = True):
        super().train(mode)
        # A replay minibatch is not an on-policy sample from the current image
        # distribution. Frozen running statistics avoid train/eval drift.
        for module in (self.rgb_stem, self.rgb_layer1, self.rgb_layer2, self.rgb_layer3):
            for layer in module.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        with torch.no_grad():
            rgb = self.rgb_stem(rgb)
            rgb = self.rgb_layer1(rgb)
            rgb = self.rgb_layer2(rgb)
        return self.rgb_layer3(rgb)

    def _visual_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1:] != (
            self.observation_spec.image_channels,
            self.observation_spec.image_height,
            self.observation_spec.image_width,
        ):
            raise ValueError(f"Expected RGB-D image [B, 8, 72, 128], received {tuple(image.shape)}.")
        live_rgb, live_depth = image[:, 0:3], image[:, 3:4]
        goal_rgb, goal_depth = image[:, 4:7], image[:, 7:8]
        paired_rgb = self._encode_rgb(torch.cat((live_rgb, goal_rgb), dim=0))
        live_rgb_features, goal_rgb_features = paired_rgb.chunk(2, dim=0)
        paired_depth = self.depth_encoder(torch.cat((live_depth, goal_depth), dim=0))
        live_depth_features, goal_depth_features = paired_depth.chunk(2, dim=0)
        live = torch.cat((live_rgb_features, live_depth_features), dim=1)
        goal = torch.cat((goal_rgb_features, goal_depth_features), dim=1)
        difference = live - goal
        fused = torch.cat((live, goal, difference, difference.abs(), live * goal), dim=1)
        latent = self.policy_trunk(self.spatial_fusion(fused))
        geometry = self.geometry_trunk(latent)
        return latent, geometry

    def forward(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
        *,
        deterministic: bool = False,
        with_log_probability: bool = True,
    ) -> ActorOutput:
        latent, geometry = self._visual_features(image)
        shared = torch.cat((latent, geometry), dim=-1)
        motion_features = self.motion_trunk(torch.cat((shared, context), dim=-1))
        mean = self.motion_mean(motion_features)
        log_std = self.motion_log_std(motion_features).clamp(self.log_std_min, self.log_std_max)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = None
        if with_log_probability:
            log_probability = distribution.log_prob(pre_tanh)
            log_probability -= torch.log(1.0 - action.square() + 1.0e-6)
            log_probability = log_probability.sum(dim=-1, keepdim=True)
        return ActorOutput(
            action=action,
            log_probability=log_probability,
            mean_action=torch.tanh(mean),
            completion_logits=self.completion_head(shared),
            pose_prediction=self.pose_head(geometry),
        )

    def from_observation(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
        with_log_probability: bool = True,
    ) -> ActorOutput:
        image, context, _, _ = split_policy_observation(observation, self.observation_spec)
        return self(
            image,
            context,
            deterministic=deterministic,
            with_log_probability=with_log_probability,
        )

    @staticmethod
    def auxiliary_loss(
        output: ActorOutput,
        pose_target: torch.Tensor,
        completion_target: torch.Tensor,
        *,
        pose_weight: float,
        completion_weight: float,
        completion_positive_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        position_loss = F.smooth_l1_loss(output.pose_prediction[:, :3], pose_target[:, :3])
        rotation_loss = F.smooth_l1_loss(output.pose_prediction[:, 3:], pose_target[:, 3:])
        pose_loss = float(pose_weight) * (position_loss + rotation_loss)
        completion_label = completion_target[:, 0]
        completion_supervised = completion_target[:, 1]
        raw_completion_loss = F.binary_cross_entropy_with_logits(
            output.completion_logits.squeeze(-1),
            completion_label,
            reduction="none",
            pos_weight=output.completion_logits.new_tensor(float(completion_positive_weight)),
        )
        completion_loss = float(completion_weight) * (
            (raw_completion_loss * completion_supervised).sum() / completion_supervised.sum().clamp_min(1.0)
        )
        return pose_loss + completion_loss, {
            "pose_aux_loss": pose_loss.detach(),
            "completion_aux_loss": completion_loss.detach(),
        }


class QNetwork(nn.Module):
    """One privileged state-action value network."""

    def __init__(self, state_size: int = 26, action_size: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        for layer in self.network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, action), dim=-1))


class TwinQCritic(nn.Module):
    """Independent clipped-double-Q critics."""

    def __init__(self, state_size: int = 26, action_size: int = 6) -> None:
        super().__init__()
        self.q1 = QNetwork(state_size, action_size)
        self.q2 = QNetwork(state_size, action_size)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)


class CompressedVisualReplayBuffer:
    """CPU replay that stores only live frames and goal-catalog indices.

    RGB is quantized to uint8 and normalized depth to offset int16. Canonical goal
    images are not duplicated per transition; they are gathered from the
    environment's goal catalog when a minibatch is decoded.
    """

    def __init__(
        self,
        capacity: int,
        *,
        observation_spec: VisualObservationSpec | None = None,
        critic_state_size: int = 26,
        action_size: int = 6,
        pin_memory: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.spec = observation_spec or VisualObservationSpec()
        self.critic_state_size = int(critic_state_size)
        self.action_size = int(action_size)
        options = {"device": "cpu", "pin_memory": bool(pin_memory)}
        frame_shape = (self.capacity, self.spec.image_height, self.spec.image_width)
        self.live_rgb = torch.empty((*frame_shape, 3), dtype=torch.uint8, **options)
        self.live_depth = torch.empty((*frame_shape, 1), dtype=torch.int16, **options)
        self.next_live_rgb = torch.empty((*frame_shape, 3), dtype=torch.uint8, **options)
        self.next_live_depth = torch.empty((*frame_shape, 1), dtype=torch.int16, **options)
        self.goal_index = torch.empty(self.capacity, dtype=torch.int32, **options)
        self.next_goal_index = torch.empty(self.capacity, dtype=torch.int32, **options)
        self.context = torch.empty((self.capacity, self.spec.context_size), dtype=torch.float16, **options)
        self.next_context = torch.empty((self.capacity, self.spec.context_size), dtype=torch.float16, **options)
        self.pose_target = torch.empty((self.capacity, self.spec.pose_target_size), dtype=torch.float16, **options)
        self.completion_target = torch.empty(
            (self.capacity, self.spec.completion_target_size), dtype=torch.uint8, **options
        )
        self.critic_state = torch.empty((self.capacity, self.critic_state_size), dtype=torch.float16, **options)
        self.next_critic_state = torch.empty((self.capacity, self.critic_state_size), dtype=torch.float16, **options)
        self.action = torch.empty((self.capacity, self.action_size), dtype=torch.float16, **options)
        self.reward = torch.empty((self.capacity, 1), dtype=torch.float32, **options)
        self.done = torch.empty((self.capacity, 1), dtype=torch.uint8, **options)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def allocated_bytes(self) -> int:
        tensors = (
            self.live_rgb,
            self.live_depth,
            self.next_live_rgb,
            self.next_live_depth,
            self.goal_index,
            self.next_goal_index,
            self.context,
            self.next_context,
            self.pose_target,
            self.completion_target,
            self.critic_state,
            self.next_critic_state,
            self.action,
            self.reward,
            self.done,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @staticmethod
    def _to_cpu(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=dtype)

    @staticmethod
    def _write(storage: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
        storage.index_copy_(0, indices, values)

    def _pack_live(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        live = image[:, 0:4].permute(0, 2, 3, 1)
        rgb = self._to_cpu((live[..., :3] * 255.0).round().clamp(0.0, 255.0), torch.uint8)
        depth = self._to_cpu(
            (live[..., 3:4] * 65535.0).round().clamp(0.0, 65535.0) - 32768.0,
            torch.int16,
        )
        return rgb, depth

    def add(
        self,
        *,
        observation: torch.Tensor,
        critic_state: torch.Tensor,
        goal_index: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_observation: torch.Tensor,
        next_critic_state: torch.Tensor,
        next_goal_index: torch.Tensor,
    ) -> None:
        image, context, pose_target, completion_target = split_policy_observation(observation, self.spec)
        next_image, next_context, _, _ = split_policy_observation(next_observation, self.spec)
        batch_size = observation.shape[0]
        if batch_size > self.capacity:
            start = batch_size - self.capacity
            image, context, pose_target, completion_target = (
                image[start:],
                context[start:],
                pose_target[start:],
                completion_target[start:],
            )
            next_image, next_context = next_image[start:], next_context[start:]
            critic_state, goal_index, action, reward, done = (
                critic_state[start:],
                goal_index[start:],
                action[start:],
                reward[start:],
                done[start:],
            )
            next_critic_state, next_goal_index = next_critic_state[start:], next_goal_index[start:]
            batch_size = self.capacity
        indices = (torch.arange(batch_size, dtype=torch.long) + self.position) % self.capacity
        live_rgb, live_depth = self._pack_live(image)
        next_live_rgb, next_live_depth = self._pack_live(next_image)
        self._write(self.live_rgb, indices, live_rgb)
        self._write(self.live_depth, indices, live_depth)
        self._write(self.next_live_rgb, indices, next_live_rgb)
        self._write(self.next_live_depth, indices, next_live_depth)
        self._write(self.goal_index, indices, self._to_cpu(goal_index, torch.int32))
        self._write(self.next_goal_index, indices, self._to_cpu(next_goal_index, torch.int32))
        self._write(self.context, indices, self._to_cpu(context, torch.float16))
        self._write(self.next_context, indices, self._to_cpu(next_context, torch.float16))
        self._write(self.pose_target, indices, self._to_cpu(pose_target, torch.float16))
        completion_u8 = self._to_cpu(completion_target.round().clamp(0.0, 1.0), torch.uint8)
        self._write(self.completion_target, indices, completion_u8)
        self._write(self.critic_state, indices, self._to_cpu(critic_state, torch.float16))
        self._write(self.next_critic_state, indices, self._to_cpu(next_critic_state, torch.float16))
        self._write(self.action, indices, self._to_cpu(action, torch.float16))
        self._write(self.reward, indices, self._to_cpu(reward.reshape(-1, 1), torch.float32))
        self._write(self.done, indices, self._to_cpu(done.reshape(-1, 1), torch.uint8))
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def _decode_image(
        self,
        indices: torch.Tensor,
        goal_indices: torch.Tensor,
        goal_rgbd_catalog: torch.Tensor,
        *,
        next_frame: bool,
        device: torch.device,
    ) -> torch.Tensor:
        rgb_store = self.next_live_rgb if next_frame else self.live_rgb
        depth_store = self.next_live_depth if next_frame else self.live_depth
        live_rgb = rgb_store[indices].to(device=device, dtype=torch.float32).div_(255.0)
        live_depth = depth_store[indices].to(device=device, dtype=torch.float32).add_(32768.0).div_(65535.0)
        goals = goal_rgbd_catalog[goal_indices.to(device=device, dtype=torch.long)]
        image = torch.cat((live_rgb, live_depth, goals), dim=-1)
        return image.permute(0, 3, 1, 2).contiguous()

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        goal_rgbd_catalog: torch.Tensor,
    ) -> SacBatch:
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} transitions, cannot sample {batch_size}.")
        device = torch.device(device)
        indices = torch.randint(self.size, (batch_size,), device="cpu")
        goal_indices = self.goal_index[indices]
        next_goal_indices = self.next_goal_index[indices]
        image = self._decode_image(
            indices,
            goal_indices,
            goal_rgbd_catalog,
            next_frame=False,
            device=device,
        )
        next_image = self._decode_image(
            indices,
            next_goal_indices,
            goal_rgbd_catalog,
            next_frame=True,
            device=device,
        )

        def transfer(storage: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return storage[indices].to(device=device, dtype=dtype)

        return SacBatch(
            image=image,
            context=transfer(self.context),
            pose_target=transfer(self.pose_target),
            completion_target=transfer(self.completion_target),
            critic_state=transfer(self.critic_state),
            action=transfer(self.action),
            reward=transfer(self.reward),
            done=transfer(self.done),
            next_image=next_image,
            next_context=transfer(self.next_context),
            next_critic_state=transfer(self.next_critic_state),
        )


class AsymmetricSacAgent:
    """SAC optimizer with a deployable visual actor and privileged critics."""

    def __init__(
        self,
        actor: AsymmetricSacActor,
        critic: TwinQCritic,
        *,
        device: torch.device | str,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_learning_rate: float = 3.0e-4,
        critic_learning_rate: float = 3.0e-4,
        alpha_learning_rate: float = 3.0e-4,
        initial_temperature: float = 0.1,
        target_entropy: float | None = None,
        pose_loss_weight: float = 0.2,
        completion_loss_weight: float = 0.2,
        completion_positive_weight: float = 3.0,
        gradient_norm: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).to(self.device).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.pose_loss_weight = float(pose_loss_weight)
        self.completion_loss_weight = float(completion_loss_weight)
        self.completion_positive_weight = float(completion_positive_weight)
        self.gradient_norm = float(gradient_norm)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_learning_rate))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_learning_rate))
        self.log_alpha = nn.Parameter(torch.tensor(float(initial_temperature), device=self.device).log())
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(alpha_learning_rate))
        self.target_entropy = -float(actor.action_size) if target_entropy is None else float(target_entropy)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.actor.training
        self.actor.eval()
        output = self.actor.from_observation(
            observation,
            deterministic=deterministic,
            with_log_probability=False,
        )
        if was_training:
            self.actor.train()
        return output.action, torch.sigmoid(output.completion_logits)

    def update(self, batch: SacBatch) -> dict[str, float]:
        self.actor.train()
        self.critic.train()
        with torch.no_grad():
            next_output = self.actor(batch.next_image, batch.next_context)
            assert next_output.log_probability is not None
            target_q1, target_q2 = self.target_critic(batch.next_critic_state, next_output.action)
            target_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_output.log_probability
            target = batch.reward + self.gamma * (1.0 - batch.done) * target_value

        q1, q2 = self.critic(batch.critic_state, batch.action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_norm)
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        output = self.actor(batch.image, batch.context)
        assert output.log_probability is not None
        actor_q1, actor_q2 = self.critic(batch.critic_state, output.action)
        sac_actor_loss = (self.alpha.detach() * output.log_probability - torch.minimum(actor_q1, actor_q2)).mean()
        auxiliary_loss, auxiliary_metrics = self.actor.auxiliary_loss(
            output,
            batch.pose_target,
            batch.completion_target,
            pose_weight=self.pose_loss_weight,
            completion_weight=self.completion_loss_weight,
            completion_positive_weight=self.completion_positive_weight,
        )
        actor_loss = sac_actor_loss + auxiliary_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_norm)
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (output.log_probability.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
                strict=True,
            ):
                target_parameter.lerp_(parameter, self.tau)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(sac_actor_loss.detach()),
            "pose_aux_loss": float(auxiliary_metrics["pose_aux_loss"]),
            "completion_aux_loss": float(auxiliary_metrics["completion_aux_loss"]),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "entropy": float(-output.log_probability.detach().mean()),
            "q1": float(q1.detach().mean()),
            "q2": float(q2.detach().mean()),
            "completion_probability": float(torch.sigmoid(output.completion_logits.detach()).mean()),
            "actor_gradient_norm": float(actor_grad_norm),
            "critic_gradient_norm": float(critic_grad_norm),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_count": self.update_count,
        }

    def save(self, path: str | Path, **metadata: Any) -> None:
        payload = self.checkpoint()
        payload.update(metadata)
        torch.save(payload, Path(path))

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        if "critic" in payload:
            self.critic.load_state_dict(payload["critic"])
            self.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.log_alpha.data.copy_(payload.get("log_alpha", self.log_alpha).to(self.device))
        self.update_count = int(payload.get("update_count", 0))
        return payload


__all__ = [
    "ActorOutput",
    "AsymmetricSacActor",
    "AsymmetricSacAgent",
    "CompressedVisualReplayBuffer",
    "QNetwork",
    "SacBatch",
    "TwinQCritic",
    "VisualObservationSpec",
    "split_policy_observation",
]
