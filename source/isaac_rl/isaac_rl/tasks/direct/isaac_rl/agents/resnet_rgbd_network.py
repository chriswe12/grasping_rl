"""Pretrained spatial RGB-D policy network with privileged pose supervision."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import NetworkBuilder
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

# This image can load kit's cuDNN 9.20 instead of PyTorch's bundled 9.7, causing
# initialization failure on the local 570 driver. The Franka Python wrapper
# selects the bundled 9.7 libraries. Retain an explicit non-cuDNN fallback for
# other callers, without changing their default backend.
if os.environ.get("ISAAC_RL_DISABLE_CUDNN") == "1":
    torch.backends.cudnn.enabled = False


class GraspRgbdResNetNetwork(NetworkBuilder.BaseNetwork):
    """Siamese RGB-D actor with a shared goal-relative geometric trunk."""

    def __init__(self, params: dict, **kwargs):
        super().__init__()
        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.value_size = kwargs.pop("value_size", 1)
        self.central_value = params.get("central_value", False)
        if self.central_value:
            raise ValueError("The visual network must not be used for the centralized critic.")

        self.image_height = int(params.get("image_height", 72))
        self.image_width = int(params.get("image_width", 128))
        self.image_channels = int(params.get("image_channels", 8))
        self.policy_context_size = int(params.get("policy_context_size", 6))
        self.pose_target_size = int(params.get("pose_target_size", 6))
        self.completion_target_size = int(params.get("completion_target_size", 2))
        self.motion_action_size = int(params.get("motion_action_size", 6))
        if actions_num != self.motion_action_size + 1:
            raise ValueError(
                "The hybrid policy requires six motion actions plus one completion "
                f"action, received actions_num={actions_num}."
            )
        self.image_values = self.image_height * self.image_width * self.image_channels
        expected_values = (
            self.image_values + self.policy_context_size + self.pose_target_size + self.completion_target_size
        )
        if tuple(input_shape) != (expected_values,):
            raise ValueError(
                "Expected flattened RGB-D, previous-action context, and privileged "
                "pose/completion target "
                f"shape {(expected_values,)}, "
                f"received {tuple(input_shape)}."
            )

        weights = ResNet18_Weights.DEFAULT if params.get("pretrained", True) else None
        backbone = resnet18(weights=weights)
        self.rgb_stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.rgb_layer1 = backbone.layer1
        self.rgb_layer2 = backbone.layer2
        self.rgb_layer3 = backbone.layer3

        # Preserve generic pretrained edges and shapes. Layer 3 remains
        # trainable so the representation can adapt to wrist-camera geometry.
        # Every BatchNorm layer keeps its pretrained running statistics; PPO
        # repeatedly revisits each rollout batch, which makes mutable running
        # statistics inconsistent between rollout and optimization inference.
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

        # Each live/goal feature has 256 RGB + 128 depth channels. Preserve
        # both features and expose signed difference, magnitude, and agreement.
        rgbd_channels = 256 + 128
        fusion_channels = 5 * rgbd_channels
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 256, kernel_size=1),
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
        geometry_feature_size = int(params.get("geometry_feature_size", 128))
        self.geometry_trunk = nn.Sequential(
            nn.Linear(256, geometry_feature_size),
            nn.ELU(),
        )
        shared_feature_size = 256 + geometry_feature_size
        self.motion_head = nn.Sequential(
            nn.Linear(shared_feature_size + self.policy_context_size, 256),
            nn.ELU(),
            nn.Linear(256, self.motion_action_size),
        )
        self.value = nn.Linear(256, self.value_size)
        self.pose_head = nn.Sequential(
            nn.Linear(geometry_feature_size, 128),
            nn.ELU(),
            nn.Linear(128, self.pose_target_size),
        )
        self.completion_head = nn.Sequential(
            nn.Linear(shared_feature_size, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.completion_motion_slowdown_start = float(params.get("completion_motion_slowdown_start", 0.70))
        self.completion_motion_speed_floor = float(params.get("completion_motion_speed_floor", 0.25))
        if not 0.0 <= self.completion_motion_slowdown_start < 1.0:
            raise ValueError("completion_motion_slowdown_start must lie in [0, 1).")
        if not 0.0 < self.completion_motion_speed_floor <= 1.0:
            raise ValueError("completion_motion_speed_floor must lie in (0, 1].")

        continuous = params["space"]["continuous"]
        sigma_value = float(continuous["sigma_init"].get("val", -1.5))
        self.sigma = nn.Parameter(torch.full((self.motion_action_size,), sigma_value))
        self.pose_loss_weight = float(params.get("pose_loss_weight", 0.2))
        self.completion_loss_weight = float(params.get("completion_loss_weight", 0.2))
        self.completion_positive_weight = float(params.get("completion_positive_weight", 3.0))
        self.aux_loss_map: dict[str, torch.Tensor | None] = {
            "pose_aux_loss": None,
            "completion_aux_loss": None,
        }

        for module in (
            self.depth_encoder,
            self.spatial_fusion,
            self.policy_trunk,
            self.geometry_trunk,
            self.motion_head,
            self.pose_head,
            self.completion_head,
            self.value,
        ):
            for layer in module.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.motion_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.motion_head[-1].bias)
        nn.init.orthogonal_(self.completion_head[-1].weight, gain=0.01)
        nn.init.constant_(self.completion_head[-1].bias, -3.0)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep every ResNet BatchNorm in inference mode, including the
        # trainable layer 3. Its affine parameters remain trainable, but its
        # running statistics cannot drift during repeated PPO mini-epochs.
        for module in (
            self.rgb_stem,
            self.rgb_layer1,
            self.rgb_layer2,
            self.rgb_layer3,
        ):
            for layer in module.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return self.value

    def get_aux_loss(self):
        return self.aux_loss_map

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        with torch.no_grad():
            rgb = self.rgb_stem(rgb)
            rgb = self.rgb_layer1(rgb)
            rgb = self.rgb_layer2(rgb)
        return self.rgb_layer3(rgb)

    def _visual_latent(self, image: torch.Tensor) -> torch.Tensor:
        live_rgb = image[:, 0:3]
        live_depth = image[:, 3:4]
        goal_rgb = image[:, 4:7]
        goal_depth = image[:, 7:8]

        paired_rgb = self._encode_rgb(torch.cat((live_rgb, goal_rgb), dim=0))
        live_rgb_features, goal_rgb_features = paired_rgb.chunk(2, dim=0)
        paired_depth = self.depth_encoder(torch.cat((live_depth, goal_depth), dim=0))
        live_depth_features, goal_depth_features = paired_depth.chunk(2, dim=0)

        live = torch.cat((live_rgb_features, live_depth_features), dim=1)
        goal = torch.cat((goal_rgb_features, goal_depth_features), dim=1)
        difference = live - goal
        fused = torch.cat((live, goal, difference, difference.abs(), live * goal), dim=1)
        return self.policy_trunk(self.spatial_fusion(fused))

    def forward(self, obs_dict: dict):
        observation = obs_dict["obs"]
        image_flat = observation[:, : self.image_values]
        context_end = self.image_values + self.policy_context_size
        policy_context = observation[:, self.image_values : context_end]
        privileged_target = observation[:, context_end:]
        pose_target = privileged_target[:, : self.pose_target_size]
        completion_target = privileged_target[:, self.pose_target_size :]
        image = image_flat.view(-1, self.image_height, self.image_width, self.image_channels)
        image = image.permute(0, 3, 1, 2).contiguous()

        latent = self._visual_latent(image)
        geometry_features = self.geometry_trunk(latent)
        shared_features = torch.cat((latent, geometry_features), dim=-1)
        completion_logits = self.completion_head(shared_features)
        completion_probability = torch.sigmoid(completion_logits)
        slowdown = (
            (completion_probability - self.completion_motion_slowdown_start)
            / (1.0 - self.completion_motion_slowdown_start)
        ).clamp(0.0, 1.0)
        slowdown = slowdown.square() * (3.0 - 2.0 * slowdown)
        motion_scale = 1.0 - slowdown.detach() * (1.0 - self.completion_motion_speed_floor)
        # Bound the Gaussian mean before RL-Games samples from it. The
        # environment still clips sampled exploration actions, but latent
        # drift can no longer make the raw mean (and bounds loss) explode.
        motion_input = torch.cat((shared_features, policy_context), dim=-1)
        mu = torch.tanh(self.motion_head(motion_input)) * motion_scale
        logstd = mu * 0.0 + self.sigma + torch.log(motion_scale.clamp_min(1.0e-4))
        value = self.value(latent)
        pose_prediction = self.pose_head(geometry_features)

        if obs_dict.get("is_train", True):
            policy_train_mask = (completion_target[:, 1] >= 0.0).to(dtype=pose_prediction.dtype)
            active_count = policy_train_mask.sum().clamp_min(1.0)
            position_loss = F.smooth_l1_loss(pose_prediction[:, :3], pose_target[:, :3], reduction="none").mean(dim=-1)
            rotation_loss = F.smooth_l1_loss(pose_prediction[:, 3:], pose_target[:, 3:], reduction="none").mean(dim=-1)
            position_loss = (position_loss * policy_train_mask).sum() / active_count
            rotation_loss = (rotation_loss * policy_train_mask).sum() / active_count
            self.aux_loss_map["pose_aux_loss"] = self.pose_loss_weight * (position_loss + rotation_loss)
            completion_label = completion_target[:, 0]
            completion_supervised = completion_target[:, 1].clamp(0.0, 1.0)
            completion_loss = F.binary_cross_entropy_with_logits(
                completion_logits.squeeze(-1),
                completion_label,
                reduction="none",
                pos_weight=completion_logits.new_tensor(self.completion_positive_weight),
            )
            completion_loss = (completion_loss * completion_supervised).sum() / completion_supervised.sum().clamp_min(
                1.0
            )
            self.aux_loss_map["completion_aux_loss"] = self.completion_loss_weight * completion_loss
        else:
            self.aux_loss_map["pose_aux_loss"] = None
            self.aux_loss_map["completion_aux_loss"] = None

        return mu, logstd, completion_logits, value, None


class GraspRgbdResNetBuilder(NetworkBuilder):
    """RL-Games builder for :class:`GraspRgbdResNetNetwork`."""

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return GraspRgbdResNetNetwork(self.params, **kwargs)
