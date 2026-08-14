# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from rl_games.algos_torch import model_builder

from .completion_model import GraspCompletionModel
from .resnet_rgbd_network import GraspRgbdResNetBuilder


model_builder.register_network("grasp_rgbd_resnet18", GraspRgbdResNetBuilder)
model_builder.register_model("grasp_completion_hybrid", GraspCompletionModel)
