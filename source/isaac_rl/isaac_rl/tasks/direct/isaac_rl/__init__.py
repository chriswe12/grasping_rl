# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Grasp-Franka-ZEDMini-RGBD-Direct-v0",
    entry_point=f"{__name__}.franka_zed_env:FrankaZedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_zed_env:FrankaZedEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

##
# Register Gym environments.
##


gym.register(
    id="Grasp-Visual-Servo-RGBD-Direct-v0",
    entry_point=f"{__name__}.isaac_rl_env:GraspVisualServoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_rl_env_cfg:GraspVisualServoEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0",
    entry_point=f"{__name__}.isaac_rl_env:GraspVisualServoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_rl_env_cfg:GraspVisualServoMultiPartEnvCfg"
        ),
        "rl_games_cfg_entry_point": (
            f"{agents.__name__}:rl_games_multipart_ppo_cfg.yaml"
        ),
    },
)

gym.register(
    id="Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0",
    entry_point=f"{__name__}.isaac_rl_env:GraspVisualServoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_rl_env_cfg:GraspVisualServoMultiPartEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": (
            f"{agents.__name__}:rl_games_multipart_ppo_cfg.yaml"
        ),
    },
)

gym.register(
    id="Grasp-Visual-Servo-RGBD-Direct-Play-v0",
    entry_point=f"{__name__}.isaac_rl_env:GraspVisualServoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_rl_env_cfg:GraspVisualServoEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0",
    entry_point=f"{__name__}.isaac_rl_env:GraspVisualServoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_rl_env_cfg:GraspVisualServoFabricaAllEnvCfg"
        ),
        "rl_games_cfg_entry_point": (
            f"{agents.__name__}:rl_games_multipart_ppo_cfg.yaml"
        ),
    },
)

gym.register(
    id="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-Play-v0",
    entry_point=f"{__name__}.isaac_rl_env:GraspVisualServoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_rl_env_cfg:GraspVisualServoFabricaAllEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": (
            f"{agents.__name__}:rl_games_multipart_ppo_cfg.yaml"
        ),
    },
)
