# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import table_clean_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Table-Clean-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.table_clean_env_cfg:TableCleanIKEnvCfg",
    },
    disable_env_checker=True,
)
