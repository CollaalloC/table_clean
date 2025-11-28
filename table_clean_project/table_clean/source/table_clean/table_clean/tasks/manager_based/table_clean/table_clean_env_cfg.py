# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

#from . import mdp
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
##
# Pre-defined configs
##
# 引入官方 LiftEnvCfg 作为基类，复用其奖励和观测逻辑
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg as OfficialFrankaIKCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_CLEAN_PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../../.."))
ASSETS_DIR = os.path.join(TABLE_CLEAN_PROJECT_DIR, "assets")
OBJECT_NAMES = ["alphabet_soup", "butter", "cream_cheese", "ketchup", "milk", "orange_juice", "tomato_sauce"]
TABLE_HEIGHT = 45.0
SCENE_SCALE = 150.0
##
# Scene definition
##


@configclass
class TableCleanIKRelEnvCfg(OfficialFrankaIKCfg):
    """
    继承官方的 Franka IK 环境配置。
    我们直接获得了：
    1. Franka High PD 机器人配置
    2. IK Relative 动作空间 (适配 Teleop)
    3. 末端传感器配置 (EE Frame)
    """
    def __post_init__(self):
        # 1. 运行父类初始化 (加载了官方的 Robot, Actions, Sensors)
        super().__post_init__()

        # =====================================================
        # 2. 修改机器人位置 (防碰撞)
        # =====================================================
        # [修正 2] 将机器人移到 y = -20.0，远离巨大的篮子 (y=8.0)
        self.scene.robot.init_state.pos = (0.0, -20.0, TABLE_HEIGHT)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # =====================================================
        # 3. 覆盖场景物体
        # =====================================================
        
        # [A] 覆盖 Table (官方默认是小桌子，我们换成大桌子)
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=UsdFileCfg(
                usd_path=f"{ASSETS_DIR}/living_room_table/living_room_table.usd",
                scale=(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # [C] 杂货 (初始定义)
        # 注意：这里的 init_state 只是第一帧的位置，reset 后会被下面的 events 覆盖
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, -10.0, TABLE_HEIGHT + 0.5], 
                rot=[1, 0, 0, 0]
            ),
            spawn=MultiAssetSpawnerCfg(
                assets_cfg=[
                    UsdFileCfg(
                        usd_path=f"{ASSETS_DIR}/{name}/{name}.usd",
                        scale=(1.0, 1.0, 1.0),
                        rigid_props=RigidBodyPropertiesCfg(
                            disable_gravity=False, linear_damping=1.0, angular_damping=1.0
                        ),
                        mass_props=MassPropertiesCfg(mass=0.5), 
                    ) for name in OBJECT_NAMES
                ],
                random_choice=True, # 每次重置随机换一个物品模型
            ),
        )

        # =====================================================
        # 4. [新增] 随机化事件 (Events)
        # =====================================================
        # 覆盖官方的 reset_object_position，使用我们自定义的范围
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform, # 使用均匀分布随机重置位置
            mode="reset", # 仅在环境重置时触发
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "pose_range": {
                    # X轴: 左右各5米范围
                    "x": (-5.0, 5.0), 
                    
                    # Y轴: [-15, -2]
                    # 解释: 机器人(-20) < 物品 < 篮子(8)
                    # 这样物品就在机器人前方，且不会掉进篮子里
                    "y": (-15.0, -2.0), 
                    
                    # Z轴: 在桌面上方 0.5~1.0米处生成，自然掉落
                    "z": (TABLE_HEIGHT + 0.5, TABLE_HEIGHT + 1.0)
                },
                "velocity_range": {}, # 初速度为0
            },
        )

        # [C] 新增 Basket (官方 Lift 任务没有篮子)
        self.scene.basket = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            spawn=UsdFileCfg(
                usd_path=f"{ASSETS_DIR}/basket/basket.usd",
                scale=(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 8.0, TABLE_HEIGHT),
            ),
        )

##
# MDP settings
##
'''

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


##
# Environment configuration
##


@configclass
class TableCleanEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TableCleanSceneCfg = TableCleanSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

'''