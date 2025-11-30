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
# 场景缩放倍数
SCENE_SCALE = 150.0
# 桌子高度
TABLE_HEIGHT = 45.0
# 机器人末端原始偏移量 (米)
ROBOT_EE_OFFSET_ORIGINAL = 0.1034
##
# Scene definition
##


@configclass
class TableCleanIKRelEnvCfg(OfficialFrankaIKCfg):
    """
    自定义环境配置：Table Clean (IK Control)
    """
    def __post_init__(self):
        # 1. 运行父类初始化
        super().__post_init__()

        # =====================================================
        # 2. 机器人设置 (位置 & 尺寸 & 动力学修正)
        # =====================================================
        
        # [A] 位置：远离篮子
        self.scene.robot.init_state.pos = (0.0, -20.0, TABLE_HEIGHT)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        
        # [B] 尺寸：放大 150 倍
        self.scene.robot.spawn.scale = (SCENE_SCALE, SCENE_SCALE, SCENE_SCALE)

        # [C] 动力学修正 (重要！)
        # 放大150倍后，机器人太重了，默认电机推不动。
        # 我们强制关闭重力，让它像在太空中一样，这样IK才能控制得动。
        self.scene.robot.spawn.rigid_props.disable_gravity = True 

        # =====================================================
        # 3. 修正 IK 控制器与传感器 (适配放大)
        # =====================================================
        scaled_offset = ROBOT_EE_OFFSET_ORIGINAL * SCENE_SCALE

        # [A] 修正动作空间 (Actions)
        self.actions.arm_action.body_offset.pos = [0.0, 0.0, scaled_offset]
        
        # [关键修改] 放大动作灵敏度！
        # 以前动一下是 0.5米，现在需要是 0.5 * 150 = 75米，否则看不出在动
        self.actions.arm_action.scale = 0.5 * SCENE_SCALE

        # [B] 修正传感器
        self.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, scaled_offset]

        # =====================================================
        # 4. 覆盖场景物体 (Table, Basket)
        # =====================================================
        
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=UsdFileCfg(
                usd_path=f"{ASSETS_DIR}/living_room_table/living_room_table.usd",
                scale=(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

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

        # =====================================================
        # 5. 生成两个随机物品 (Object A & Object B)
        # =====================================================
        # 定义通用的生成器配置
        common_spawner = MultiAssetSpawnerCfg(
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
            random_choice=True, # 随机选一个
        )

        # 物品 A
        self.scene.object_a = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_A",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -10.0, TABLE_HEIGHT + 0.5]),
            spawn=common_spawner, # 复用配置
        )

        # 物品 B
        self.scene.object_b = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_B",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[5.0, -10.0, TABLE_HEIGHT + 0.5]),
            spawn=common_spawner, # 复用配置
        )

        # [重要] 移除原本的 self.scene.object，防止报错或混淆
        if hasattr(self.scene, "object"):
            del self.scene.object

        # =====================================================
        # 6. 随机化事件 (适配双物品)
        # =====================================================
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # 使用正则匹配 object_a 和 object_b
                "asset_cfg": SceneEntityCfg("object_.*"), 
                "pose_range": {
                    "x": (-5.0, 5.0), 
                    "y": (-15.0, -2.0),
                    "z": (TABLE_HEIGHT + 0.5, TABLE_HEIGHT + 1.0)
                },
                "velocity_range": {},
            },
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