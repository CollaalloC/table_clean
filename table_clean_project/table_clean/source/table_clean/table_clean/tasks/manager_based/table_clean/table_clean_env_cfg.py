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
    自定义环境配置：Table Clean (IK Control, 双物品)
    """
    def __post_init__(self):
        # 1. 运行父类初始化
        super().__post_init__()

        # =====================================================
        # 2. 机器人设置
        # =====================================================
        self.scene.robot.init_state.pos = (0.0, -20.0, TABLE_HEIGHT)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.spawn.scale = (SCENE_SCALE, SCENE_SCALE, SCENE_SCALE)
        self.scene.robot.spawn.rigid_props.disable_gravity = True 

        # =====================================================
        # 3. 修正 IK 控制器与传感器
        # =====================================================
        scaled_offset = ROBOT_EE_OFFSET_ORIGINAL * SCENE_SCALE
        self.actions.arm_action.body_offset.pos = [0.0, 0.0, scaled_offset]
        # 放大动作灵敏度，确保键盘控制能动
        self.actions.arm_action.scale = 0.5 * SCENE_SCALE
        self.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, scaled_offset]

        # =====================================================
        # 4. 覆盖场景物体
        # =====================================================
        
        # [A] 桌子
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=UsdFileCfg(
                usd_path=f"{ASSETS_DIR}/living_room_table/living_room_table.usd",
                scale=(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # [B] 定义通用的杂货生成器
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

        # [关键策略]
        # 1. 这里的名字叫 "object"，直接覆盖父类的定义。
        # 这样父类所有的奖励函数(reaching_object等)都会自动追踪这个物体，不会报错。
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -10.0, TABLE_HEIGHT + 0.5]),
            spawn=common_spawner,
        )

        # 2. 这是第二个物体，命名为 "object_2"
        self.scene.object_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[5.0, -10.0, TABLE_HEIGHT + 0.5]),
            spawn=common_spawner,
        )

        # [C] 篮子
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
        # 5. [修复] 随机化事件 (拆分为两个独立事件)
        # =====================================================
        
        # 随机化物体 1 (Object)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),  # 明确指定 object
                "pose_range": {
                    "x": (-5.0, 5.0), 
                    "y": (-15.0, -2.0),
                    "z": (TABLE_HEIGHT + 0.5, TABLE_HEIGHT + 1.0)
                },
                "velocity_range": {},
            },
        )

        # 随机化物体 2 (Object 2)
        self.events.reset_object_2_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object_2"), # 明确指定 object_2
                "pose_range": {
                    "x": (-5.0, 5.0), 
                    "y": (-15.0, -2.0),
                    "z": (TABLE_HEIGHT + 0.5, TABLE_HEIGHT + 1.0)
                },
                "velocity_range": {},
            },
        )

        # =====================================================
        # 6. 补充观测
        # =====================================================
        self.observations.policy.object_2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_2")}
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