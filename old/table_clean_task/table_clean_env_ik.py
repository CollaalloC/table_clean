# table_clean_env_ik.py

import os
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.multi_asset.multi_asset_spawner_cfg import MultiAssetSpawnerCfg
from isaaclab.utils import configclass

# 引入必要的 IK 控制器配置
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

# 预定义配置
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG # 注意：使用 HIGH_PD 版本，IK效果更好

# 路径配置 (请确保正确)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DATA_DIR = os.path.join(SCRIPT_DIR, "assets")

OBJECT_NAMES = [
    "alphabet_soup", "butter", "cream_cheese", 
    "ketchup", "milk", "orange_juice", "tomato_sauce"
]

@configclass
class FrankaTableCleanIKEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. 机器人：使用 High PD 配置，这对于 IK 跟踪非常重要
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 调整机器人位置到桌面上 (假设桌子表面在 Z=45.0)
        # 机器人放在 (0, -0.6) 处，面向 +Y 方向
        self.scene.robot.init_state.pos = (0.0, -0.6, 45.0) 
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        # --------------------------------------------------------
        # [核心修改] 动作空间改为 差分逆运动学 (Diff-IK)
        # --------------------------------------------------------
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand", # 末端执行器名称
            # 这里的 controller 定义了 IK 的解算方式
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=True, 
                ik_method="dls"
            ),
            scale=0.5, # 灵敏度
            # 末端执行器的偏移量 (Panda 手掌中心)
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # 夹爪动作保持不变
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        self.commands.object_pose.body_name = "panda_hand"
        # 禁用 object_pose 命令的重采样，防止目标点乱跳
        self.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

        # --------------------------------------------------------
        # 场景物体 (保持你之前的设置)
        # --------------------------------------------------------
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=UsdFileCfg(
                usd_path=os.path.join(ASSETS_DATA_DIR, "living_room_table", "living_room_table.usd"),
                scale=(150.0, 150.0, 150.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        self.scene.basket = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            spawn=UsdFileCfg(
                usd_path=os.path.join(ASSETS_DATA_DIR, "basket", "basket.usd"),
                scale=(150.0, 150.0, 150.0),
            ),
            # 篮子放在机器人前方可达区域
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.3, 45.0)),
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            # 物体放在机器人和篮子之间
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -0.2, 45.05], rot=[1, 0, 0, 0]),
            spawn=MultiAssetSpawnerCfg(
                assets=[
                    UsdFileCfg(
                        usd_path=os.path.join(ASSETS_DATA_DIR, name, f"{name}.usd"),
                        scale=(1.0, 1.0, 1.0),
                        rigid_props=RigidBodyPropertiesCfg(
                            disable_gravity=False, linear_damping=0.5, angular_damping=0.5,
                        ),
                    ) for name in OBJECT_NAMES
                ],
                random_choice=True,
            ),
        )

        # 传感器配置
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers[""].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )