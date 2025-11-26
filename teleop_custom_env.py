import os
import argparse
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# ---------------------------------------------------------
# 1. 启动参数设置
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Custom Table Clean Teleop")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device: keyboard only")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. 导入必要的 Isaac Lab 模块
# ---------------------------------------------------------
import isaaclab.envs
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.multi_asset.multi_asset_spawner_cfg import MultiAssetSpawnerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

# 导入官方的 Lift 环境配置作为基类
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# 导入 IK 控制器配置 (这是实现遥控的关键)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# 导入键盘接口
# 尝试直接从子模块导入，以避开可能存在的 __init__.py 问题
try:
    from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard, Se3KeyboardCfg
except ImportError:
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

# ---------------------------------------------------------
# 3. 定义你的“魔改”配置类
# ---------------------------------------------------------

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DATA_DIR = os.path.join(SCRIPT_DIR, "assets")
OBJECT_NAMES = ["alphabet_soup", "butter", "cream_cheese", "ketchup", "milk", "orange_juice", "tomato_sauce"]

@configclass
class CustomTableCleanIKEnvCfg(LiftEnvCfg):
    """
    继承自官方 LiftEnvCfg，但修改了场景物体和动作空间
    """
    def __post_init__(self):
        super().__post_init__()

        # --- A. 修改机器人 (抬高到桌面上方) ---
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 调整机器人位置到桌面上 (假设桌子表面在 Z=45.0)
        # 机器人放在 (0, -0.6) 处，面向 +Y 方向
        self.scene.robot.init_state.pos = (0.0, -0.6, 45.0) 
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        # --- B. 修改动作空间 (启用 IK 以支持键盘控制) ---
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        # 夹爪动作
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.commands.object_pose.body_name = "panda_hand"
        # 禁用 object_pose 命令的重采样
        self.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

        # --- C. 替换场景物体 (你的自定义资产) ---
        
        # 1. 大桌子
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=UsdFileCfg(
                usd_path=os.path.join(ASSETS_DATA_DIR, "living_room_table", "living_room_table.usd"),
                scale=(150.0, 150.0, 150.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # 2. 篮子
        self.scene.basket = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            spawn=UsdFileCfg(
                usd_path=os.path.join(ASSETS_DATA_DIR, "basket", "basket.usd"),
                scale=(150.0, 150.0, 150.0),
            ),
            # 篮子放在机器人前方可达区域
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.3, 45.0)),
        )

        # 3. 随机杂货 (作为操作对象)
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
                random_choice=True, # 随机选一个
            ),
        )

        # --- D. 必要的传感器 ---
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

# ---------------------------------------------------------
# 4. 注册新环境 (Registration)
# ---------------------------------------------------------
# 这就是“魔改”生效的一步：告诉 Gym 有个新环境叫这个名字
gym.register(
    id="Isaac-Table-Clean-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": CustomTableCleanIKEnvCfg, # 指向我们上面定义的类
    },
)

# ---------------------------------------------------------
# 5. 主运行逻辑
# ---------------------------------------------------------
def main():
    print("[INFO] Creating custom environment: Isaac-Table-Clean-IK-v0")
    
    # 直接使用刚刚注册的 ID 创建环境
    env = gym.make("Isaac-Table-Clean-IK-v0", render_mode="rgb_array")

    # 创建键盘控制器
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05)
        )
        print("Keyboard Active: W/A/S/D/Q/E + K")
    else:
        raise ValueError("Only keyboard supported for this demo")

    env.reset()
    teleop_interface.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # 获取指令
            delta_pose = teleop_interface.advance()
            # 广播并发送
            actions = delta_pose.repeat(env.unwrapped.num_envs, 1)
            env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
