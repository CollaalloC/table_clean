
# run_teleop_clean.py

import argparse
from isaaclab.app import AppLauncher

# 1. 启动应用
parser = argparse.ArgumentParser(description="Teleop Table Clean")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device: keyboard, spacemouse, gamepad")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入库
import gymnasium as gym
import torch
import isaaclab.envs # 注册 Isaac 环境

# 导入我们的设备接口
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg

# 导入刚才写的配置
from table_clean_env_ik import FrankaTableCleanIKEnvCfg

def main():
    # 3. 创建环境
    # 直接传入配置类，而不是任务名称字符串
    env_cfg = FrankaTableCleanIKEnvCfg()
    env_cfg.scene.num_envs = 1 # 遥控只控制一个环境
    
    # Gymnasium 实例化
    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg) # 名字只是占位符，重要的是 cfg

    # 4. 创建遥控设备
    if args_cli.teleop_device.lower() == "keyboard":
        # 键盘灵敏度配置
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05)
        )
        print("Keyboard Teleop Active: Use W/A/S/D/Q/E to move, K to grab")
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05, rot_sensitivity=0.05))
    else:
        raise ValueError("Unsupported device")

    # 重置
    env.reset()
    teleop_interface.reset()

    # 5. 仿真循环
    while simulation_app.is_running():
        with torch.inference_mode():
            # A. 获取设备指令 [x, y, z, roll, pitch, yaw, gripper]
            delta_pose = teleop_interface.advance()
            
            # B. 广播到环境 (虽然只有1个环境，但格式需要匹配)
            actions = delta_pose.repeat(env.unwrapped.num_envs, 1)
            
            # C. 发送动作
            # 这里的 actions 会被 DifferetialIKAction 处理，转化为关节运动
            env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()