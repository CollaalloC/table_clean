
# run_teleop_clean.py

import argparse
from isaaclab.app import AppLauncher

# 1. 参数解析 (必须在 AppLauncher 之前)
parser = argparse.ArgumentParser(description="Teleop Table Clean")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device: keyboard, spacemouse, gamepad")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 导入库 (必须在启动应用之后)
import gymnasium as gym
import torch
import logging
from collections.abc import Callable

import isaaclab.envs
from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from isaaclab.managers import TerminationTermCfg as DoneTerm

# 导入我们的环境配置
from old.table_clean_task.table_clean_env_ik import FrankaTableCleanIKEnvCfg

# 设置日志
logger = logging.getLogger(__name__)

def main():
    """Run teleoperation with the table clean environment."""
    
    # 4. 创建环境配置
    env_cfg = FrankaTableCleanIKEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 修改配置以适应遥操作
    # 禁用时间超时
    env_cfg.terminations.time_out = None
    # 确保命令不重采样
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

    # 5. 创建环境
    try:
        env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # 6. 遥操作控制标志
    should_reset = False
    teleop_active = True

    # 回调函数
    def reset_env():
        nonlocal should_reset
        should_reset = True
        print("Reset triggered")

    def toggle_teleop():
        nonlocal teleop_active
        teleop_active = not teleop_active
        print(f"Teleoperation active: {teleop_active}")

    teleop_callbacks = {
        "R": reset_env,
        "RESET": reset_env,
        "START": lambda: print("Start (No-op)"), # 占位
    }

    # 7. 创建遥操作设备
    teleop_interface = None
    sensitivity = args_cli.sensitivity
    
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
        print("Keyboard Teleop Active: Use W/A/S/D/Q/E to move, K to grab, R to reset")
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
        print("SpaceMouse Teleop Active")
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
        )
    else:
        logger.error(f"Unsupported device: {args_cli.teleop_device}")
        env.close()
        simulation_app.close()
        return

    # 添加回调
    for key, callback in teleop_callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except Exception as e:
            logger.warning(f"Failed to add callback {key}: {e}")

    # 初始重置
    env.reset()
    teleop_interface.reset()

    print(f"Teleoperation started. Device: {teleop_interface}")

    # 8. 仿真循环
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # 获取设备指令
                # advance() 返回的是动作张量，通常是 [x, y, z, roll, pitch, yaw, gripper]
                action = teleop_interface.advance()
                
                if teleop_active:
                    # 广播动作到所有环境
                    # 注意：action 是 (1, action_dim)，需要 repeat 到 (num_envs, action_dim)
                    # 如果 action 已经是 (action_dim,)，则需要 unsqueeze
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                    
                    actions = action.repeat(env.unwrapped.num_envs, 1)
                    
                    # 执行动作
                    env.step(actions)
                else:
                    # 如果暂停遥操作，只渲染
                    # env.step(torch.zeros_like(actions)) # 或者发送零动作
                    # 为了保持物理模拟运行，最好还是 step，但发送零动作或者保持当前位置
                    # 这里简单起见，继续 step 零动作
                    zero_actions = torch.zeros_like(action).repeat(env.unwrapped.num_envs, 1)
                    # 保持夹爪状态可能更好，但这里先发0
                    env.step(zero_actions)

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
        
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            break
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            break

    # 清理
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    main()
    simulation_app.close()