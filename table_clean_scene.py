# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import random
import math

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Table Clean Scenario - Random Init via Config.")
#parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# 这里的 import 必须在 simulation_app 启动之后
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

# ---------------------------------------------------------
#  配置区域
# ---------------------------------------------------------
# 请确保此路径指向包含 alphabet_soup 等文件夹的父级目录
ASSETS_DATA_DIR = "/home/user/Downloads/FCloud OmniBot赛事指定文件/table_clean/assets"

OBJECT_NAMES = [
    "alphabet_soup", "basket", "butter", "cream_cheese", 
    "ketchup", "milk", "orange_juice", "tomato_sauce"
]

# 桌面参数设置 (根据你的 usd 模型实际尺寸调整)
# 假设桌子中心在 (0,0)，以下是桌面的一半长宽
TABLE_BOUNDS_X = 0.3  # 桌面 X 轴范围 (-0.3 到 0.3)
TABLE_BOUNDS_Y = 0.3  # 桌面 Y 轴范围 (-0.3 到 0.3)
TABLE_HEIGHT = 0.45   # 桌面的大致高度 (米)
DROP_HEIGHT = 0.1     # 物体生成在桌面以上多少米 (防止穿模)

def get_random_orientation():
    """生成一个绕 Z 轴随机旋转的四元数 (w, x, y, z)"""
    # 随机选择一个角度 (-pi 到 pi)
    angle = random.uniform(-math.pi, math.pi)
    # 欧拉角转四元数公式 (仅绕 Z 轴)
    # w = cos(angle/2), z = sin(angle/2), x=0, y=0
    w = math.cos(angle / 2)
    z = math.sin(angle / 2)
    return (w, 0.0, 0.0, z)

def design_scene():
    """Designs the scene by spawning assets with random initial positions."""
    
    # 1. 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 2. 光照
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))

    # 3. 创建物品父节点 Xform
    prim_utils.create_prim("/World/Objects", "Xform")

    # 4. 生成桌子 (位置固定)
    table_usd_path = os.path.join(ASSETS_DATA_DIR, "living_room_table", "living_room_table.usd")
    if os.path.exists(table_usd_path):
        cfg_table = sim_utils.UsdFileCfg(usd_path=table_usd_path, scale=(150.0,150.0, 150.0))
        # 桌子放在原点
        cfg_table.func("/World/Table", cfg_table, translation=(0.0, 0.0, 0.0))
    else:
        print(f"[ERROR] Table USD not found: {table_usd_path}")

    # 5. 生成物品 (位置随机)
    # 定义通用的刚体属性
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    mass_props = sim_utils.MassPropertiesCfg(mass=0.5)

    for i, name in enumerate(OBJECT_NAMES):
        usd_path = os.path.join(ASSETS_DATA_DIR, name, f"{name}.usd")
        
        # 检查文件是否存在
        if not os.path.exists(usd_path):
            print(f"[WARNING] Object USD not found: {usd_path}")
            continue

        # 配置物品
        cfg_obj = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1.0, 1.0, 1.0),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props
        )

        # --- 核心逻辑：在这里计算随机位置 ---
        
        # 1. 随机 X, Y 坐标 (在桌面范围内)
        rand_x = random.uniform(-TABLE_BOUNDS_X, TABLE_BOUNDS_X)
        rand_y = random.uniform(-TABLE_BOUNDS_Y, TABLE_BOUNDS_Y)
        
        # 2. Z 坐标 (桌面高度 + 悬空高度)
        # 为了避免所有物体生成在同一个高度互相重叠，我们稍微错开一点点高度
        # 或者仅仅依赖物理引擎弹开它们。这里加上 i*0.05 稍微错落排列
        spawn_z = TABLE_HEIGHT + DROP_HEIGHT + (i * 0.05)
        
        # 3. 随机旋转
        rand_rot = get_random_orientation()

        # 4. 执行生成 (func)
        # 这里的 translation 和 orientation 参数决定了初始状态
        prim_path = f"/World/Objects/{name}"
        cfg_obj.func(
            prim_path, 
            cfg_obj, 
            translation=(rand_x, rand_y, spawn_z), 
            orientation=rand_rot
        )
        print(f"[INFO] Spawning {name} at ({rand_x:.2f}, {rand_y:.2f})")

def main():
    """Main function."""
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 设置相机视角
    sim.set_camera_view([1.2, 1.2, 1.2], [0.0, 0.0, 0.4])

    # 构建场景 (随机化在这里发生)
    design_scene()

    # 重置仿真 (物理引擎准备就绪)
    sim.reset()
    print("[INFO]: Setup complete. Objects placed randomly.")

    # 仿真循环
    while simulation_app.is_running():
        # 执行物理步进
        # 这里不需要任何手动的位置重置代码，物体会遵循物理定律掉落并静止
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()