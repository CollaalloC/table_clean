# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import random
import math
import numpy as np

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Table Clean - Final Scene")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaacsim.core.utils.bounds import compute_combined_aabb 

# ---------------------------------------------------------
#  配置区域
# ---------------------------------------------------------
# 自动获取当前脚本目录下的 assets 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DATA_DIR = os.path.join(SCRIPT_DIR, "assets")

# 移除篮子，因为它单独处理
OBJECT_NAMES = [
    "alphabet_soup", "butter", "cream_cheese", 
    "ketchup", "milk", "orange_juice", "tomato_sauce"
]

# 场景缩放倍数 (桌子和篮子)
SCENE_SCALE = 150.0 

def get_upright_orientation():
    """
    生成直立状态的随机旋转 (只绕 Z 轴旋转)
    这样物品就是底面朝下，不会倒
    """
    # 随机选择一个偏航角 (Yaw)
    yaw = random.uniform(-math.pi, math.pi)
    
    # 欧拉角转四元数 (Roll=0, Pitch=0, Yaw=random)
    # q = [w, x, y, z]
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (w, 0.0, 0.0, z)

def design_scene_and_spawn():
    """构建场景"""
    
    # 1. 地面 & 光照
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))

    prim_utils.create_prim("/World/Objects", "Xform")

    # ==========================================
    # 2. 加载桌子 (缩放 150x)
    # ==========================================
    table_usd_path = os.path.join(ASSETS_DATA_DIR, "living_room_table", "living_room_table.usd")
    
    if not os.path.exists(table_usd_path):
        print(f"[ERROR] ❌ Table USD not found at: {table_usd_path}")
        return
    
    print(f"[INFO] Loading Table (Scale {SCENE_SCALE}x)...")
    cfg_table = sim_utils.UsdFileCfg(
        usd_path=table_usd_path,
        scale=(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE), 
    )
    cfg_table.func("/World/Table", cfg_table, translation=(0.0, 0.0, 0.0))

    # ==========================================
    # 3. 计算桌面几何信息
    # ==========================================
    try:
        raw_aabb = compute_combined_aabb(["/World/Table"])
    except Exception as e:
        print(f"[WARNING] Could not compute AABB: {e}")
        raw_aabb = [-0.75, -0.4, 0.0, 0.75, 0.4, 0.45]

    raw_width = raw_aabb[3] - raw_aabb[0]
    raw_depth = raw_aabb[4] - raw_aabb[1]
    raw_height = raw_aabb[5] 
    
    actual_scale_factor = 1.0
    if raw_height < 5.0: 
        actual_scale_factor = SCENE_SCALE
    
    real_width = raw_width * actual_scale_factor
    real_depth = raw_depth * actual_scale_factor
    real_surface_z = raw_aabb[5] * actual_scale_factor

    print("="*60)
    print(f"[INFO] Table Surface Z: {real_surface_z:.3f} m")
    print(f"[INFO] Table Size: {real_width:.2f} x {real_depth:.2f} m")
    print("="*60)

    # ==========================================
    # 4. 加载篮子 (偏向一侧)
    # ==========================================
    basket_path = os.path.join(ASSETS_DATA_DIR, "basket", "basket.usd")
    
    # 计算篮子位置：放在桌面的右侧 (Y轴正方向 1/4 处)
    # 假设桌子中心在 (0,0)
    basket_y_offset = real_depth * 0.25 
    basket_pos = (0.0, basket_y_offset, 45.0)

    if os.path.exists(basket_path):
        print(f"[INFO] Spawning Basket at offset Y={basket_y_offset:.2f}")
        cfg_basket = sim_utils.UsdFileCfg(
            usd_path=basket_path,
            scale=(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE),
        )
        # 篮子是静态的 (Static)，作为障碍物
        cfg_basket.func("/World/Objects/FixedBasket", cfg_basket, translation=basket_pos)
    else:
        print(f"[WARNING] Basket USD not found")

    # ==========================================
    # 5. 随机生成 2 个物品 (避免篮子区域)
    # ==========================================
    
    # [修改点] 只随机选取 2 个物品
    selected_items = random.sample(OBJECT_NAMES, 2)
    print(f"[INFO] Selected items: {selected_items}")

    # [修改点] 定义物品生成区域
    # 篮子在 Y > 0 的区域，所以我们限制物品在 Y < 0 的区域
    # 或者限制在 Y = [-0.4 * depth, 0.0] 之间
    spawn_area_y_min = -real_depth * 0.3
    spawn_area_y_max = 0.0  # 不超过中线，绝对安全，不会掉进篮子
    spawn_area_x_bound = real_width * 0.3 # X轴范围保持正常

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, 
        max_depenetration_velocity=5.0,
        # 增加一点线性阻尼，让物品掉落后不那么容易乱滚
        linear_damping=0.5, 
        angular_damping=0.5
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    mass_props = sim_utils.MassPropertiesCfg(mass=0.2)

    for i, name in enumerate(selected_items):
        usd_path = os.path.join(ASSETS_DATA_DIR, name, f"{name}.usd")
        if not os.path.exists(usd_path): continue

        # 1. 随机位置 (限制在安全区域)
        rand_x = random.uniform(-spawn_area_x_bound, spawn_area_x_bound)
        rand_y = random.uniform(spawn_area_y_min, spawn_area_y_max)
        
        # 2. 生成高度
        # 既然要求不倒，我们尽量放低一点，紧贴桌面生成
        # real_surface_z 是桌面，+0.05m (5cm) 稍微悬空一点点防止穿模即可
        # 不同的物品高度不同，+0.1m 比较通用
        spawn_z = real_surface_z + 0.1 
        
        # 3. 随机旋转 (直立)
        # 使用新写的 get_upright_orientation，只绕 Z 轴转
        rand_rot = get_upright_orientation()

        cfg_obj = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1.0, 1.0, 1.0),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props
        )
        
        prim_path = f"/World/Objects/{name}"
        cfg_obj.func(prim_path, cfg_obj, translation=(rand_x, rand_y, spawn_z), orientation=rand_rot)
        print(f"[INFO] Spawning {name} at ({rand_x:.2f}, {rand_y:.2f}) - Upright")

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 调整相机视角 (看清桌面布局)
    sim.set_camera_view([30.0, 160.0, 250.0], [0.0, 0.0, 65.0])

    design_scene_and_spawn()

    sim.reset()
    print("[INFO]: Simulation started.")

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()