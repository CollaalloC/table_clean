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
parser = argparse.ArgumentParser(description="Table Clean - Manual Scale Fix")
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
ASSETS_DATA_DIR = "/home/user/Downloads/FCloud OmniBot赛事指定文件/table_clean/assets"

OBJECT_NAMES = [
    "alphabet_soup", "basket", "butter", "cream_cheese", 
    "ketchup", "milk", "orange_juice", "tomato_sauce"
]

# [关键修改] 定义统一的缩放倍数
TABLE_SCALE = 150.0 

def get_random_orientation():
    angle = random.uniform(-math.pi, math.pi)
    w = math.cos(angle / 2)
    z = math.sin(angle / 2)
    return (w, 0.0, 0.0, z)

def design_scene_and_spawn():
    """构建场景、测量桌子并随机放置物品"""
    
    # 1. 地面 & 光照
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))

    prim_utils.create_prim("/World/Objects", "Xform")

    # 2. 加载桌子 (应用缩放)
    table_usd_path = os.path.join(ASSETS_DATA_DIR, "living_room_table", "living_room_table.usd")
    
    if not os.path.exists(table_usd_path):
        print(f"[ERROR] ❌ Table USD not found at: {table_usd_path}")
        return
    
    print(f"[INFO] Loading Table with Scale {TABLE_SCALE}x")
    cfg_table = sim_utils.UsdFileCfg(
        usd_path=table_usd_path,
        scale=(TABLE_SCALE, TABLE_SCALE, TABLE_SCALE), 
    )
    cfg_table.func("/World/Table", cfg_table, translation=(0.0, 0.0, 0.0))

    # 3. 计算桌子尺寸 (AABB)
    try:
        # 获取原始边界（此时可能还未包含缩放信息）
        raw_aabb = compute_combined_aabb(["/World/Table"])
    except Exception as e:
        print(f"[WARNING] Could not compute AABB: {e}")
        # 给一个兜底的默认值 (基于你之前的 log)
        raw_aabb = [-0.75, -0.4, 0.0, 0.75, 0.4, 0.45]

    # [关键逻辑] 强制手动应用缩放计算实际尺寸
    # 我们假设 raw_aabb 返回的是未缩放的尺寸，所以我们要手动乘 TABLE_SCALE
    
    raw_width = raw_aabb[3] - raw_aabb[0]
    raw_depth = raw_aabb[4] - raw_aabb[1]
    raw_height = raw_aabb[5] # 假设底部在 0
    
    # 计算这一步非常重要：
    # 如果 AABB 已经很大(比如 >10米)，说明 IsaacSim 已经应用了缩放，我们就不乘了。
    # 如果 AABB 很小(比如 <2米)，说明这是原始尺寸，我们需要乘 150。
    
    actual_scale_factor = 1.0
    if raw_height < 5.0: # 阈值判断：如果高度小于5米，肯定还没乘 150
        print("[INFO] AABB seems unscaled. Applying manual scale factor.")
        actual_scale_factor = TABLE_SCALE
    
    real_width = raw_width * actual_scale_factor
    real_depth = raw_depth * actual_scale_factor
    real_surface_z = raw_aabb[5] * actual_scale_factor # 最高点 Z * 缩放

    print("="*60)
    print(f"[INFO] Table Dimensions Correction:")
    print(f"       Raw Measured Height: {raw_aabb[5]:.3f} m")
    print(f"       Applied Scale:       {actual_scale_factor} x")
    print(f"       -> REAL Surface Z:   {real_surface_z:.3f} m")
    print("="*60)

    # 4. 基于修正后的尺寸生成物品
    safe_margin = 0.8 
    
    # 假设桌子中心在 (0,0)
    bounds_x = (real_width / 2) * safe_margin
    bounds_y = (real_depth / 2) * safe_margin

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    mass_props = sim_utils.MassPropertiesCfg(mass=0.5)

    for i, name in enumerate(OBJECT_NAMES):
        usd_path = os.path.join(ASSETS_DATA_DIR, name, f"{name}.usd")
        if not os.path.exists(usd_path): continue

        # 随机位置 (在巨大的桌面上分布)
        rand_x = random.uniform(-bounds_x, bounds_x)
        rand_y = random.uniform(-bounds_y, bounds_y)
        
        # 高度：在修正后的桌面高度 (约 67.5米) 上方生成
        # 加上一点点偏移量 (0.2m) 加上索引偏移，防止重叠
        spawn_z = real_surface_z + 0.2 + (i * 0.1)
        
        rand_rot = get_random_orientation()

        # 生成物品
        cfg_obj = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1.0, 1.0, 1.0), # 物品保持原样，不随桌子放大
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props
        )
        
        prim_path = f"/World/Objects/{name}"
        cfg_obj.func(prim_path, cfg_obj, translation=(rand_x, rand_y, spawn_z), orientation=rand_rot)
        print(f"[INFO] Spawning {name} at Z={spawn_z:.2f}")

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # [修改] 相机需要拉得非常远才能看到 67米高的桌子全貌
    # 比如放到 Z=80米 的位置
    sim.set_camera_view([20.0, 20.0, 80.0], [0.0, 0.0, 60.0])

    design_scene_and_spawn()

    sim.reset()
    print("[INFO]: Simulation started.")

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()