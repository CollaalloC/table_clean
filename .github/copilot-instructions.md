# GitHub Copilot Instructions for Isaac Lab Projects

## Project Context
This workspace contains a robotics simulation project built on **NVIDIA Isaac Lab** (formerly Orbit).
- **Core Framework**: Isaac Lab (based on Isaac Sim).
- **Primary Language**: Python 3.10+.
- **Key Libraries**: `isaaclab`, `torch`, `gymnasium`.
- **Project Type**: Robotic manipulation (Franka Emika Panda), Sim-to-Real, Reinforcement Learning.

## Architecture & Patterns

### 1. Environment Configuration (`*EnvCfg`)
- Environments are defined using configuration classes decorated with `@configclass`.
- Inherit from `ManagerBasedRLEnvCfg` or specific task configs (e.g., `LiftEnvCfg`).
- **Pattern**: Separate configuration for Scene, Observations, Actions, Events, and Rewards.
- **Example**:
  ```python
  @configclass
  class MyEnvCfg(ManagerBasedRLEnvCfg):
      scene: MySceneCfg = MySceneCfg()
      observations: MyObservationsCfg = MyObservationsCfg()
      actions: MyActionsCfg = MyActionsCfg()
  ```

### 2. Scene Definition
- Use `InteractiveSceneCfg` to define the simulation world.
- Assets are added as properties (e.g., `RigidObjectCfg`, `ArticulationCfg`).
- **Assets**: Use `UsdFileCfg` to load USD files.
- **Example**:
  ```python
  @configclass
  class MySceneCfg(InteractiveSceneCfg):
      robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
      table: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", ...)
  ```

### 3. Simulation App Launching
- **CRITICAL**: `AppLauncher` must be initialized **before** importing `isaaclab` modules that depend on the simulator (like `gymnasium` wrappers or `torch` devices).
- **Pattern**:
  ```python
  from isaaclab.app import AppLauncher
  # ... argparse setup ...
  app_launcher = AppLauncher(args)
  simulation_app = app_launcher.app
  # NOW import other modules
  import isaaclab.envs
  import torch
  ```

## Development Workflow

### 1. Running Scripts
- Use the python interpreter from the Isaac Lab environment.
- **Command**: `python path/to/script.py --num_envs 1` (or other args).
- **Teleoperation**: `python run_teleop_clean.py --teleop_device keyboard`.

### 2. Extension Management
- The project uses a template structure (`table_clean_project`).
- **Install**: `python -m pip install -e source/table_clean` (inside `table_clean_project`).
- **Verify**: `python scripts/list_envs.py`.

### 3. Tensor Operations
- Isaac Lab uses PyTorch tensors on the GPU (`device="cuda:0"`).
- **Shape Convention**: `(num_envs, ...)`
- Avoid moving tensors to CPU unless necessary for logging/saving.

## Common Pitfalls & Best Practices
- **USD Paths**: Ensure paths to `.usd` files are correct. Use absolute paths or relative to a known root.
- **Coordinate Systems**: Isaac Sim uses **Z-up** by default (standard for robotics), but check specific asset orientation.
- **Task Management**: Use `TaskManagers` for modular logic (Rewards, Terminations, Curriculum).
- **Omitted Lines**: When editing, never remove `...` markers if they preserve context.
