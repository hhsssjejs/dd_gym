from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .d2.d2_config import D2DHStandCfg, D2DHStandCfgPPO

from .d2.d2_env import D2DHStandEnv

from humanoid.utils.task_registry import task_registry

task_registry.register("d2", D2DHStandEnv, D2DHStandCfg(), D2DHStandCfgPPO())
