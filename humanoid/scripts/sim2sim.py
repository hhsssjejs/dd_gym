import math
import numpy as np
# import mujoco
# import mujoco.viewer as viewer
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
# from humanoid.envs import XBotLCfg
from humanoid.envs import *
from humanoid.utils import Logger
import torch
import pygame
from threading import Thread
from humanoid.utils.helpers import get_load_path
import os
import time

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1, -0.2, -0.5

joystick_use = True
joystick_opened = False

if joystick_use:

    pygame.init()

    try:
        # 获取手柄
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")

    # 用于控制线程退出的标志
    exit_flag = False


    # 处理手柄输入的线程
    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd

        while not exit_flag:
            # 获取手柄输入
            pygame.event.get()

            # 更新机器人命令
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 0.8
            yaw_vel_cmd = -joystick.get_axis(3) * 1.0

            x_vel_cmd = np.clip(x_vel_cmd, -1, 1)
            # 等待一小段时间，可以根据实际情况调整
            pygame.time.delay(100)

        # 启动线程


    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('body-orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('body-angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)

        if 'ankle_roll' or '6_link' in body_name:  # 根据你的模型具体命名选择
            # print("index",i,"  body_name", body_name)
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double))

        if env_cfg.asset.name == 'x1':
            if 'x1-body' in body_name:  # 根据你的模型具体命名选择
                base_pos = data.xpos[i][:3].copy().astype(np.double)
        else:
            if 'pelvis' or 'base_link' in body_name:  # 根据你的模型具体命名选择
                base_pos = data.xpos[i][:3].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    # print("target_q", target_q)
    # print("q", q)
    # print("kp", kp)
    # print("target_dq", target_dq)
    # print("dq", dq)
    # print("kd", kd)
    torque_out = (target_q + (cfg.robot_config.default_dof_pos) - q) * kp - dq * kd
    return torque_out


def run_mujoco(policy, cfg, env_cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    print("Load mujoco xml from:", cfg.sim_config.mujoco_model_path)
    # 从XML配置文件中创建MuJoCo模型对象。这通常包含了仿真的所有物理特性和对象定义。
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)

    # 设置模型的时间步长，这个时间步长决定了物理仿真的精度和速度。
    model.opt.timestep = cfg.sim_config.dt

    # 使用模型创建仿真数据对象，这个对象用于存储每一步仿真的状态数据。
    data = mujoco.MjData(model)

    # viewer.launch_passive(model, data)

    num_actuated_joints = env_cfg.env.num_actions  # This should match the number of actuated joints in your model
    # data.qpos[1] = -2.0
    # data.qpos[0] = -2.0
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    # data.qpos[2] = 0.1
    # 执行一步物理仿真。这会根据当前的状态和模型定义来更新仿真数据（例如，物体的位置和速度）。
    mujoco.mj_step(model, data)

    # 创建一个视图器对象，用于可视化仿真。这使得用户可以看到仿真环境和其中的物体如何随时间演变。
    viewer = mujoco_viewer.MujocoViewer(model, data)

    viewer.cam.azimuth = 90    # 方位角（左右旋转）
    viewer.cam.elevation = -10  # 俯仰角（上下旋转）
    viewer.cam.distance = 4   # 观察距离
    viewer.cam.lookat[:] = [0, 0, 1]  # 观察目标点 (x, y, z)

    # 初始化目标关节角度数组，这里全部设置为零。这个数组用于定义期望的控制目标状态。
    target_q = np.zeros((env_cfg.env.num_actions), dtype=np.double)

    # 初始化动作数组，同样全部设置为零。在强化学习或控制任务中，这个数组将被用来存储每一步的控制命令。
    action = np.zeros((env_cfg.env.num_actions), dtype=np.double)

    # 初始化一个历史观测的双端队列，用于存储一定数量的过去观测数据。这对于需要考虑时间序列信息的任务特别有用。
    hist_obs = deque()
    for _ in range(env_cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, env_cfg.env.num_single_obs], dtype=np.double))

    # 初始化一个计数器，用于跟踪低级控制循环的迭代次数。
    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)

    stop_state_log = 12000

    
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
        # time1 = time.time()
        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)
        q = (q[-env_cfg.env.num_actions:])
        dq = (dq[-env_cfg.env.num_actions:])

        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces


        if count_lowlevel % cfg.sim_config.decimation == 0:
            ###### for stand only #######
            if hasattr(env_cfg.commands, "sw_switch"):
                vel_norm = np.sqrt(x_vel_cmd ** 2 + y_vel_cmd ** 2 + yaw_vel_cmd ** 2)
                if env_cfg.commands.sw_switch and vel_norm <= env_cfg.commands.stand_com_threshold:
                    count_lowlevel = 0

            obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            # eu_ang[2] = 0
            eu_ang[1] -= 0.1

            # if not start_flag or end_flag:
            #     count_lowlevel = 0.0
            if env_cfg.env.num_commands == 5:
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / env_cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / env_cfg.rewards.cycle_time)
                obs[0, 2] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            if env_cfg.env.num_commands == 3:
                obs[0, 0] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            obs[0, env_cfg.env.num_commands:env_cfg.env.num_commands + env_cfg.env.num_actions] = (q - (cfg.robot_config.default_dof_pos)) * env_cfg.normalization.obs_scales.dof_pos
            obs[0, env_cfg.env.num_commands + env_cfg.env.num_actions:env_cfg.env.num_commands + 2 * env_cfg.env.num_actions] = dq * env_cfg.normalization.obs_scales.dof_vel
            obs[0, env_cfg.env.num_commands + 2 * env_cfg.env.num_actions:env_cfg.env.num_commands + 3 * env_cfg.env.num_actions] = action
            obs[0, env_cfg.env.num_commands + 3 * env_cfg.env.num_actions:env_cfg.env.num_commands + 3 * env_cfg.env.num_actions + 3] = omega
            obs[0, env_cfg.env.num_commands + 3 * env_cfg.env.num_actions + 3:env_cfg.env.num_commands + 3 * env_cfg.env.num_actions + 6] = eu_ang

            ####### for stand only #######
            if env_cfg.env.add_stand_bool:
                vel_norm = np.sqrt(x_vel_cmd ** 2 + y_vel_cmd ** 2 + yaw_vel_cmd ** 2)
                stand_command = (vel_norm <= env_cfg.commands.stand_com_threshold)
                obs[0, -1] = stand_command

            # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

            obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
            for i in range(env_cfg.env.frame_stack):
                policy_input[0, i * env_cfg.env.num_single_obs: (i + 1) * env_cfg.env.num_single_obs] = hist_obs[i][0, :]

            # arr = policy_input[0]
            # arr_reshaped = arr.reshape((15, 47))

            # # 遍历这个二维数组，打印每一行
            # for row in arr_reshaped:
            #     print(row)

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # print("\n", action)
            action = np.clip(action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)

            target_q = action * env_cfg.control.action_scale


        # time3 = time.time()

        target_dq = np.zeros((env_cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = (tau)
        # data.ctrl = tau
        applied_tau = data.actuator_force

        # print("\ntau", tau)
        # print("q", q)

        mujoco.mj_step(model, data)
        # time4 = time.time()
        viewer.render()
        # time5 = time.time()
        count_lowlevel += 1
        idx = 5
        dof_pos_target = target_q + (cfg.robot_config.default_dof_pos)
        if _ < stop_state_log:
            dict = {
                'base_height': base_z,
                'foot_z_l': foot_z[0] - env_cfg.rewards.feet_to_ankle_distance,
                'foot_z_r': foot_z[1] - env_cfg.rewards.feet_to_ankle_distance,
                'foot_forcez_l': foot_force_z[0],
                'foot_forcez_r': foot_force_z[1],
                'base_vel_x': v[0],
                'command_x': x_vel_cmd,
                'base_vel_y': v[1],
                'command_y': y_vel_cmd,
                'base_vel_z': v[2],
                'base_vel_yaw': omega[2],
                'command_yaw': yaw_vel_cmd,
                'dof_pos_target': dof_pos_target[idx],
                'dof_pos': q[idx],
                'dof_vel': dq[idx],
                'dof_torque': applied_tau[idx],
                'cmd_dof_torque': tau[idx],
            }


            # 添加 dof_pos_target 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos_target_{i}'] = dof_pos_target[i].item()

            # 添加 dof_pos 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos_{i}'] = q[i].item()

            # 添加 dof_torque 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_torque_{i}'] = applied_tau[i].item()

            # 添加 dof_vel 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_vel_{i}'] = dq[i].item()
            logger.log_states(dict=dict)

        elif _ == stop_state_log:
            logger.plot_states()
            # logger.save_log_to_mat(filename = "/home/pi/paper/d11/AIM/data_noarm.mat")
            pass

    viewer.close()


if __name__ == '__main__':
    import argparse

    # 创建ArgumentParser对象，用于处理命令行参数
    parser = argparse.ArgumentParser(description='Deployment script.')
    # 添加命令行参数 --load_model，这是一个必须提供的字符串参数，用于指定要加载的模型路径
    parser.add_argument('--load_model', type=str,
                        help='Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.')
    # 添加任务名
    parser.add_argument('--task', type=str, required=True,
                        help='task name.')
    # 解析命令行参数
    args = parser.parse_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)


    # 定义一个配置类，包含模拟环境和机器人的配置
    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sim_duration = 100.0  # 模拟的持续时间，单位是秒
            dt = 0.001  # 模拟的时间步长
            decimation = 10  # 降采样率

        class robot_config:
            # 定义机器人关节的PD控制器的增益
            kps = np.array([env_cfg.control.stiffness[joint] for joint in env_cfg.control.stiffness.keys()] * 2, dtype=np.double)
            kds = np.array([env_cfg.control.damping[joint] for joint in env_cfg.control.damping.keys()] * 2, dtype=np.double)

            print("kps", kps)

            tau_limit = 500. * np.ones(env_cfg.env.num_actions, dtype=np.double)  # 定义关节力矩的限制

            default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))
            print("default_dof_pos", default_dof_pos)


    # 加载模型
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, 'exported_policies')
    if args.load_model == None:
        jit_path = os.listdir(root_path)
        jit_path.sort()
        model_path = os.path.join(root_path, jit_path[-1])
    else:
        model_path = os.path.join(root_path, args.load_model)
    jit_name = os.listdir(model_path)
    model_path = os.path.join(model_path, jit_name[-1])
    policy = torch.jit.load(model_path)
    print("Load model from:", model_path)
    # 使用加载的模型和配置运行Mujoco模拟
    run_mujoco(policy, Sim2simCfg(), env_cfg)
