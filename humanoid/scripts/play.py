import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

import pygame
from threading import Thread

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
# x_scale, y_scale, yaw_scale = 2.5, 2.0, 0.0
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
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd

        while not exit_flag:
            # 获取手柄输入
            pygame.event.get()

            # 更新机器人命令
            x_vel_cmd = -joystick.get_axis(1) * 1.2
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1

            x_vel_cmd = np.clip(x_vel_cmd, -0.8, 0.8)

            # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

            # 等待一小段时间，可以根据实际情况调整
            pygame.time.delay(100)

        # 启动线程


    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # env_cfg.init_state.pos = [0.0, 0.0, 1.2]
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.measure_heights = True;

    env_cfg.domain_rand.add_ext_force = True

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    # env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continuous_push = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    # env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    # env_cfg.noise.noise_level = 1.0
    # env_cfg.sim.dt = 0.0005
    # env_cfg.sim.sim_duration = 60
    # env_cfg.control.decimation = 40
    env_cfg.commands.heading_command = False

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    robot_index = 0  # which robot is used for logging
    joint_index = 5  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                      np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S') + args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    obs = env.get_observations()

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    default_joint_angles = [0.4, 0.05, -0.31, 0.49, -0.21, 0., 0.15, -0.4, -0.05, 0.31, 0.49, -0.21, 0., 0.15]
    for i in range(10 * stop_state_log):

        actions = policy(obs.detach())  # * 0.

        if FIX_COMMAND:
            env.commands[:, 0] = 0.8  # 1.0
            env.commands[:, 1] = 0
            env.commands[:, 2] = 0
            env.commands[:, 3] = 0.

        else:
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        if i > stop_state_log * 0.2 and i < stop_state_log:
            dict = {
                'base_height': env.root_states[robot_index, 2].item(),
                'foot_z_l': env.rigid_state[robot_index, 6, 2].item() - env_cfg.rewards.feet_to_ankle_distance,
                'foot_z_r': env.rigid_state[robot_index, 13, 2].item() - env_cfg.rewards.feet_to_ankle_distance,
                'foot_forcez_l': env.contact_forces[robot_index, 4, 2].item(),
                'foot_forcez_r': env.contact_forces[robot_index, 9, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'command_x': x_vel_cmd,
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'command_y': y_vel_cmd,
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'command_yaw': yaw_vel_cmd,
                'dof_pos_target': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, 0].item(),
                'dof_vel': env.dof_vel[robot_index, 0].item(),
                'dof_torque': env.torques[robot_index, 0].item(),
                'command_sin': obs[0, 0].item(),
                'command_cos': obs[0, 1].item(),
            }

            # 添加 dof_pos_target 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos_target[{i}]'] = actions[robot_index, i].item() * env.cfg.control.action_scale + default_joint_angles[i],

            # 添加 dof_pos 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos[{i}]'] = env.dof_pos[robot_index, i].item(),

            # 添加 dof_torque 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_torque[{i}]'] = env.torques[robot_index, i].item(),

            # 添加 dof_vel 的项
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_vel[{i}]'] = env.dof_vel[robot_index, i].item(),

            logger.log_states(dict=dict)

        elif _ == stop_state_log:
            logger.plot_states()
        elif i == stop_state_log:
            logger.plot_states()

        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    # logger.print_rewards()

    if RENDER:
        video.release()


if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
