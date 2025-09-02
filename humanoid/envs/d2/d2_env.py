from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from humanoid.utils.math import wrap_to_pi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import Terrain
from .trajectory_generator_x2 import TrajectoryGenerator

# from collections import deque
def copysign_new(a, b):
    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)


def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class D2DHStandEnv(LeggedRobot):
    '''
    D2DHStandEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (Terrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_stance_mask(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = self.cfg.rewards.feet_to_ankle_distance
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.ref_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.ref_dof_vel = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.trajectory_generator = TrajectoryGenerator(vx=0.0, vy=0.0, wz=0.0, base_height=self.cfg.rewards.base_height_target, swing_height=self.cfg.rewards.target_feet_height,
                                                        stance_length=0.0, stance_width=0.279, num_envs=self.num_envs, device=self.device)
        self.trajectory_generator.cpg.dt = self.cfg.control.decimation * self.cfg.sim.dt
        self.stance_ratio = self.cfg.rewards.stance_ratio
        self.trajectory_generator.cpg.BETA[-1] = self.stance_ratio
        self.trajectory_generator.stance_ratio = self.stance_ratio
        self.trajectory_generator.swing_ratio = 1 - self.stance_ratio


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y

        self.rand_push_force[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] *= 0

        self.root_states[:, 7:9] += self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)  # angular vel xyz

        self.rand_push_torque[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] *= 0
        self.root_states[:, 10:13] += self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_cycle_time(self):
        # 根据vel_x进行变化     
        vel_x = self.commands[:, 0]
        vel_max = self.command_ranges["lin_vel_x"][1]
        CT_min = self.cfg.rewards.cycle_time_range[0]
        CT_max = self.cfg.rewards.cycle_time_range[1]
        cycle_time = CT_max - (CT_max - CT_min) * abs(vel_x) / vel_max
        return cycle_time

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        # cycle_time = self._get_cycle_time()

        if self.cfg.commands.sw_switch:
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            self.phase_length_buf[stand_command] = 0  # set this as 0 for which env is standing
            # self.gait_start is rand 0 or 0.5
            phase = (self.phase_length_buf * self.dt / cycle_time + self.gait_start) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.dt / cycle_time + self.gait_start

        # phase continue increase，if want robot stand, set 0
        # phase = phase * 0.8
        return phase

    def compute_contact_mask(self, phase):
        """
        计算接触概率函数 C(phi_t^i)。
        :param phase: 形状为 (num_envs,) 的张量，表示相位
        :return: 形状为 (num_envs, 2) 的张量，表示接触概率（左右脚）
        """
        phase = phase.to(self.device)

        phi_stance = self.cfg.rewards.stance_ratio
        phi_shift_r = phi_stance - 1.0
        phi_shift_l = phi_stance - 1.5
        sigma = 0.02
        
        # 计算未平移和已平移的归一化相位变量
        def compute_phi_bar(phase, phi_shift):
            phi_mod = torch.remainder(phase + phi_shift, 1.0)
            return torch.where(
                phi_mod < phi_stance,
                phi_stance * (phi_mod / phi_stance),
                0.5 + phi_stance * ((phi_mod - phi_stance) / (1 - phi_stance))
            )
        
        # phi_bar_no_shift = compute_phi_bar(phase, 0)
        phi_bar_shift_r = compute_phi_bar(phase, phi_shift_r)
        phi_bar_shift_l = compute_phi_bar(phase, phi_shift_l)
        
        # 定义标准正态分布的 CDF 函数
        def Phi(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=self.device))))

        # 计算接触概率 C_no_shift 和 C_shift
        def compute_contact_probability(phi_bar, phi_mod):
            return (
                Phi(phi_bar / sigma) * (1 - Phi((phi_mod - phi_stance) / sigma)) +
                Phi((phi_mod - 1) / sigma) * (1 - Phi((phi_mod - 1.5) / sigma))
            )

        # C_no_shift = compute_contact_probability(phi_bar_no_shift, torch.remainder(phase, 1.0))
        C_shift_r = compute_contact_probability(phi_bar_shift_r, torch.remainder(phase + phi_shift_r, 1.0))
        C_shift_l = compute_contact_probability(phi_bar_shift_l, torch.remainder(phase + phi_shift_l, 1.0))

        # 生成 stance mask
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = C_shift_l  # 左脚
        stance_mask[:, 1] = C_shift_r   # 右脚

        return stance_mask

    def _get_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        phase_clamp = phase.clone()
        phase_clamp = phase_clamp % 1.0
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)

        stance_mask = self.compute_contact_mask(phase_clamp)
        # print(stance_mask)
        # stand mask == 1 means stand leg 
        return stance_mask
    
    def _get_hard_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)

        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # # Add double support phase
        stance_mask[torch.abs(sin_pos) < 0.05] = 1

        # print(stance_mask)
        # stand mask == 1 means stand leg 
        return stance_mask

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return

        # rand sample 
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                                    gait_time_range[1],
                                                    (len(envs), 1), device=self.device)
            random_tensor_list.append(random_tensor_single)

        # random_tensor_list = [[gait1_env1, gait1_env2,...], [gait2_env1, gait2_env2,...], [gait3_env1, gait3_env2,...]...]
        # random_tensor = [[gait1_env1, gait2_env1,...], [gait1_env2, gait2_env2,...], [gait1_env3, gait2_env3,...]...]
        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        # current_sum = [[gait1_env1+gait2_env1+...],[gait1_env2+gait2_env2+...],[gait1_env3+gait2_env3+...]...]
        current_sum = torch.sum(random_tensor, dim=1, keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        # scaled_tensor=[[0, env1_gait1_duration_tick, env1_gait2_duration_tick,...], [0, env2_gait1_duration_tick, env2_gait2_duration_tick,...],...]
        scaled_tensor[:, 1:] = scaled_tensor[:, :-1].clone()
        scaled_tensor[:, 0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor, dim=1).int()

    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:, i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

        self.trajectory_generator.vx[env_ids] = self.commands[env_ids, 0]
        self.trajectory_generator.vy[env_ids] = self.commands[env_ids, 1]
        self.trajectory_generator.wz[env_ids] = self.commands[env_ids, 2]

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_sagittal_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_rotate_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_walk_omnidirectional_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)

    # apply force and torque on fix body
    def _add_ext_force(self):
        apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        apply_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # apply force and torque on fix body
        for i in range(self.num_bodies):
            if i == 0:
                apply_forces[:, i, :] = self.ext_forces
                apply_torques[:, i, :] = self.ext_torques
            else:
                apply_forces[:, i, :] = self.ext_forces * 0.1
                apply_torques[:, i, :] = self.ext_torques * 0.1

        apply_forces[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] *= 0
        apply_torques[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] *= 0

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(apply_forces), gymtorch.unwrap_tensor(apply_torques), gymapi.ENV_SPACE)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # get all robot surrounding height
            self.measured_heights = self._get_heights()

        # TODO self.common_step_counter don't set zero when robot reset, so is it curriculum for push duration???
        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter / self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i] / self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()

        if self.cfg.domain_rand.add_ext_force:
            i = int(self.common_step_counter / self.cfg.domain_rand.add_update_step)
            if i >= len(self.cfg.domain_rand.add_duration):
                i = len(self.cfg.domain_rand.add_duration) - 1
            duration = self.cfg.domain_rand.add_duration[i] / self.dt
            if self.common_step_counter % self.cfg.domain_rand.ext_force_interval <= duration:

                if self.common_step_counter % self.cfg.domain_rand.ext_force_interval == 0:
                    force_xy = torch_rand_float(-self.cfg.domain_rand.ext_force_max_xy, self.cfg.domain_rand.ext_force_max_xy, (self.num_envs, 2), device=self.device)
                    force_z = torch_rand_float(-self.cfg.domain_rand.ext_force_max_z, self.cfg.domain_rand.ext_force_max_z, (self.num_envs, 1), device=self.device)
                    self.ext_forces = torch.cat((force_xy, force_z), 1)
                    self.ext_torques = torch_rand_float(-self.cfg.domain_rand.ext_torque_max, self.cfg.domain_rand.ext_torque_max, (self.num_envs, 3), device=self.device)

                self._add_ext_force()

    def compute_ref_state(self):
        if self.cfg.rewards.use_ref_ik:
            phase = self._get_phase()
            phase_clamp = phase.clone()
            t = phase_clamp % 1.0
            zero_tensor = torch.zeros_like(self.default_dof_pos)
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            self.ref_dof_pos[stand_command,:] = self.default_dof_pos
            self.ref_dof_vel[stand_command,:] = zero_tensor

            sin_pos = torch.sin(2 * torch.pi * t)
            # like_sin_posl = self.like_sin_signal_l(self.stance_ratio)
            # like_sin_posr = self.like_sin_signal_r(self.stance_ratio)
            # like_sin_pos_l = like_sin_posl.clone()
            # like_sin_pos_r = like_sin_posr.clone()
            like_sin_pos_l = -sin_pos.clone()
            like_sin_pos_r = sin_pos.clone()
            ones_tensor = torch.ones_like(like_sin_pos_l)
            paths_raw = self.trajectory_generator.get_end_effector_ref(t)
            joint_pos_raw = self.trajectory_generator.InverseKinematics(paths_raw)
            joint_vel_raw = self.trajectory_generator.get_joint_vel(t)
            self.ref_dof_pos[~stand_command, :6] = joint_pos_raw[~stand_command, :, 0]
            self.ref_dof_pos[~stand_command, 6] = like_sin_pos_l[~stand_command] * self.cfg.rewards.final_swing_joint_delta_pos[6] + ones_tensor[~stand_command] * self.default_dof_pos[0, 6]
            self.ref_dof_pos[~stand_command, 7:13] = joint_pos_raw[~stand_command, :, 1]
            self.ref_dof_pos[~stand_command, 13] = like_sin_pos_r[~stand_command] * self.cfg.rewards.final_swing_joint_delta_pos[13] + ones_tensor[~stand_command] * self.default_dof_pos[0, 13]

            self.ref_dof_vel[~stand_command, :6] = joint_vel_raw[~stand_command, :, 0]
            self.ref_dof_vel[~stand_command, 6] = .0
            self.ref_dof_vel[~stand_command, 7:13] = joint_vel_raw[~stand_command, :, 1]
            self.ref_dof_vel[~stand_command, 13] = .0

            # condition = (torch.abs(sin_pos) < self.stance_ratio - 0.5) & (~stand_command)
            # self.ref_dof_pos[condition, 6] = ones_tensor[~stand_command] * self.default_dof_pos[0, 6]
            # self.ref_dof_pos[condition, 13] = ones_tensor[~stand_command] * self.default_dof_pos[0, 13]
        else:
            phase = self._get_phase()
            sin_pos = torch.sin(2 * torch.pi * phase)
            sin_pos_l = sin_pos.clone()
            sin_pos_r = sin_pos.clone()

            arm_sin_pos_l = sin_pos.clone()  # -1   1
            arm_sin_pos_r = sin_pos.clone()

            self.ref_dof_pos = torch.zeros_like(self.dof_pos)

            sin_pos_l[sin_pos_l > 0] = 0
            self.ref_dof_pos[:, 0] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[0]
            self.ref_dof_pos[:, 1] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[1]
            self.ref_dof_pos[:, 2] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[2]
            self.ref_dof_pos[:, 3] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[3]
            self.ref_dof_pos[:, 4] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[4]
            self.ref_dof_pos[:, 5] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[5]
            # self.ref_dof_pos[:, 6] = -arm_sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[6]
            # print(phase[0], sin_pos_l[0])
            # right
            sin_pos_r[sin_pos_r < 0] = 0
            self.ref_dof_pos[:, 6] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[6]
            self.ref_dof_pos[:, 7] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[7]
            self.ref_dof_pos[:, 8] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[8]
            self.ref_dof_pos[:, 9] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[9]
            self.ref_dof_pos[:, 10] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[10]
            self.ref_dof_pos[:, 11] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[11]
            # self.ref_dof_pos[:, 13] = arm_sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[13]

            self.ref_dof_pos[torch.abs(sin_pos) <= 0.05] = 0.
            self.ref_dof_pos += self.default_dof_pos



    def like_sin_signal_r(self, stance_ratio):
        phase = self._get_phase()
        phase_clamp = phase.clone()
        t = phase_clamp % 1.0
        swing_ratio = 1 - stance_ratio
        a1 = torch.ge(t, 0); b1 = torch.lt(t, swing_ratio)
        a2 = torch.ge(t, swing_ratio); b2 = torch.lt(t, 1.0)

        a = [0.0, 0.05, 2.5, -9.4, 6.0, 4.8]  # 多项式系数
        ref_feet_height = 20 * sum(a[k] * ((t * 5 / (10 * swing_ratio)) ** k) for k in range(6)) * 1.0
        
        like_sin_pos = torch.logical_and(a1, b1) * ref_feet_height + \
                        torch.logical_and(a2, b2) * 0.0 
        return like_sin_pos

    
    def like_sin_signal_l(self, stance_ratio):
        phase = self._get_phase()
        phase_clamp = phase.clone()
        t = phase_clamp % 1.0
        swing_ratio = 1 - stance_ratio
        a1 = torch.ge(t, 0); b1 = torch.lt(t, swing_ratio-0.5)
        a2 = torch.ge(t, swing_ratio-0.5); b2 = torch.lt(t, 0.5)
        a3 = torch.ge(t, 0.5); b3 = torch.lt(t, 1.0-(stance_ratio-0.5))
        a = [0.0, 0.05, 2.5, -9.4, 6.0, 4.8]  # 多项式系数
        ref_feet_height_1 = 20 * sum(a[k] * (((t-0.5) * 5 / (10 * swing_ratio)) ** k) for k in range(6)) * 1.0
        ref_feet_height_2 = 20 * sum(a[k] * (((t+0.5) * 5 / (10 * swing_ratio)) ** k) for k in range(6)) * 1.0
        
        like_sin_pos = torch.logical_and(a1, b1) * ref_feet_height_2 + \
                       torch.logical_and(a2, b2) * 0.0 + \
                       torch.logical_and(a3, b3) * ref_feet_height_1
        return like_sin_pos

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: self.cfg.env.num_commands] = 0.  # commands
        noise_vec[self.cfg.env.num_commands: self.cfg.env.num_commands + self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[self.cfg.env.num_commands + self.num_actions: self.cfg.env.num_commands + 2 * self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[self.cfg.env.num_commands + 2 * self.num_actions: self.cfg.env.num_commands + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[self.cfg.env.num_commands + 3 * self.num_actions: self.cfg.env.num_commands + 3 * self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[self.cfg.env.num_commands + 3 * self.num_actions + 3: self.cfg.env.num_commands + 3 * self.num_actions + 6] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_hard_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        base_height = torch.unsqueeze(self._get_base_heights(), dim=1)
        feet_height = self._get_feet_heights()
        feet_contact_forces = self.contact_forces[:, self.feet_indices, 2].clip(0, 600)

        critic_obs_rpy = self.base_euler_xyz.clone()
        critic_obs_rpy[:, 2] = 0
        # critic no lag
        diff = self.dof_pos - self.ref_dof_pos
        # 73
        privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            critic_obs_rpy * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 10.,  # 1 # sum of all fix link mass
            stance_mask,  # 2
            contact_mask,  # 2
            base_height, # 1
            feet_height, # 2
            feet_contact_forces / 500.0, # 2
        ), dim=-1)

        # dof_pos and dof_vel has same lag
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                      self.cfg.domain_rand.dof_lag_timesteps_range[1] + 1, (self.num_envs,), device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:, self.dof_lag_timestep.long()]
            # dof_pos and dof_vel has different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.dof_pos_lag_timesteps_range[1] + 1, (self.num_envs,), device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :, self.dof_pos_lag_timestep.long()]

            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.dof_vel_lag_timesteps_range[1] + 1, (self.num_envs,), device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        # imu lag, including rpy and omega
        if self.cfg.domain_rand.add_imu_lag:
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                      self.cfg.domain_rand.imu_lag_timesteps_range[1] + 1, (self.num_envs,), device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:, :3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:, -3:].clone()
        # no imu lag
        else:
            self.lagged_base_ang_vel = self.base_ang_vel[:, :3].clone()
            self.lagged_base_euler_xyz = self.base_euler_xyz[:, -3:].clone()

        self.lagged_base_euler_xyz[:, 2] = 0
        # obs q and dq
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        # 47
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,  # 12
            dq,  # 12
            self.actions,  # 12
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.env.num_single_obs == 48:
            stand_command = (torch.norm(self.commands[:, :3], dim=1, keepdim=True) <= self.cfg.commands.stand_com_threshold)
            obs_buf = torch.cat((obs_buf, stand_command), dim=1)

        # if self.cfg.terrain.measure_heights:
        #     # TODO why measure_heights add noise for privileged_obs ???
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     privileged_obs_buf = torch.cat((privileged_obs_buf.clone(), heights), dim=-1)

        if self.add_noise:
            # add obs noise
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)
        # maxlen is frame_stack
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        # self.obs_buf frame_stack * num_single_obs
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        # self.privileged_obs_buf single_num_privileged_obs * c_frame_stack
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset rand dof_pos and dof_vel=0
        self._reset_dofs(env_ids)

        # reset base position
        self._reset_root_states(env_ids)

        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self.randomize_lag_props(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device) * 0.5

        # resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # fix reset gravity bug
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        # clear obs history buffer and privileged obs buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait), dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device) * 0.5

    # ================================================ Rewards ================================================== #
    # def Gaussian kernel function
    # sigma increase, function fat
    def G_alpha_sigma_torch(self, x, alpha=1, sigma=1):
        return alpha * torch.exp(-(x / sigma) ** 2)

    # def generalized Cauchy kernel function
    # sigma increase, function fat // beta increase, function square
    def C_alpha_beta_sigma_torch(self, x, alpha=1, beta=1, sigma=1):
        return alpha * ((x / sigma) ** (2 * beta) + 1) ** -1

    def _reward_ref_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()

        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos.clone()

        diff = joint_pos - pos_target

        diff_norm = torch.norm(diff, dim=1)

        r = torch.exp(-2 * diff_norm) - 0.2 * diff_norm.clamp(0, 0.5)
        r[stand_command] = 1.0
        return r
    
    def _reward_ref_joint_pos_arm(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()

        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos.clone()

        diff = joint_pos - pos_target
        arm_diff = torch.norm(diff[:, [6]], dim=1) + torch.norm(diff[:, [13]], dim=1)

        r = torch.exp(-5 * arm_diff) - 0.1 * arm_diff.clamp(0, 0.5)
        r[stand_command] = 0.2
        return r

    def _reward_ref_joint_vel(self):
        """
        Calculates the reward based on the difference between the current joint velocity and the target joint velocity.
        """
        error = self.dt * torch.norm(self.dof_vel - self.ref_dof_vel, dim=1)
        rew = torch.exp(-2 * error)

        ori_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        ori_error = torch.exp(-ori_error)
        vel_all = torch.norm(self.commands[:, :3], dim=1) 
        
        rew *= vel_all / torch.max(vel_all)
        rew *= ori_error / torch.max(ori_error)
        return rew

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        # !!! use knee_indices
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact conditions.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm1 = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)  # 拧
        foot_speed_norm2 = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)  # 滑动
        rew = torch.sqrt(foot_speed_norm1 + foot_speed_norm2)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_hard_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) < 0.05] = 1
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.6) * first_contact
        # air_time = (self.feet_air_time - 0.5) * first_contact
        rew_airTime = torch.sum(air_time, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        # return air_time.sum(dim=1)
        return rew_airTime

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_hard_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        # # 检查 stance_mask 是否完全为 False
        # is_stance_all_false = (stance_mask[:, 0] == 0) & (stance_mask[:, 1] == 0)
        # # 当 stance_mask 完全为 False 时，使用特定的奖励逻辑
        # reward = torch.where(is_stance_all_false.unsqueeze(1),  # 扩展维度以匹配 reward 的形状
        #                     torch.where(contact == stance_mask, 3.5, -2.0),
        #                     reward)
        return torch.mean(reward, dim=1)

    def _reward_feet_swing(self):
        # 当脚处于摆动时  & 脚接触地面了  给负奖励  (其余情况奖励为0)
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # print("contact",contact)
        # Compute swing mask
        swing_mask = 1 - self._get_hard_stance_mask()
        # print("swing_mask",swing_mask)

        rew_swing = -torch.sum(contact * swing_mask, dim=1)
        # print("rew_swing",rew_swing)
        return rew_swing   
    
    def _reward_gait_feet_frc_perio(self):
        """Penalize foot force during the swing phase of the gait."""
        foot_velocities = torch.norm(self.rigid_state[:, self.feet_indices, 7:10], dim=2).view(self.num_envs, -1)
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        swing_mask = 1 - self._get_hard_stance_mask().clone()
        left_frc_swing_mask = swing_mask[:, 0]
        right_frc_swing_mask = swing_mask[:, 1]
        left_frc_score = left_frc_swing_mask * (torch.exp(-50 * torch.square(foot_forces[:, 0])))
        right_frc_score = right_frc_swing_mask * (torch.exp(-50 * torch.square(foot_forces[:, 1])))
        return left_frc_score + right_frc_score


    def _reward_gait_feet_spd_perio(self):
        """Penalize foot speed during the support phase of the gait."""
        foot_velocities = torch.norm(self.rigid_state[:, self.feet_indices, 7:10], dim=2).view(self.num_envs, -1)
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        stance_mask = self._get_hard_stance_mask().clone()
        left_spd_support_mask = stance_mask[:, 0]
        right_spd_support_mask = stance_mask[:, 1]
        left_spd_score = left_spd_support_mask * (torch.exp(-150 * torch.square(foot_velocities[:, 0])))
        right_spd_score = right_spd_support_mask * (torch.exp(-150 * torch.square(foot_velocities[:, 1])))
        return left_spd_score + right_spd_score 

    def _reward_tracking_contacts_shaped_vel_and_force(self):
        foot_velocities = torch.norm(self.rigid_state[:, self.feet_indices, 7:10], dim=2).view(self.num_envs, -1)
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        stance_mask = self._get_hard_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = 0
        for i in range(2):
            reward += - (stance_mask[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] / 5.))) - (1 - stance_mask[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] / 100.))
        return reward

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)

        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # print(self.contact_forces[:, self.feet_indices, :])
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos

        # addtional_roll = torch.norm(joint_diff[:, [0]], dim=1) + torch.norm(joint_diff[:, [6]], dim=1)
        # addtional_yaw = torch.norm(joint_diff[:, [1]], dim=1) + torch.norm(joint_diff[:, [7]], dim=1)
        left_yaw_roll = joint_diff[:, [0, 1, 5]]
        right_yaw_roll = joint_diff[:, [6, 7, 11]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 50) - 0.01 * torch.norm(joint_diff, dim=1)
            #   torch.exp(-addtional_roll * 50) + torch.exp(-addtional_yaw * 50)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        base_height = self._get_base_heights()
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        # stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        r = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
        # r[stand_command] = r.clone()[stand_command] * 0.7
        # r[stand_command] = 1.0
        return r

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error_square = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_abs = torch.sum(torch.abs(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        ang_vel_error_square = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_abs = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        r_square = torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.feet_height = self._get_feet_heights() - self.cfg.rewards.feet_to_ankle_distance
        # Compute swing mask
        swing_mask = 1 - self._get_hard_stance_mask()

        # like_sin_posl = self.like_sin_signal_l(self.stance_ratio)
        # like_sin_posr = self.like_sin_signal_r(self.stance_ratio)
        # like_sin_pos_l = like_sin_posl.clone()
        # like_sin_pos_r = like_sin_posr.clone()
        
        # ref_feet_height = torch.zeros_like(self.feet_air_time)
        # ref_feet_height[:, 0] += like_sin_pos_l * self.cfg.rewards.target_feet_height
        # ref_feet_height[:, 1] += like_sin_pos_r * self.cfg.rewards.target_feet_height

        # rew = torch.abs(self.feet_height - ref_feet_height) < 0.01
        # return torch.sum(rew * swing_mask, dim=1)

        # feet height should larger than target feet height at the peak
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height) * (self.feet_height < self.cfg.rewards.target_feet_height_max)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_ankle_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        ankle_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.torques[:, ankle_idx]), dim=1)

    def _reward_feet_rotation(self):
        feet_euler_xyz = self.feet_euler_xyz
        rotation = torch.sum(torch.square(feet_euler_xyz[:, :, :2]), dim=[1, 2])
        feet_yaw_err = self.feet_euler_xyz[:, :, 2].clone()

        feet_yaw_err[:, 0] = self.feet_euler_xyz[:, 0, 2] - self.base_euler_xyz[:, 2]
        feet_yaw_err[:, 1] = self.feet_euler_xyz[:, 1, 2] - self.base_euler_xyz[:, 2]

        feet_yaw_err[feet_yaw_err < -np.pi] += 2 * np.pi
        feet_yaw_err[feet_yaw_err > np.pi] -= 2 * np.pi

        feet_yaw_mismatch = torch.exp(-torch.sum(torch.abs(feet_yaw_err), dim=1) * 10)

        # rotation = torch.sum(torch.square(feet_euler_xyz[:,:,1]),dim=1)
        r = torch.exp(-rotation * 15) + feet_yaw_mismatch

        return r

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_stand_still(self):
        # penalize motion at zero commands
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(stand_command, r.clone(), torch.zeros_like(r))
        return r

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_base_heading(self):
        # Reward tracking desired base heading
        command_heading = self.commands[:, 4]  # yaw
        base_heading_error = torch.abs(wrap_to_pi(command_heading - self.base_heading.squeeze(1)))
        # print("command_heading",command_heading)
        # print("self.base_heading",self.base_heading)
        return self._neg_exp(base_heading_error, a=torch.pi / 2)

    def _neg_exp(self, x, a=1):
        """ shorthand helper for negative exponential e^(-x/a)
            a: range of x
        """
        return torch.exp(-(x / a) / self.cfg.rewards.tracking_sigma)
