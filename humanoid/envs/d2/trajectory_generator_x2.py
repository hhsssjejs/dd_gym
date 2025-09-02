import torch
import numpy as np
from .hopf_oscillator import HOPF_CPG
import time
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

class TrajectoryGenerator:
    def __init__(self, vx, vy, wz, base_height, swing_height, stance_length, stance_width, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.stance_ratio = 0.5
        self.swing_ratio = 1 - self.stance_ratio
        # urdf information
        self.a0x = 0.0 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a0y = 0.0895 * torch.tensor([1.0, -1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a0z = -0.13693 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a1x = 0.0 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a1y = 0.05 * torch.tensor([1.0, -1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a2z = -0.13420 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a3x = -0.00500 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a3z = -0.16050 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a4x = -0.02000 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a4z = -0.28000 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.a5z = -0.068 * torch.tensor([1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)

        self.ay_all = 0.1395 * torch.tensor([1.0, -1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)

        self.cpg = HOPF_CPG(num_envs=self.num_envs, device=self.device)
        self.vx = vx * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.vy = vy * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wz = wz * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 脚底板一点的位置是相对于base坐标系描述的
        self.feet_z_in_base = -base_height * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.swing_height = swing_height * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.stance_length = stance_length * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.stance_width = stance_width * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

        self.stance_time = self.cpg.get_stance_time()
        self.stance_time = self.stance_ratio * 0.75 * 0.5
        self.dx = self.vx * self.stance_time
        self.dy = self.vy * self.stance_time
        # 利用两腿x方向速差实现yaw方向旋转
        self.kl = torch.tensor([-1.0, 1.0], device=self.device, requires_grad=False).unsqueeze(0).repeat(self.num_envs, 1)
        self.dwx = self.wz * (self.stance_width / 2.0) * self.stance_time

    def get_end_effector_ref(self, phase):
        EndEffectorRef = torch.zeros(self.num_envs, 3, 2, dtype=torch.float, device=self.device, requires_grad=False)
        a1 = torch.ge(phase, 0); b1 = torch.lt(phase, self.swing_ratio)
        a2 = torch.ge(phase, self.swing_ratio); b2 = torch.lt(phase, 1.0)

        a = [0.0, 0.05, 2.5, -9.4, 6.0, 4.8]  # 多项式系数
        ref_feet_height = 20 * sum(a[k] * ((phase * 5 / (10 * self.swing_ratio)) ** k) for k in range(6)) * 1.0
        
        like_sin_pos_r = torch.logical_and(a1, b1) * ref_feet_height + \
                        torch.logical_and(a2, b2) * 0.0 
        

        a1 = torch.ge(phase, 0); b1 = torch.lt(phase, self.swing_ratio-0.5)
        a2 = torch.ge(phase, self.swing_ratio-0.5); b2 = torch.lt(phase, 0.5)
        a3 = torch.ge(phase, 0.5); b3 = torch.lt(phase, 1.0-(self.stance_ratio-0.5))
        a = [0.0, 0.05, 2.5, -9.4, 6.0, 4.8]  # 多项式系数
        ref_feet_height_1 = 20 * sum(a[k] * (((phase-0.5) * 5 / (10 * self.swing_ratio)) ** k) for k in range(6)) * 1.0
        ref_feet_height_2 = 20 * sum(a[k] * (((phase+0.5) * 5 / (10 * self.swing_ratio)) ** k) for k in range(6)) * 1.0
        
        like_sin_pos_l = torch.logical_and(a1, b1) * ref_feet_height_2 + \
                        torch.logical_and(a2, b2) * 0.0 + \
                        torch.logical_and(a3, b3) * ref_feet_height_1
        
        # 为便于多项式插值，将[-pi, pi)映射到[-1, 1)
        # swing: [-1, 0), 0 --> -1
        # stance: [0, 1), 1 --> 0

        #right leg
        # in swing -phase: 0 --> 1
        mask = torch.logical_or((like_sin_pos_r > 0.0),(phase == 0.0))
        # mask = like_sin_pos_r > 0.0 or phase == 0.0
        phase_sw = phase/self.swing_ratio
        EndEffectorRef[mask, 0, 1] = self.a0x[mask, 1] + self.a1x[mask, 1] + self.a3x[mask, 1] + self.a4x[mask, 1] - (self.dx[mask] + self.kl[mask, 1] * self.dwx[mask]) * (6.0 * (phase_sw[mask]) ** 5 - 15.0 * (phase_sw[mask]) ** 4 + 10.0 * (phase_sw[mask]) ** 3 - 0.5)
        EndEffectorRef[mask, 1, 1] = self.a0y[mask, 1] + self.a1y[mask, 1] + self.dy[mask] * (6.0 * (phase_sw[mask]) ** 5 - 15.0 * (phase_sw[mask]) ** 4 + 10.0 * (phase_sw[mask]) ** 3 - 0.5)
        EndEffectorRef[mask, 2, 1] = self.feet_z_in_base[mask] + self.swing_height[mask] * (like_sin_pos_r[mask])
    
        # in stance phase: 1 --> 0
        phase_st = 1.0 - (phase-self.swing_ratio)/self.stance_ratio
        EndEffectorRef[~mask, 0, 1] = self.a0x[~mask, 1] + self.a1x[~mask, 1] + self.a3x[~mask, 1] + self.a4x[~mask, 1] - (self.dx[~mask] + self.kl[~mask, 1] * self.dwx[~mask]) * (6.0 * phase_st[~mask] ** 5 - 15.0 * phase_st[~mask] ** 4 + 10.0 * phase_st[~mask] ** 3 - 0.5)
        EndEffectorRef[~mask, 1, 1] = self.a0y[~mask, 1] + self.a1y[~mask, 1] + self.dy[~mask] * (6.0 * phase_st[~mask] ** 5 - 15.0 * phase_st[~mask] ** 4 + 10.0 * phase_st[~mask] ** 3 - 0.5)
        EndEffectorRef[~mask, 2, 1] = self.feet_z_in_base[~mask]
        
        #left leg
        if self.stance_ratio <= 0.5: 
            mask = like_sin_pos_l > 0.0
            sl = phase > 0.5
            phase_sw[sl] = (phase[sl]-0.5)/self.swing_ratio
            phase_sw[~sl] = (phase[~sl]+0.5)/self.swing_ratio
            # phase_sw = (phase-0.5)/self.swing_ratio if phase > 0.5 else (phase+0.5)/self.swing_ratio
            EndEffectorRef[mask, 0, 0] = self.a0x[mask, 0] + self.a1x[mask, 0] + self.a3x[mask, 0] + self.a4x[mask, 0] - (self.dx[mask] + self.kl[mask, 0] * self.dwx[mask]) * (6.0 * (phase_sw[mask]) ** 5 - 15.0 * (phase_sw[mask]) ** 4 + 10.0 * (phase_sw[mask]) ** 3 - 0.5)
            EndEffectorRef[mask, 1, 0] = self.a0y[mask, 0] + self.a1y[mask, 0] + self.dy[mask] * (6.0 * (phase_sw[mask]) ** 5 - 15.0 * (phase_sw[mask]) ** 4 + 10.0 * (phase_sw[mask]) ** 3 - 0.5)
            EndEffectorRef[mask, 2, 0] = self.feet_z_in_base[mask] + self.swing_height[mask] * (like_sin_pos_l[mask])
        
            # in stance phase: 1 --> 0
            phase_st = 1.0 - (phase-(0.5-self.stance_ratio))/self.stance_ratio
            EndEffectorRef[~mask, 0, 0] = self.a0x[~mask, 0] + self.a1x[~mask, 0] + self.a3x[~mask, 0] + self.a4x[~mask, 0] - (self.dx[~mask] + self.kl[~mask, 0] * self.dwx[~mask]) * (6.0 * phase_st[~mask] ** 5 - 15.0 * phase_st[~mask] ** 4 + 10.0 * phase_st[~mask] ** 3 - 0.5)
            EndEffectorRef[~mask, 1, 0] = self.a0y[~mask, 0] + self.a1y[~mask, 0] + self.dy[~mask] * (6.0 * phase_st[~mask] ** 5 - 15.0 * phase_st[~mask] ** 4 + 10.0 * phase_st[~mask] ** 3 - 0.5)
            EndEffectorRef[~mask, 2, 0] = self.feet_z_in_base[~mask]
        else:
            mask = like_sin_pos_l > 0.0
            phase_sw = (phase-0.5)/self.swing_ratio
            # phase_sw = (phase-0.5)/self.swing_ratio if phase > 0.5 else (phase+0.5)/self.swing_ratio
            EndEffectorRef[mask, 0, 0] = self.a0x[mask, 0] + self.a1x[mask, 0] + self.a3x[mask, 0] + self.a4x[mask, 0] - (self.dx[mask] + self.kl[mask, 0] * self.dwx[mask]) * (6.0 * (phase_sw[mask]) ** 5 - 15.0 * (phase_sw[mask]) ** 4 + 10.0 * (phase_sw[mask]) ** 3 - 0.5)
            EndEffectorRef[mask, 1, 0] = self.a0y[mask, 0] + self.a1y[mask, 0] + self.dy[mask] * (6.0 * (phase_sw[mask]) ** 5 - 15.0 * (phase_sw[mask]) ** 4 + 10.0 * (phase_sw[mask]) ** 3 - 0.5)
            EndEffectorRef[mask, 2, 0] = self.feet_z_in_base[mask] + self.swing_height[mask] * (like_sin_pos_l[mask])

            sl = phase > 0.5
            phase_st[sl] = 1.0 - (phase[sl]-(0.5+(1-self.stance_ratio)))/self.stance_ratio
            phase_st[~sl] = 1.0 - (phase[~sl])/self.stance_ratio
            EndEffectorRef[~mask, 0, 0] = self.a0x[~mask, 0] + self.a1x[~mask, 0] + self.a3x[~mask, 0] + self.a4x[~mask, 0] - (self.dx[~mask] + self.kl[~mask, 0] * self.dwx[~mask]) * (6.0 * phase_st[~mask] ** 5 - 15.0 * phase_st[~mask] ** 4 + 10.0 * phase_st[~mask] ** 3 - 0.5)
            EndEffectorRef[~mask, 1, 0] = self.a0y[~mask, 0] + self.a1y[~mask, 0] + self.dy[~mask] * (6.0 * phase_st[~mask] ** 5 - 15.0 * phase_st[~mask] ** 4 + 10.0 * phase_st[~mask] ** 3 - 0.5)
            EndEffectorRef[~mask, 2, 0] = self.feet_z_in_base[~mask]

        return EndEffectorRef
    
    def InverseKinematics(self, EndEffectorRef):
        # EndEffectorRef是在base坐标系中描述的
        px = EndEffectorRef[:, 0, :]
        py = EndEffectorRef[:, 1, :]
        pz = EndEffectorRef[:, 2, :]

        tmp1 = torch.sqrt(torch.square(py-self.ay_all) + torch.square(pz-self.a0z-self.a5z))
        tmp2 = torch.sqrt(torch.square(px-self.a0x-self.a1x) + torch.square(-tmp1))

        # print("y axis: ", py-self.ay_all)
        # q0 = torch.arctan2(tmp1, px-self.a0x-self.a1x)
        q1 = torch.arctan2(py-self.ay_all, -pz+self.a0z+self.a5z)
        q2 = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        l_up_knee = torch.sqrt(torch.square(self.a2z+self.a3z) + torch.square(self.a3x))
        l_low_knee = torch.sqrt(torch.square(self.a4z) + torch.square(self.a4x))
        l_init = torch.sqrt(torch.square(self.a2z+self.a3z+self.a4z) + torch.square(self.a3x+self.a4x))
        q3_init = torch.arccos(torch.clip((torch.square(l_init) - torch.square(l_up_knee) - torch.square(l_low_knee))/(2.0 * l_up_knee * l_low_knee), -1.0, 1.0))
        # print("q3_init :",q3_init)
        q3 = torch.arccos(torch.clip((torch.square(tmp2) - torch.square(l_up_knee) - torch.square(l_low_knee))/(2.0 * l_up_knee * l_low_knee), -1.0, 1.0)) - q3_init

        q_tmp = torch.arctan2(px-self.a0x-self.a1x, tmp1)
        # print("q3 :",q3)
        # print("q_tmp :",q_tmp)
        q0 = -torch.arctan2(l_low_knee*torch.sin(q3), l_up_knee + l_low_knee*torch.cos(q3)) + q_tmp

        q4 = -(q0+q3)
        q5 = -q1

        qRef = torch.zeros(self.num_envs, 6, 2, device=self.device, requires_grad=False)

        qRef[:, 0, :] = q0
        qRef[:, 1, :] = q1
        qRef[:, 2, :] = q2
        qRef[:, 3, :] = q3
        qRef[:, 4, :] = q4
        qRef[:, 5, :] = q5

        return qRef
    
    def ForwardKinematics(self, qRef):
        # 返回相对于base坐标系描述的由ankle交点延伸至脚底板上一点的三维坐标
        EndEffectorPos = torch.zeros(self.num_envs, 3, 2, device=self.device, requires_grad=False)
        EndEffectorPos[:, 0, :] = self.a0x + self.a1x + self.a3z*torch.sin(qRef[:, 2, :]) + self.a4z*torch.sin(qRef[:, 2, :]+qRef[:, 3, :])
        EndEffectorPos[:, 1, :] = self.a0y - self.a2z*torch.sin(qRef[:, 0, :]) - self.a3z*torch.sin(qRef[:, 0, :])*torch.cos(qRef[:, 2, :]) - self.a4z*torch.sin(qRef[:, 0, :])*torch.cos(qRef[:, 2, :]+qRef[:, 3, :])
        EndEffectorPos[:, 2, :] = self.a0z + self.a5z + self.a2z*torch.cos(qRef[:, 0, :]) + self.a3z*torch.cos(qRef[:, 0, :])*torch.cos(qRef[:, 2, :]) + self.a4z*torch.cos(qRef[:, 0, :])*torch.cos(qRef[:, 2, :]+qRef[:, 3, :])
        
        return EndEffectorPos
    
    def get_joint_vel(self,phase):
        # dt = self.cpg.dt
        # gamma = self.cpg.get_gamma()
        dt = 0.01
        # phase = self.cpg.get_raw_phase()
        phase_head = (phase + dt) % 1.0
        phase_tail = (phase - dt) % 1.0
        EndEffectorRef_head = self.get_end_effector_ref(phase_head)
        EndEffectorRef_tail = self.get_end_effector_ref(phase_tail)
        joint_vel = (self.InverseKinematics(EndEffectorRef_head) - self.InverseKinematics(EndEffectorRef_tail)) / (2*dt)

        return joint_vel

    def test_path_generation(self, total_time):
        """
        模拟CPG一段时间
        total_time: 总模拟时间
        """
        dt = self.cpg.dt
        num_steps = int(total_time / dt)

        paths = torch.zeros((num_steps, 3, 2), device=self.device)
        leg_pos = torch.zeros((num_steps, 3, 2), device=self.device)
        phases = torch.zeros((num_steps, 2), device=self.device)
        joint_poses = torch.zeros((num_steps, 6, 2), device=self.device)
        joint_vels = torch.zeros((num_steps, 6, 2), device=self.device)

        env_ids = torch.arange(0, self.num_envs, device=self.device)

        change_flag = 1

        self.cpg.reset_gait(1, env_ids)
        T_set = self.cpg.T
        BETA_set = self.cpg.beta
        
        for i in range(num_steps):
            self.cpg.step()
            paths_raw = self.get_end_effector_ref(self.cpg.get_raw_phase())
            paths[i, : , :] = paths_raw[0]

            joint_pos_raw = self.InverseKinematics(paths_raw)
            joint_pos = joint_pos_raw[0]
            joint_vel = self.get_joint_vel()[0]
            joint_poses[i, :, :] = joint_pos
            joint_vels[i, :, :] = joint_vel
            leg_pos[i, :, :] = self.ForwardKinematics(joint_pos_raw)[0]

        return np.linspace(0, total_time, num_steps), paths, joint_poses, joint_vels, leg_pos

    def test_nocpg_trajectory(self):
        phase = torch.zeros(self.num_envs,dtype=torch.float32, device=self.device,requires_grad=False)
        dt = 0.001 * torch.ones(self.num_envs,dtype=torch.float32, device=self.device,requires_grad=False)
        z1 = []
        y1 = []
        x1 = []
        q10 = []
        q11 = []
        q12 = []
        q13 = []
        q14 = []
        q15 = []
        z2 = []
        y2 = []
        x2 = []
        for i in range(1000):
            phase = dt*i % 1.0 / 1.0
            # print(type(phase))
            # print(phase.size())
            path_raw = self.get_end_effector_ref(phase)
            joint_pos_raw = self.InverseKinematics(path_raw)
            joint_vel_raw = self.get_joint_vel(phase)
            z1.append(path_raw[0,0,0].cpu().numpy())
            y1.append(path_raw[0,1,0].cpu().numpy())
            x1.append(path_raw[0,2,0].cpu().numpy())
            z2.append(path_raw[0,0,1].cpu().numpy())
            y2.append(path_raw[0,1,1].cpu().numpy())
            x2.append(path_raw[0,2,1].cpu().numpy())
            q10.append(joint_pos_raw[0,0,0].cpu().numpy())
            q11.append(joint_pos_raw[0,1,0].cpu().numpy())
            q12.append(joint_pos_raw[0,2,0].cpu().numpy())
            q13.append(joint_pos_raw[0,3,0].cpu().numpy())
            q14.append(joint_pos_raw[0,4,0].cpu().numpy())
            q15.append(joint_pos_raw[0,5,0].cpu().numpy())
            # print(path_raw)
        ax = plt.axes(projection='3d')
        ax.plot3D(x1, y1, z1, 'gray')
        ax.plot3D(x2, y2, z2, 'red')
        # plt.show()

        plt.figure()
        plt.plot(q10, label='Lq0', linewidth=1)
        plt.plot(q11, label='Lq1', linewidth=1)
        plt.plot(q12, label='Lq2', linewidth=1)
        plt.plot(q13, label='Lq3', linewidth=1)
        plt.plot(q14, label='Lq4', linewidth=1)
        plt.plot(q15, label='Lq5', linewidth=1)
        plt.show()


    def test_and_plot(self, total_time):
        """
        模拟CPG并绘制结果
        total_time: 总模拟时间
        """
        time_start = time.time()
        time_step, paths, joint_poses, joint_vels, leg_pos = self.test_path_generation(total_time)
        time_end = time.time()
        print(f"time_consumption: {time_end - time_start}")

        paths = paths.cpu()
        joint_poses = joint_poses.cpu()
        joint_vels = joint_vels.cpu()
        leg_pos = leg_pos.cpu()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
        plotrange = 1000
        axs[0].plot(paths[:plotrange, 0, 0], paths[:plotrange, 1, 0], paths[:plotrange, 2, 0], label='FL', linewidth=2)
        axs[0].plot(paths[:plotrange, 0, 1], paths[:plotrange, 1, 1], paths[:plotrange, 2, 1], label='FR', linewidth=2)
       
        axs[0].set_title('output_paths')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_zlabel('z')
        axs[0].legend()
        axs[0].grid(True)

        fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].plot(joint_poses[:, 0, 0], label='Lq0', linewidth=2)
        axs[0].plot(joint_poses[:, 1, 0], label='Lq1', linewidth=2)
        axs[0].plot(joint_poses[:, 2, 0], label='Lq2', linewidth=2)
        axs[0].plot(joint_poses[:, 3, 0], label='Lq3', linewidth=2)
        axs[0].plot(joint_poses[:, 4, 0], label='Lq4', linewidth=2)
        axs[0].plot(joint_poses[:, 5, 0], label='Lq5', linewidth=2)
        axs[0].plot(joint_poses[:, 0, 1], label='Rq0', linewidth=2)
        axs[0].plot(joint_poses[:, 1, 1], label='Rq1', linewidth=2)
        axs[0].plot(joint_poses[:, 2, 1], label='Rq2', linewidth=2)
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(joint_vels[:, 0, 0], label='Ldq0', linewidth=2)
        axs[1].plot(joint_vels[:, 1, 0], label='Ldq1', linewidth=2)
        axs[1].plot(joint_vels[:, 2, 0], label='Ldq2', linewidth=2)
        axs[1].plot(joint_vels[:, 0, 1], label='Rdq0', linewidth=2)
        axs[1].plot(joint_vels[:, 1, 1], label='Rdq1', linewidth=2)
        axs[1].plot(joint_vels[:, 2, 1], label='Rdq2', linewidth=2)
        axs[1].legend()
        axs[1].grid(True)

        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        

        line1, = ax1.plot([], [], linewidth=2, label = 'FL')
        line2, = ax2.plot([], [], linewidth=2, label = 'FR')

        line11, = ax1.plot([], [], linewidth=2, label = 'FL_leg')
        line22, = ax2.plot([], [], linewidth=2, label = 'FR_leg')

        # 初始化函数
        def init():
            line1.set_data([], [])
            line1.set_3d_properties([])
            line2.set_data([], [])
            line2.set_3d_properties([])


            line11.set_data([], [])
            line11.set_3d_properties([])
            line22.set_data([], [])
            line22.set_3d_properties([])

            return line1, line2, line11, line22

        # 更新函数
        frames=len(paths[:, 0, 0])
        i = np.linspace(0, 50-1, 50)
        print(f"frames:{frames}\ni:{i}\n")

        def update(frame):
            i = np.linspace(frame, frame+30, 30)
            i = i.astype(int)
            x1 = paths[i, 0, 0]
            y1 = paths[i, 1, 0]
            z1 = paths[i, 2, 0]
            x2 = paths[i, 0, 1]
            y2 = paths[i, 1, 1]
            z2 = paths[i, 2, 1]
            
            frame = int(frame)
            print(frame+30)
            x11 = leg_pos[frame+30, 0, 0]
            y11 = leg_pos[frame+30, 1, 0]
            z11 = leg_pos[frame+30, 2, 0]
            x22 = leg_pos[frame+30, 0, 1]
            y22 = leg_pos[frame+30, 1, 1]
            z22 = leg_pos[frame+30, 2, 1]

            line1.set_data(x1, y1)
            line1.set_3d_properties(z1)
            line2.set_data(x2, y2)
            line2.set_3d_properties(z2)

            line11.set_data(x11, y11)
            line11.set_3d_properties(z11)
            line22.set_data(x22, y22)
            line22.set_3d_properties(z22)

            return line1, line2, line11, line22
        
        ax1.set_xlim(-.5, .5)
        ax1.set_ylim(-.5, .5)
        ax1.set_zlim(-1.0, -0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlim(-.5, .5)
        ax2.set_ylim(-.5, .5)
        ax2.set_zlim(-1.0, -0.8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.legend()
        ax2.grid(True)

        ani = FuncAnimation(fig, update, frames=np.linspace(0, len(paths[:, 0, 0])-31, len(paths[:, 0, 0]-30)), init_func=init, blit=True)
        
        plt.show()

if __name__ == '__main__':
    trajectory_generator = TrajectoryGenerator(vx=0.5, vy=0.0, wz=0.0, base_height=0.762, swing_height=0.038, stance_length=0.0, stance_width=0.279, num_envs=1,device='cpu')
    trajectory_generator.test_nocpg_trajectory()