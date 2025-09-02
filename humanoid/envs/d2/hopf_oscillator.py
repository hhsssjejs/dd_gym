import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class HOPF_CPG:
    def __init__(self, num_envs, device):
        """
        初始化CPG模型参数
        alpha1: 收敛系数, 越大收敛越快, 但越振荡, 小则反之(这里用alpha1和alpha2分别表示cos和sin上的收敛系数, 但取值相同)
        mu: 信号幅值, 输出轨迹的半径
        gamma: 角速度, zjutransitionh论文中用的是gamma
        """
        self.num_envs = num_envs
        self.device = device

        # 振荡器参数
        self.mu = 1.0
        self.alpha1 = 50.0
        self.alpha2 = 50.0
        self.b = 50.0 # stance-swing transfer speed
        self.dt = 0.01 # time step probably
        self.k = 0.5 # transition时的超调强度

        # stance period ratio
        self.BETA = torch.tensor([0.75, 0.5, 0.5, 0.25], device=self.device, requires_grad=False)
        # 耦合强度
        self.SIGMA = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, requires_grad=False)
        # cycle time [s] for CPG
        self.CYCLE_TIME = torch.tensor([0.6, 0.67, 0.5, 0.66], device=self.device, requires_grad=False)

        # gait related index, and hold leg index (-1 for nothing)
        self.gait = -1 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.previous_gait = -1 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.target_gait = -1 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        # negative for two legs, positive for one leg
        self.hold_leg = -1 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)

        # walk（漫步），trot（小跑），pace（踱步），bound（奔跑）
        self.OFFSET = torch.zeros(4, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.OFFSET[0, :] = torch.tensor([0.0, np.pi], device=self.device, requires_grad=False)
        self.OFFSET[1, :] = torch.tensor([0.0, np.pi], device=self.device, requires_grad=False)
        self.OFFSET[2, :] = torch.tensor([0.0, np.pi], device=self.device, requires_grad=False)
        self.OFFSET[3, :] = torch.tensor([0.0, np.pi], device=self.device, requires_grad=False)
        
        self.R_mat = torch.zeros(self.num_envs, 4, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.f_mat = torch.zeros(self.num_envs, 4, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset = self.OFFSET[-1, :].clone() * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.T = self.CYCLE_TIME[-1] * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.beta = self.BETA[-1] * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.sigma = self.SIGMA[-1] * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.q = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.q_dot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        self.phase_raw = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gamma = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.r_square = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

    def update_R_mat(self, env_ids):
        self.R_mat[env_ids, :, :] = 0.0
        for i in range(2):
            for j in range(2):
                theta = self.offset[env_ids, j] - self.offset[env_ids, i]
                self.R_mat[env_ids, 2*j, 2*i] = torch.cos(theta)
                self.R_mat[env_ids, 2*j, 2*i+1] = -torch.sin(theta)
                self.R_mat[env_ids, 2*j+1, 2*i] = torch.sin(theta)
                self.R_mat[env_ids, 2*j+1, 2*i+1] = torch.cos(theta)

    # smoothly transfer the gait by changing the R matrix little by little
    def update_R_mat_transition(self, init_gait, target_gait, env_ids):
        self.R_mat[env_ids, :, :] = 0.0
        self.update_phase()

        control_variable = self.get_control_variable(init_gait, target_gait)

        for i in range(2):
            for j in range(2):
                target_theta = self.OFFSET[target_gait.to(torch.long), j] - self.OFFSET[target_gait.to(torch.long), i]
                init_theta = self.OFFSET[init_gait.to(torch.long), j] - self.OFFSET[init_gait.to(torch.long), i]

                eta = (self.OFFSET[target_gait.to(torch.long), control_variable.to(torch.long)] - self.phase[env_ids.to(torch.long), control_variable.to(torch.long)]) / \
                        (self.OFFSET[target_gait.to(torch.long), control_variable.to(torch.long)] - self.OFFSET[init_gait.to(torch.long), control_variable.to(torch.long)])

                theta = target_theta + self.k * eta * (target_theta - init_theta)
                
                self.R_mat[env_ids, 2*j, 2*i] = torch.cos(theta)
                self.R_mat[env_ids, 2*j, 2*i+1] = -torch.sin(theta)
                self.R_mat[env_ids, 2*j+1, 2*i] = torch.sin(theta)
                self.R_mat[env_ids, 2*j+1, 2*i+1] = torch.cos(theta)

    # get the latest relative phases when doing phase transfer 
    def update_phase(self):
        for i in range(2):
            self.phase[:, i] = torch.arctan2(self.q[:, 2*i+1], self.q[:, 2*i])
        for i in range(2):
            self.phase[:, i] -= self.phase[:, 0]
            self.phase[:, i] = torch.where(self.phase[:, i] < 0, self.phase[:, i] + 2*torch.pi, self.phase[:, i])
        self.phase[:, 0] = 0.0

    # get the index where phase hasn't reach the desired value during transfer
    # 选出任意一个相位差来计算eta，用于transition中所有的theta计算
    def get_control_variable(self, init_gait, target_gait):
        p = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        threshold = 0.1
        diff = self.OFFSET[target_gait.to(torch.long)] - self.OFFSET[init_gait.to(torch.long)]
        mask = torch.abs(diff) > threshold
        p = mask.to(torch.int)
        p = p.argmax(dim=-1)

        return p
    
    # get raw phase
    def get_raw_phase(self):
        self.phase_raw[:, :] = 0.0
        for i in range(2):
            self.phase_raw[:, i] = torch.arctan2(self.q[:, 2*i+1], self.q[:, 2*i])

        return self.phase_raw
    
    # unused function, skipthat, gamma has been calculated in step function
    def get_gamma(self):
        self.gamma[:, :] = 0.0
        for i in range(2):
            self.gamma[:, i] = (-torch.pi / self.T * (1.0 / self.beta / (1.0 + torch.exp(-self.b * self.q[:, 2*i+1])) + 1.0 / (1.0 - \
                self.beta) / (1.0 + torch.exp(self.b * self.q[:, 2*i+1]))))
            mask = self.gait < 0
            self.gamma[mask, i] = 0.0

        return self.gamma
    
    # reset everything
    def reset(self, env_ids):
        self.q[env_ids, :] = torch.tensor([1.0, 0.0, -1.0, 0.0], device=self.device, requires_grad=False)
        self.hold_leg[env_ids,] = -1
        self.reset_gait(-1, env_ids)

    def reset_gait(self, gait, env_ids):
        self.gait[env_ids] = gait
        self.previous_gait[env_ids] = gait
        self.target_gait[env_ids] = gait

        self.beta[env_ids] = self.BETA[gait].clone()
        self.sigma[env_ids] = self.SIGMA[gait].clone()
        self.T[env_ids] = self.CYCLE_TIME[gait].clone()
        self.offset[env_ids, :] = self.OFFSET[gait, :].clone()

        if gait >= 0:
            self.phase[env_ids, :] = self.offset[env_ids, :].clone()
            for i in range(2):
                self.q[env_ids, 2*i] = torch.cos(self.phase[env_ids, i])
                self.q[env_ids, 2*i+1] = torch.sin(self.phase[env_ids, i])
            self.update_R_mat(env_ids)
        else:
            self.q[env_ids, :] = torch.tensor([1.0, 0.0, -1.0, 0.0], device=self.device, requires_grad=False)
            self.q_dot[env_ids, :] = torch.zeros(4, device=self.device, requires_grad=False)
            self.phase_raw[env_ids, :] = np.pi/2.0 * torch.ones(2, device=self.device, requires_grad=False)
            self.phase[env_ids, :] = torch.zeros(2, device=self.device, requires_grad=False)
            print("-1 gait!!!!!!")
            
    # no change the gait, just reset env_ids
    # def reset_legs(self, env_ids):
    #     # self.gait[env_ids] = gait
    #     # self.previous_gait[env_ids] = gait
    #     # self.target_gait[env_ids] = gait

    #     # self.beta[env_ids,:] = self.BETA[gait].clone()
    #     # self.sigma[env_ids,:] = self.SIGMA[gait].clone()
    #     # self.T[env_ids,:] = self.CYCLE_TIME[gait].clone()
    #     # self.offset[env_ids,:] = self.OFFSET[gait,:].clone()

    #     # if gait >= 0:
    #     self.q[env_ids, :] = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0], device=self.device, requires_grad=False)
    #     self.hold_leg[env_ids,] = -1
    #     self.phase[env_ids,:] = self.offset[env_ids,:].clone()

    #     for i in range(2):
    #         self.q[env_ids,2*i] = torch.cos(self.phase[env_ids,i])
    #         self.q[env_ids,2*i+1] = torch.sin(self.phase[env_ids,i])

    #     self.update_R_mat(env_ids)

    # call this function when you wanna do gait transfer
    def change_gait(self, gait, env_ids):
        if self.hold_leg[0] < 0:
            condition_previous_gait = self.previous_gait[env_ids] < 0
            condition_gait = self.gait[env_ids] < 0
            condition_target_gait = self.target_gait[env_ids] < 0

            combined_condition = condition_previous_gait | condition_gait | condition_target_gait
            env_ids_reset = env_ids[combined_condition]
            env_ids_update = env_ids[~combined_condition]

            self.reset_gait(gait, env_ids_reset)

            self.previous_gait[env_ids_update] = self.target_gait[env_ids_update]
            self.target_gait[env_ids_update] = gait
            self.gait[env_ids_update] = gait

            self.beta[env_ids_update] = self.BETA[gait]
            self.sigma[env_ids_update] = self.SIGMA[gait]
            self.T[env_ids_update] = self.CYCLE_TIME[gait]
            self.offset[env_ids_update] = self.OFFSET[gait, :].clone()

            self.update_R_mat(env_ids)

    # a step to calculate
    def step_(self, env_ids):
        for i in range(2):
            self.r_square[env_ids, i] = torch.square(self.q[env_ids, 2*i]) + torch.square(self.q[env_ids, 2*i+1])

        self.f_mat[env_ids, :, :] = 0.0
        
        for i in range(2):
            self.f_mat[env_ids, 2*i, 2*i] = self.alpha1 * (self.mu**2 - self.r_square[env_ids, i])
            self.f_mat[env_ids, 2*i+1, 2*i+1] = self.alpha2 * (self.mu**2 - self.r_square[env_ids, i])
            
            gamma = torch.pi / self.T[env_ids] * (1.0 / self.beta[env_ids]/ (1.0 + torch.exp(-self.b * self.q[env_ids, 2*i+1])) +\
                 1.0 / (1.0 - self.beta[env_ids]) / (1.0 + torch.exp(self.b * self.q[env_ids, 2*i+1])))
 
            self.f_mat[env_ids, 2*i, 2*i+1] = gamma
            self.f_mat[env_ids, 2*i+1, 2*i] = -gamma

        self.q_dot = self.f_mat @ self.q.unsqueeze(-1) + self.R_mat @ self.q.unsqueeze(-1) * self.sigma.unsqueeze(-1).unsqueeze(-1)
        self.q_dot = self.q_dot.squeeze(-1)

        self.q[env_ids] = self.q[env_ids] + self.q_dot[env_ids] * self.dt

    # gait tranfer check, wrapped step function
    def step(self):
        # self.dt += random.uniform(-self.dt, self.dt)
        env_ids = torch.tensor(range(self.num_envs), device=self.device).to(torch.long)
        condition = self.gait[env_ids] != self.previous_gait[env_ids]
        env_ids = env_ids[condition]

        if not env_ids.numel() == 0:
            self.update_R_mat_transition(self.previous_gait[env_ids], self.target_gait[env_ids], env_ids)
        control_variable = self.get_control_variable(self.previous_gait, self.target_gait)

        condition = abs(self.phase[env_ids, control_variable[env_ids].to(torch.long)] - self.OFFSET[self.target_gait[env_ids].to(torch.long), control_variable[env_ids].to(torch.long)]) < 0.1 * np.pi

        if not env_ids.numel() == 0:
            env_ids = env_ids[condition]
            if not env_ids.numel() == 0:
                self.previous_gait[env_ids] = self.gait[env_ids]
                self.update_R_mat(env_ids)

        # if self.gait >= 0:
        env_ids = (self.gait >= 0).nonzero()
        self.step_(env_ids)

    # get transition status, is it gonna tranfer or not
    def get_transition_status(self):
        if self.gait != self.previous_gait:
            return True
        else:
            return False
    
    # get output
    def get_position(self):
        return self.q
    
    # get velocity of output
    def get_velocity(self):
        return self.q_dot
    
    # get leg stance time
    def get_stance_time(self):
        if self.hold_leg[0] < 0:
            return self.T * self.beta
    
    # get current gait index
    def get_gait_index(self):
        return self.gait
    
    # get hold leg index in one leg gait (negative for nothing)
    def get_hold_leg(self):
        return self.hold_leg
            
    # step step step till the end, change gait if you want, gather output and phase
    def simulate(self, total_time):
        """
        模拟CPG一段时间
        total_time: 总模拟时间
        """
        dt = self.dt
        num_steps = int(total_time / dt)
        positions = torch.zeros((num_steps, 4), device=self.device)
        phases = torch.zeros((num_steps, 2), device=self.device)
       
        env_ids = torch.tensor(range(self.num_envs), device=self.device)
        self.reset_gait(3, env_ids)

        change_flag = 1
        
        for i in range(num_steps):
            if i > num_steps*3/4:
                if change_flag == 3:
                    self.change_gait(1, env_ids)
                    change_flag = 1
            elif i > num_steps/2:
                if change_flag == 2:
                    self.change_gait(3, env_ids)
                    change_flag = 3
            elif i > num_steps/4:
                if change_flag == 1:
                    self.change_gait(1, env_ids)
                    change_flag = 2
            self.step()
            positions[i, :] = self.get_position()[0]
            phases[i, :] = self.get_raw_phase()[0]

        phases = phases / np.pi

        return positions, phases, np.linspace(0, total_time, num_steps)

    # simulate and plot the output and phase
    def simulate_and_plot(self, total_time):
        """
        模拟CPG并绘制结果
        total_time: 总模拟时间
        """
        positions, phases, _ = self.simulate(total_time)
        positions = positions.cpu()
        phases = phases.cpu()
        fig, axs = plt.subplots(3, 1, figsize=(12, 5))
        axs[0].plot(positions[:, 0], label='FL cos', linewidth=2)
        axs[0].plot(positions[:, 1], label='FL sin', linewidth=2, linestyle='--')
        axs[0].plot(positions[:, 2], label='FR cos', linewidth=2)
        axs[0].plot(positions[:, 3], label='FR sin', linewidth=2, linestyle='--')
        axs[0].set_title('position output Plot of Hopf (CPG)')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('position')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(phases[:, 0], label='FL', linewidth=2)
        axs[1].plot(phases[:, 1], label='FR', linewidth=2)
        axs[1].set_title('Phase Space Plot of Hopf (CPG)')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Phase')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(positions[:, 0], positions[:, 1], label='FL', linewidth=2)
        axs[2].plot(positions[:, 2], positions[:, 3], label='FR', linewidth=2)
        axs[2].set_title('trajectory Plot of Hopf (CPG)')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('y')
        axs[2].legend()
        axs[2].grid(True)
        plt.show()

# 使用模型并绘制结果
if __name__ == '__main__':
    cpg_model = HOPF_CPG(num_envs=10, device='cuda:0')
    cpg_model.simulate_and_plot(total_time=40.0)