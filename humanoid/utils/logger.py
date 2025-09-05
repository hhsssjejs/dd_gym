import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import scipy.io

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()



    def save_log_to_mat(self, filename="log_data.mat"):
        """
        Saves the logged state and reward data to a .mat file for MATLAB visualization.
        
        Parameters:
            filename (str): The name of the file to save the data.
        """
        data_to_save = {
            "state_log": {key: np.array(value) for key, value in self.state_log.items()},
            "rew_log": {key: np.array(value) for key, value in self.rew_log.items()},
            "dt": self.dt,
            "num_episodes": self.num_episodes
        }
        
        scipy.io.savemat(filename, data_to_save)
        print(f"Log data saved to {filename}")


    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process1 = Process(target=self._plot_position)
        # self.plot_process2 = Process(target=self._plot_torque)
        # self.plot_process3 = Process(target=self._plot_vel)
        # self.plot_process4 = Process(target=self._plot_tn_rms)
        # self.plot_process5 = Process(target=self._plot_tn)
        # self.plot_process6 = Process(target=self._plot_torque_vel)
        # self.plot_process7 = Process(target=self._plot_position1)
        # self.plot_process8 = Process(target=self._plot_torque1)
        # self.plot_process9 = Process(target=self._plot_vel1)
        # self.plot_process10 = Process(target=self._plot_tn_rms1)
        # self.plot_process11 = Process(target=self._plot_tn1)
        # self.plot_process12 = Process(target=self._plot_torque_vel1)
        self.plot_process.start()
        # self.plot_process1.start()
        # self.plot_process2.start()
        # self.plot_process3.start()
        # self.plot_process4.start()
        # self.plot_process5.start()
        # self.plot_process6.start()
        # self.plot_process7.start()
        # self.plot_process8.start()
        # self.plot_process9.start()
        # self.plot_process10.start()
        # self.plot_process11.start()
        # self.plot_process12.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        # if log["command_sin"]: a.plot(time, log["command_sin"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        # a = axs[0, 1]
        # if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        # if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        # # if log["command_cos"]: a.plot(time, log["command_cos"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        # a.legend()
        # # plot base vel yaw
        # a = axs[0, 2]
        # if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        # a.legend()
        a = axs[0, 1]
        if log["foot_z_l"]: a.plot(time, log["foot_z_l"], label='measured')
        # if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        # if log["command_cos"]: a.plot(time, log["command_cos"], label='commanded')
        a.set(xlabel='time [s]', ylabel='foot_z_l [m]', title='foot_z_l')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["foot_z_r"]: a.plot(time, log["foot_z_r"], label='measured')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='foot_z_r [m]', title='foot_z_r')
        a.legend()
        # # plot base vel z
        # a = axs[1, 2]
        # if log["command_sin"]: a.plot(time, log["command_sin"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='command sin', title='Command Sin')
        # a.legend()
        # # plot contact forces
        a = axs[2, 0]
        if log["base_height"]: a.plot(time, log["base_height"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base_height', title='base_height')
        a.legend()

        # a = axs[2, 1]
        # if log["foot_forcez_l"]: a.plot(time, log["foot_forcez_l"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='foot_forcez_l', title='foot_forcez_l')
        # a.legend()

        # a = axs[2, 2]
        # if log["foot_forcez_r"]: a.plot(time, log["foot_forcez_r"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='foot_forcez_r', title='foot_forcez_r')
        # a.legend()

        a = axs[2, 1]
        if log["dof_torque_6"]: a.plot(time, log["dof_torque_6"], label='commanded')
        a.set(xlabel='time [s]', ylabel='dof_torque_6', title='dof_torque_6')
        a.legend()

        a = axs[2, 2]
        if log["dof_torque_12"]: a.plot(time, log["dof_torque_12"], label='commanded')
        a.set(xlabel='time [s]', ylabel='dof_torque_12', title='dof_torque_12')
        a.legend()

        a = axs[1, 0]
        if log["dof_torque_4"]: a.plot(time, log["dof_torque_4"], label='commanded')
        a.set(xlabel='time [s]', ylabel='dof_torque_4', title='dof_torque_4')
        a.legend()

        a = axs[1, 1]
        if log["dof_torque_11"]: a.plot(time, log["dof_torque_11"], label='commanded')
        a.set(xlabel='time [s]', ylabel='dof_torque_11', title='dof_torque_11')
        a.legend()

        # a = axs[1, 0]
        # if log["dof_pos_6"]: a.plot(time, log["dof_pos_6"], label='measured')
        # a.set(xlabel='time [s]', ylabel='foot height', title='left foot height')
        # a.legend()
        # # plot contact forces
        # a = axs[1, 1]
        # if log["dof_pos_13"]: a.plot(time, log["dof_pos_13"], label='measured')
        # a.set(xlabel='time [s]', ylabel='foot height', title='right foot height')
        # a.legend()
        plt.show()

    def _plot_position(self):
        nb_rows = 3
        nb_cols = 2 
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_pos_0"]: a.plot(time, log["dof_pos_0"], label='measured')
        if log["dof_pos_target_0"]: a.plot(time, log["dof_pos_target_0"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[0]')
        a.legend()

        a = axs[0, 1]
        if log["dof_pos_7"]: a.plot(time, log["dof_pos_7"], label='measured')
        if log["dof_pos_target_7"]: a.plot(time, log["dof_pos_target_7"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[7]')
        a.legend()

        a = axs[1, 0]
        if log["dof_pos_3"]: a.plot(time, log["dof_pos_3"], label='measured')
        if log["dof_pos_target_3"]: a.plot(time, log["dof_pos_target_3"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[3]')
        a.legend()

        a = axs[1, 1]
        if log["dof_pos_10"]: a.plot(time, log["dof_pos_10"], label='measured')
        if log["dof_pos_target_10"]: a.plot(time, log["dof_pos_target_10"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[10]')
        a.legend()

        a = axs[2, 0]
        if log["dof_pos_6"]: a.plot(time, log["dof_pos_6"], label='measured')
        if log["dof_pos_target_6"]: a.plot(time, log["dof_pos_target_6"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[6]')
        a.legend()

        a = axs[2, 1]
        if log["dof_pos_13"]: a.plot(time, log["dof_pos_13"], label='measured')
        if log["dof_pos_target_13"]: a.plot(time, log["dof_pos_target_13"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[13]')
        a.legend()
        plt.show()

    def _plot_position1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[1, 2]
        if log["dof_pos_13"]: a.plot(time, log["dof_pos_13"], label='measured')
        if log["dof_pos_target_13"]: a.plot(time, log["dof_pos_target_13"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[11]')
        a.legend()

        a = axs[0, 0]
        if log["dof_pos_7"]: a.plot(time, log["dof_pos_7"], label='measured')
        if log["dof_pos_target_7"]: a.plot(time, log["dof_pos_target_7"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[6]')
        a.legend()

        a = axs[0, 1]
        if log["dof_pos_8"]: a.plot(time, log["dof_pos_8"], label='measured')
        if log["dof_pos_target_8"]: a.plot(time, log["dof_pos_target_8"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[7]')
        a.legend()

        a = axs[0, 2]
        if log["dof_pos_9"]: a.plot(time, log["dof_pos_9"], label='measured')
        if log["dof_pos_target_9"]: a.plot(time, log["dof_pos_target_9"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[8]')
        a.legend()

        a = axs[1, 0]
        if log["dof_pos_10"]: a.plot(time, log["dof_pos_10"], label='measured')
        if log["dof_pos_target_10"]: a.plot(time, log["dof_pos_target_10"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[9]')
        a.legend()

        a = axs[1, 1]
        if log["dof_pos_11"]: a.plot(time, log["dof_pos_11"], label='measured')
        if log["dof_pos_target_11"]: a.plot(time, log["dof_pos_target_11"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[10]')
        a.legend()
        plt.show()

    def _plot_torque(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque_0"] != []: a.plot(time, log["dof_torque_0"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_0]')
        a.legend()
        a = axs[0, 1]
        if log["dof_torque_1"] != []: a.plot(time, log["dof_torque_1"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_2"] != []: a.plot(time, log["dof_torque_2"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_3"] != []: a.plot(time, log["dof_torque_3"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_4"] != []: a.plot(time, log["dof_torque_4"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_5"] != []: a.plot(time, log["dof_torque_5"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_5]')
        a.legend()
        plt.show()

    def _plot_torque1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque_6"] != []: a.plot(time, log["dof_torque_6"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_6]')
        a.legend()
        a = axs[0, 1]
        if log["dof_torque_7"] != []: a.plot(time, log["dof_torque_7"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_8"] != []: a.plot(time, log["dof_torque_8"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_9"] != []: a.plot(time, log["dof_torque_9"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_10"] != []: a.plot(time, log["dof_torque_10"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_11"] != []: a.plot(time, log["dof_torque_11"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='torque_11]')
        a.legend()
        plt.show()

    def _plot_vel(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_vel_0"]: a.plot(time, log["dof_vel_0"], label='measured')
        if log["dof_vel_target_0"]: a.plot(time, log["dof_vel_target_0"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[0]')
        a = axs[0, 1]
        a.legend()
        if log["dof_vel_1"]: a.plot(time, log["dof_vel_1"], label='measured')
        if log["dof_vel_target_1"]: a.plot(time, log["dof_vel_target_1"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_vel_2"]: a.plot(time, log["dof_vel_2"], label='measured')
        if log["dof_vel_target_2"]: a.plot(time, log["dof_vel_target_2"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_vel_3"]: a.plot(time, log["dof_vel_3"], label='measured')
        if log["dof_vel_target_3"]: a.plot(time, log["dof_vel_target_3"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_vel_4"]: a.plot(time, log["dof_vel_4"], label='measured')
        if log["dof_vel_target_4"]: a.plot(time, log["dof_vel_target_4"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_vel_5"]: a.plot(time, log["dof_vel_5"], label='measured')
        if log["dof_vel_target_5"]: a.plot(time, log["dof_vel_target_5"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[5]')
        a.legend()
        plt.show()

    def _plot_vel1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_vel_6"]: a.plot(time, log["dof_vel_6"], label='measured')
        if log["dof_vel_target_6"]: a.plot(time, log["dof_vel_target_6"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[6]')
        a = axs[0, 1]
        a.legend()
        if log["dof_vel_7"]: a.plot(time, log["dof_vel_7"], label='measured')
        if log["dof_vel_target_7"]: a.plot(time, log["dof_vel_target_7"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_vel_8"]: a.plot(time, log["dof_vel_8"], label='measured')
        if log["dof_vel_target_8"]: a.plot(time, log["dof_vel_target_8"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_vel_9"]: a.plot(time, log["dof_vel_9"], label='measured')
        if log["dof_vel_target_9"]: a.plot(time, log["dof_vel_target_9"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_vel_10"]: a.plot(time, log["dof_vel_10"], label='measured')
        if log["dof_vel_target_4"]: a.plot(time, log["dof_vel_target_4"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_vel_11"]: a.plot(time, log["dof_vel_11"], label='measured')
        if log["dof_vel_target_5"]: a.plot(time, log["dof_vel_target_11"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity[11]')
        a.legend()
        plt.show()

    def _plot_tn_rms(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        a = axs[0, 0]
        if log["dof_torque_0"] != [] and log["dof_vel_0"] != []:
            vel_array = np.array(log["dof_vel_0"])
            torque_array = np.array(log["dof_torque_0"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[0]')
        a.legend()

        a = axs[0, 1]
        if log["dof_torque_1"] != [] and log["dof_vel_1"] != []:
            vel_array = np.array(log["dof_vel_1"])
            torque_array = np.array(log["dof_torque_1"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_1"] != [] and log["dof_vel_1"] != []:
            vel_array = np.array(log["dof_vel_1"])
            torque_array = np.array(log["dof_torque_1"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_3"] != [] and log["dof_vel_3"] != []:
            vel_array = np.array(log["dof_vel_3"])
            torque_array = np.array(log["dof_torque_3"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_4"] != [] and log["dof_vel_4"] != []:
            vel_array = np.array(log["dof_vel_4"])
            torque_array = np.array(log["dof_torque_4"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_5"] != [] and log["dof_vel_5"] != []:
            vel_array = np.array(log["dof_vel_5"])
            torque_array = np.array(log["dof_torque_5"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[5]')
        a.legend()
        plt.show()

    def _plot_tn_rms1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        a = axs[0, 0]
        if log["dof_torque_6"] != [] and log["dof_vel_6"] != []:
            vel_array = np.array(log["dof_vel_6"])
            torque_array = np.array(log["dof_torque_6"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[6]')
        a.legend()
        a = axs[0, 1]
        if log["dof_torque_7"] != [] and log["dof_vel_7"] != []:
            vel_array = np.array(log["dof_vel_7"])
            torque_array = np.array(log["dof_torque_7"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_8"] != [] and log["dof_vel_8"] != []:
            vel_array = np.array(log["dof_vel_8"])
            torque_array = np.array(log["dof_torque_8"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_9"] != [] and log["dof_vel_9"] != []:
            vel_array = np.array(log["dof_vel_9"])
            torque_array = np.array(log["dof_torque_9"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_10"] != [] and log["dof_vel_10"] != []:
            vel_array = np.array(log["dof_vel_10"])
            torque_array = np.array(log["dof_torque_10"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_11"] != [] and log["dof_vel_11"] != []:
            vel_array = np.array(log["dof_vel_11"])
            torque_array = np.array(log["dof_torque_11"])

            rms_vel = np.sqrt(np.mean(vel_array ** 2))
            rms_torque = np.sqrt(np.mean(torque_array ** 2))
            a.plot(rms_vel, rms_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN_RMS[11]')
        a.legend()
        plt.show()

    def _plot_tn(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        a = axs[0, 0]
        if log["dof_torque_0"] != [] and log["dof_vel_0"] != []:
            vel_array = np.array(log["dof_vel_0"])
            torque_array = np.array(log["dof_torque_0"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[0]')
        a.legend()

        a = axs[0, 1]
        if log["dof_torque_1"] != [] and log["dof_vel_1"] != []:
            vel_array = np.array(log["dof_vel_1"])
            torque_array = np.array(log["dof_torque_1"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_1"] != [] and log["dof_vel_1"] != []:
            vel_array = np.array(log["dof_vel_1"])
            torque_array = np.array(log["dof_torque_1"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_3"] != [] and log["dof_vel_3"] != []:
            vel_array = np.array(log["dof_vel_3"])
            torque_array = np.array(log["dof_torque_3"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_4"] != [] and log["dof_vel_4"] != []:
            vel_array = np.array(log["dof_vel_4"])
            torque_array = np.array(log["dof_torque_4"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_5"] != [] and log["dof_vel_5"] != []:
            vel_array = np.array(log["dof_vel_5"])
            torque_array = np.array(log["dof_torque_5"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[5]')
        a.legend()
        plt.show()

    def _plot_tn1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        a = axs[0, 0]
        if log["dof_torque_6"] != [] and log["dof_vel_6"] != []:
            vel_array = np.array(log["dof_vel_6"])
            torque_array = np.array(log["dof_torque_6"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[6]')
        a.legend()
        a = axs[0, 1]
        if log["dof_torque_7"] != [] and log["dof_vel_7"] != []:
            vel_array = np.array(log["dof_vel_7"])
            torque_array = np.array(log["dof_torque_7"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_8"] != [] and log["dof_vel_8"] != []:
            vel_array = np.array(log["dof_vel_8"])
            torque_array = np.array(log["dof_torque_8"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_9"] != [] and log["dof_vel_9"] != []:
            vel_array = np.array(log["dof_vel_9"])
            torque_array = np.array(log["dof_torque_9"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_10"] != [] and log["dof_vel_10"] != []:
            vel_array = np.array(log["dof_vel_10"])
            torque_array = np.array(log["dof_torque_10"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_11"] != [] and log["dof_vel_11"] != []:
            vel_array = np.array(log["dof_vel_11"])
            torque_array = np.array(log["dof_torque_11"])

            abs_vel = np.abs(vel_array)
            abs_torque = np.abs(torque_array)
            a.plot(abs_vel, abs_torque, '*', label='measured')
        a.set(xlabel='Velocity [rad/s]', ylabel='Joint Torque [Nm]', title='TN[11]')
        a.legend()
        plt.show()

    def _plot_torque_vel(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque_0"] != []: a.plot(time, log["dof_torque_0"], label='measured_torque')
        if log["dof_vel_0"]: a.plot(time, log["dof_vel_0"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_0]')
        a.legend()
        a = axs[0, 1]
        if log["dof_torque_1"] != []: a.plot(time, log["dof_torque_1"], label='measured_torque')
        if log["dof_vel_1"]: a.plot(time, log["dof_vel_1"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_1]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_2"] != []: a.plot(time, log["dof_torque_2"], label='measured_torque')
        if log["dof_vel_2"]: a.plot(time, log["dof_vel_2"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_2]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_3"] != []: a.plot(time, log["dof_torque_3"], label='measured_torque')
        if log["dof_vel_3"]: a.plot(time, log["dof_vel_3"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_3]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_4"] != []: a.plot(time, log["dof_torque_4"], label='measured_torque')
        if log["dof_vel_4"]: a.plot(time, log["dof_vel_4"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_4]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_5"] != []: a.plot(time, log["dof_torque_5"], label='measured_torque')
        if log["dof_vel_5"]: a.plot(time, log["dof_vel_5"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_5]')
        a.legend()
        plt.show()

    def _plot_torque_vel1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque_6"] != []: a.plot(time, log["dof_torque_6"], label='measured_torque')
        if log["dof_vel_6"]: a.plot(time, log["dof_vel_6"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_6]')
        a.legend()
        a = axs[0, 1]
        if log["dof_torque_7"] != []: a.plot(time, log["dof_torque_7"], label='measured_torque')
        if log["dof_vel_7"]: a.plot(time, log["dof_vel_7"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_7]')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque_8"] != []: a.plot(time, log["dof_torque_8"], label='measured_torque')
        if log["dof_vel_8"]: a.plot(time, log["dof_vel_8"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_8]')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque_9"] != []: a.plot(time, log["dof_torque_9"], label='measured_torque')
        if log["dof_vel_9"]: a.plot(time, log["dof_vel_9"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_9]')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque_10"] != []: a.plot(time, log["dof_torque_10"], label='measured_torque')
        if log["dof_vel_10"]: a.plot(time, log["dof_vel_10"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_10]')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque_11"] != []: a.plot(time, log["dof_torque_11"], label='measured_torque')
        if log["dof_vel_11"]: a.plot(time, log["dof_vel_11"], label='measured_vel')
        a.set(xlabel='time [s]', ylabel=' ', title='Velocity torque_11]')
        a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
