from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class D2DHStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66  # all histroy obs num
        short_frame_stack = 5  # short history step
        c_frame_stack = 3  # all history privileged obs num
        num_single_obs = 47  # 5+14+14+14+3+3
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 78  # 5+14+14+14+14+3+3+3+2+3+1+1+2+2

        # cmd, pos, vel, action, diff, lin_vel, ang_vel, rpy, push_force, push_torque, env_friction, body_mass, stand_mask, contact_mask
        # 5, 14, 14 ,14, 14, 3, 3, 3, 2, 3, 1, 1, 2, 2
        single_linvel_index = 53  # 5 + 14 + 14 + 14 + 14
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False
        num_commands = 5  # sin_pos cos_pos vx vy vz

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.9

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/d2/urdf/d2.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/d2/mjcf/scene.xml'

        name = "d2"
        foot_name = '6_link'
        knee_name = '4_link'

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
        terminate_projected_gravity = -0.8

        rotor_inertia = [
            0.331776,  # LEFT LEG
            0.331776,
            0.036864,
            0.331776,
            0.0101088,
            0.0101088,
            # 0.036864,
            0.331776,  # RIGHT LEG
            0.331776,
            0.036864,
            0.331776,
            0.0101088,
            0.0101088,
            # 0.036864,
        ]

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = True
        static_friction = 0.6  #
        dynamic_friction = 0.6  #
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        terrain_dict = {"flat": 1.0,
                        "rough flat": 0.0,
                        "slope up": 0.0,
                        "slope down": 0.0,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "stairs up": 0.,
                        "stairs down": 0.,
                        "discrete": 0.0,
                        "wave": 0.0, }
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]  # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 1.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.06]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # "left_shoulder_pitch_joint": 0.2,
            # "right_shoulder_pitch_joint": 0.2,
            'leg_l1_joint': 0.0,
            'leg_l2_joint': 0.0,
            'leg_l3_joint': -0.05,
            'leg_l4_joint': 0.2,
            'leg_l5_joint': -0.15,
            'leg_l6_joint': 0.0,
            'leg_r1_joint': 0.0,
            'leg_r2_joint': 0.0,
            'leg_r3_joint': -0.05,
            'leg_r4_joint': 0.2,
            'leg_r5_joint': -0.15, 
            'leg_r6_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        # stiffness = {
        #     '1_joint': 400, '2_joint': 180, '3_joint': 100,
        #     '4_joint': 100, '5_joint': 100, '6_joint': 50}
        # damping = {
        #     '1_joint': 45, '2_joint': 20, '3_joint': 10,
        #     '4_joint': 10, '5_joint': 4, '6_joint': 4}
        
        stiffness = {'1_joint': 800., '2_joint': 300.,'3_joint': 200.,
                    '4_joint': 200., '5_joint': 200., '6_joint': 100.}
        damping = {'1_joint':90., '2_joint': 40.,'3_joint': 20., 
                    '4_joint': 20., '5_joint': 7., '6_joint': 7.}

        # action scale: target_angle = action_scale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0., 1.0]  # [0., 1.3]
        restitution_range = [0.0, 0.4]
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

        # push
        push_robots = True
        push_interval_s = 4  # every this second, push robot
        update_step = 1000 * 24  # after this count, increase push_duration index
        push_duration = [0, 0.001]  # increase push duration during training
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.5

        add_ext_force = True
        ext_force_max_xy = 150
        ext_force_max_z = 30
        ext_torque_max = 12
        ext_force_interval_s = 8
        add_update_step = 1000 * 24
        add_duration = [0, 0.2, 0.6, 1., 1.4]

        randomize_base_mass = True
        added_mass_range = [-2, 2]  # base mass rand range, base mass is all fix link sum mass

        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]  # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = True
        joint_armature_range = [0.0001, 0.05]  # Factor
        joint_1_armature_range = [-0.0331776, 0.0331776]
        joint_2_armature_range = [-0.0331776, 0.0331776]
        joint_3_armature_range = [-0.0036864, 0.0036864]
        joint_4_armature_range = [-0.0331776, 0.0331776]
        joint_5_armature_range = [-0.00101088, 0.00101088]
        joint_6_armature_range = [-0.00101088, 0.00101088]
        joint_7_armature_range = [-0.0331776, 0.0331776]
        joint_8_armature_range = [-0.0331776, 0.0331776]
        joint_9_armature_range = [-0.0036864, 0.0036864]
        joint_10_armature_range = [-0.0331776, 0.0331776]
        joint_11_armature_range = [-0.00101088, 0.00101088]
        joint_12_armature_range = [-0.00101088, 0.00101088]

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]

        add_dof_lag = True  # 这个是接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟一样
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False  # 不常用always False， 每一次接收到的信号都随机延迟
        dof_lag_timesteps_range = [0, 40]  # lag index, *dt is time lag

        add_dof_pos_vel_lag = False  # 这个是接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟不同
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False  # 不常用always False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False  # 不常用always False
        dof_vel_lag_timesteps_range = [7, 25]

        add_imu_lag = False  # 这个是 imu 的延迟
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False  # 不常用always False
        imu_lag_timesteps_range = [1, 10]

        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 25.  # time before command are changed[s]

        gait = ["walk_sagittal", "stand", "walk_omnidirectional", "walk_omnidirectional"]  # gait type during training
        # not time, is proportion during whole life time
        gait_time_range = {"walk_sagittal": [2, 6],
                           "walk_lateral": [2, 6],
                           "rotate": [2, 3],
                           "stand": [2, 3],
                           "walk_omnidirectional": [4, 6]}
        # stand_time = 18.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        stand_com_threshold = 0.05  # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True  # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [-0.5, 1.0]  # [-0.4, 1.0]  #[-0.8, 1.] # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 1.025
        foot_min_dist = 0.24
        foot_max_dist = 1.0
        use_ref_ik = False

        #  X + 0.2    Z + 0.07    30度
        # final_swing_joint_delta_pos = [0.652573,     0.274948,    -0.351518 ,    0.304196  ,  -0.215858 ,-0.000459622,       # left leg
        #                                 0.3,                                                   # left arm    摆动为正

        #                                -0.652576,   -0.274948,    0.351517 ,   0.304197 ,  -0.215859 ,0.000468325,        # right leg
        #                                 0.3    ]                                                # right arm

        final_swing_joint_delta_pos = [0.0, 0.0, -0.32, 0.81, -0.35, 0,  # left leg

                                       0.0, 0.0, -0.32, 0.81, -0.35, 0,  # right leg
                                        ]  # right arm

        target_feet_height = 0.055  # m  0.047 / 0.07 / 0.035
        target_feet_height_max = 0.065  # 0.065  / 0.09
        feet_to_ankle_distance = 0.054

        stance_ratio = 0.5
        cycle_time = 0.8  # sec ??
        cycle_time_range = [0.6, 1.1]  # 速度0 为最大值   速度最大为最小值
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 600  # 1400  #850  #700  # forces above this value are penalized

        class scales:
            ref_joint_pos = 2.2
            # ref_joint_pos_arm = 0.7
            feet_swing = 2.0
            # ref_joint_vel = 0.2
            feet_clearance = 1.
            feet_contact_number = 2.0
            gait_feet_frc_perio = 1.0
            gait_feet_spd_perio = 1.0
            # tracking_contacts_shaped_vel_and_force = 1.0
            # gait
            # feet_air_time = 1.2
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            # contact 
            feet_contact_forces = -0.002
            # vel tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            vel_mismatch_exp = 0.8  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # stand_still = 5
            # base pos
            default_joint_pos = 1.0
            orientation = 1.
            feet_rotation = 0.3
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.03
            torques = -8e-9
            # ankle_torques = -8e-8
            # feet_stumble = -1.0
            dof_vel = -2e-8
            dof_acc = -1e-7
            collision = -1.
            stand_still = 2.5
            # limits
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1

            # feet_height_tracking = 0.8   
            # feet_vel_tracking = 0.4 

            # base_heading = 1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.


class D2DHStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims = [256, 128, 64]

        # for long_history cnn only
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64  # long history output dim
        in_channels = D2DHStandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        # grad_penalty_coef_schedule = [0.000001, 0.000001, 700, 1400]
        grad_penalty_coef_schedule = [0.0, 0.0, 700, 1400]

        # if D2DHStandCfg.terrain.measure_heights:
        #     lin_vel_idx = (D2DHStandCfg.env.single_num_privileged_obs + D2DHStandCfg.terrain.num_height) * (D2DHStandCfg.env.c_frame_stack - 1) + D2DHStandCfg.env.single_linvel_index
        # else:
        lin_vel_idx = D2DHStandCfg.env.single_num_privileged_obs * (D2DHStandCfg.env.c_frame_stack - 1) + D2DHStandCfg.env.single_linvel_index

        sym_loss = True
        obs_permutation = [-0.0001, -1, 2, -3, -4,
                           11, -12, -13, 14, 15, -16,  5, -6, -7, 8, 9, -10, 
                           23, -24, -25, 26, 27, -28,  17, -18, -19, 20, 21, -22, 
                           35, -36, -37, 38, 39, -40,  29, -30, -31, 32, 33, -34, 
                           -41, 42, -43, -44, 45, -46]
        act_permutation = [6, -7, -8, 9, 10, -11,  0.0001, -1, -2, 3, 4, -5]
        frame_stack = D2DHStandCfg.env.frame_stack
        sym_coef = 1.0

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 20000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'd2'

        # load and resume
        resume = False
        load_run = "2025-02-07_17-07-40"  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        run_name = ""
