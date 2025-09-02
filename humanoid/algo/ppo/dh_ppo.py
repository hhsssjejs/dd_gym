import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_dh import ActorCriticDH
from .rollout_storage import RolloutStorage
import numpy as np


class DHPPO:
    actor_critic: ActorCriticDH

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 lin_vel_idx=45,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 grad_penalty_coef_schedule=[0, 0, 0],
                 sym_loss=False,
                 obs_permutation=None,
                 act_permutation=None,
                 frame_stack=66,
                 sym_coef=1.0,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic: ActorCriticDH = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(),
                                    lr=learning_rate)
        self.state_estimator_optimizer = optim.Adam(self.actor_critic.state_estimator.parameters(),
                                                    lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_short_obs = self.actor_critic.num_short_obs
        self.lin_vel_idx = lin_vel_idx

        # Adaptation
        self.gradient_penalty_coef_schedule = grad_penalty_coef_schedule
        self.counter = 0

        # sym
        self.sym_loss = sym_loss
        self.sym_coef = sym_coef
        if self.sym_loss:
            self.act_perm_mat = torch.zeros((len(act_permutation), len(act_permutation))).cuda()
            for i, perm in enumerate(act_permutation):
                self.act_perm_mat[int(abs(perm))][i] = np.sign(perm)
            obs_permutation_stack = []
            for i in range(frame_stack):
                for p in obs_permutation:
                    obs_permutation_stack.append(np.sign(p) * (abs(p) + i * len(obs_permutation)))
            self.obs_perm_mat = torch.zeros((len(obs_permutation_stack), len(obs_permutation_stack))).cuda()
            for i, perm in enumerate(obs_permutation_stack):
                self.obs_perm_mat[int(abs(perm))][i] = np.sign(perm)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, None, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]

        if grad_log_prob is None:
            grad_log_prob = torch.zeros_like(obs_batch)

        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_state_estimator_loss = 0
        mean_grad_penalty_loss = 0
        mean_sym_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            obs_est_batch = obs_batch.clone()

            obs_est_batch.requires_grad_()

            self.actor_critic.act(obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            state_estimator_input = obs_est_batch[:, -self.num_short_obs:]
            est_lin_vel = self.actor_critic.state_estimator(state_estimator_input)
            ref_lin_vel = critic_obs_batch[:, self.lin_vel_idx:self.lin_vel_idx + 3].clone()
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # sym loss
            sym_loss = 0
            if self.sym_loss:
                mirror_obs = torch.matmul(obs_est_batch, self.obs_perm_mat)
                mirror_act = self.actor_critic.act_inference(mirror_obs)
                m_mirror_act = torch.matmul(mirror_act, self.act_perm_mat)
                sym_loss = (mu_batch - m_mirror_act).pow(2).mean()
                # print("shapes:",obs_batch.shape,mirror_obs.shape,mirror_act.shape,m_mirror_act.shape,mu_batch.shape)

            # Calculate the gradient penalty loss
            gradient_penalty_loss = self._calc_grad_penalty(obs_est_batch, actions_log_prob_batch)

            gradient_stage = min(max((self.counter - self.gradient_penalty_coef_schedule[2]), 0) / self.gradient_penalty_coef_schedule[3], 1)
            gradient_penalty_coef = gradient_stage * (self.gradient_penalty_coef_schedule[1] - self.gradient_penalty_coef_schedule[0]) + self.gradient_penalty_coef_schedule[0]

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # update all actor_critic.parameters()
            loss = (surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    torch.nn.MSELoss()(est_lin_vel, ref_lin_vel) +
                    0. * gradient_penalty_coef * gradient_penalty_loss +
                    3. * self.sym_coef * sym_loss)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # didn't use a seperate optimizer to update state estimator NN
            state_estimator_loss = torch.nn.MSELoss()(est_lin_vel, ref_lin_vel)
            # self.state_estimator_optimizer.zero_grad()
            # state_estimator_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.state_estimator.parameters(), self.max_grad_norm)
            # self.state_estimator_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_state_estimator_loss += state_estimator_loss.item()
            mean_grad_penalty_loss += gradient_penalty_loss.item()
            mean_sym_loss += sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_state_estimator_loss /= num_updates
        mean_grad_penalty_loss /= num_updates
        mean_sym_loss /= num_updates
        self.storage.clear()
        self.update_counter()

        return mean_value_loss, mean_surrogate_loss, mean_state_estimator_loss, mean_grad_penalty_loss, mean_sym_loss

    def update_counter(self):
        self.counter += 1
