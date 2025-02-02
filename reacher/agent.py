import model
from collections import namedtuple, deque
import random
import numpy as np
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

@dataclass
class DDPGConfig:
    num_state: int = 0
    num_action: int = 0
    num_agent: int = 1
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    batch_size: int = 128
    update_every_timestamp: int = 20
    update_time_each_stamp: int = 10
    discount_factor: float = 0.99
    replay_buffer_size: int = 1e6
    weight_decay: float = 0

class DDPGAgent():
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        # declare the network
        self.actor = model.Actor(config.num_state, config.num_action).to(self.device)
        self.actor_target = model.Actor(config.num_state, config.num_action).to(self.device)

        self.critic = model.Critic(config.num_state, config.num_action).to(self.device)
        self.critic_target = model.Critic(config.num_state, config.num_action).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate, weight_decay=config.weight_decay)

        self.noise = OUNoise((config.num_agent, config.num_action), 2)

        # experience replay queue
        self.memory = ReplayBuffer(int(config.replay_buffer_size), config.batch_size, self.device)

        # record time steps
        self.t_step = 0

        # model weight file name
        self.default_actor_weight_prefix = '/ddpg_actor'
        self.default_critic_weight_prefix = '/ddpg_critic'
        self.default_actor_target_weight_prefix = '/ddpg_actor_target'
        self.default_critic_target_weight_prefix = '/ddpg_critic_target'

    def reset(self):
        self.noise.reset()

    def act(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).cpu().data.numpy()
        self.actor.train()

        # return actions
        actions += self.noise.sample()
        return np.clip(actions, -1., 1.)


    def step(self, state, action, reward, next_state, done):
        """add one observation tuple into replay buffer, and optimize the model

        Params
        ======
            state (array_like): the current state
            action (array_like): the current action taken
            reward (array_like): the reward after taking the current action
            next_state (array_like): the next state after taking action
            done (array_like): whether the episode is finished
        """
        # Update the memory with the latest experience, and perform learn step
        # self.memory.add(state, action, reward, next_state, done)
        for i in range(self.config.num_agent):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        if len(self.memory) < self.config.batch_size:
            return False
        self.t_step = (self.t_step + 1) % self.config.update_every_timestamp
        if 0 != self.t_step:
            return False

        for _ in range(self.config.update_time_each_stamp):
            experiences = self.memory.sample()
            self.__learn(experiences, self.config.discount_factor)
        return True

    def __learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        next_actions = self.actor_target(next_states)
        # compute the Q value from the next states
        next_state_Q = self.critic_target(next_states, next_actions)
        q_targets = rewards + (discount_factor * next_state_Q * (1- dones))
        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_targets, q_expected)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        action_pred = self.actor(states)
        actor_loss = -self.critic(states, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------- soft update weights ------------------------ #
        TAU = 1e-3
        self.__soft_update(self.actor, self.actor_target, TAU)
        self.__soft_update(self.critic, self.critic_target, TAU)


    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, folder_path, file_appendix = ''):
        torch.save(self.actor.state_dict(), folder_path + self.default_actor_weight_prefix + file_appendix + '.pth')
        torch.save(self.actor_target.state_dict(), folder_path + self.default_actor_target_weight_prefix + file_appendix + '.pth')
        torch.save(self.critic.state_dict(), folder_path + self.default_critic_weight_prefix + file_appendix + '.pth')
        torch.save(self.critic_target.state_dict(), folder_path + self.default_critic_target_weight_prefix + file_appendix + '.pth')

    def load_model(self, folder_path, file_appendix = ''):
        self.actor.load_state_dict(torch.load(folder_path + self.default_actor_weight_prefix + file_appendix + '.pth'))
        self.actor_target.load_state_dict(torch.load(folder_path + self.default_actor_target_weight_prefix + file_appendix + '.pth'))
        self.critic.load_state_dict(torch.load(folder_path + self.default_critic_weight_prefix + file_appendix + '.pth'))
        self.critic_target.load_state_dict(torch.load(folder_path + self.default_critic_target_weight_prefix + file_appendix + '.pth'))


class ReplayBuffer():
    # To store experience tuples
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        # create a new experience, and add to memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # randomly sample batch num of experience
        experiences = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size  
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
