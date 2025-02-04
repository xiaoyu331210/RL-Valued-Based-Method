from collections import namedtuple, deque
import random
import numpy as np
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class ReplayBuffer():
    # To store experience tuples
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.experience = Experience
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

class ReplayBufferMultiAgent():
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.experience = Experience
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        # Each element is a tuple containing joint info for all agents.
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).to(self.device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

# On-policy collection
Experience2 = namedtuple("Experience2", field_names=["state", "action", "reward", "next_state", "done", "log_prob"])
class ReplayBufferMAPPO():
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done, log_prob):
        # Each element is a tuple containing joint info for all agents.
        e = Experience2(state, action, reward, next_state, done, log_prob)
        self.memory.append(e)
    
    def sample(self):
        return Experience2(*zip(*self.memory))
    
    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

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

def soft_update(local_model, target_model, tau):
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