import model
from collections import namedtuple, deque
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64         # minibatch size
LR = 1e-5               # learning rate
UPDATE_EVERY = 1        # how often to update the network

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("device", device)

class ReachAgent():
    def __init__(self, state_num, action_num, discount_factor):
        self.action_num = action_num
        self.discount_factor = discount_factor

        # declare the network
        self.actor = model.Actor(state_num, action_num).to(device)
        self.actor_target = model.Actor(state_num, action_num).to(device)

        self.critic = model.Critic(state_num, action_num).to(device)
        self.critic_target = model.Critic(state_num, action_num).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        # experience replay queue
        self.memory = ReplayBuffer(int(1e5), BATCH_SIZE)

        self.t_step = 0

    def act(self, states):
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
        self.actor.train()

        return actions.cpu().data.numpy()


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
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < BATCH_SIZE:
            return False
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if 0 == self.t_step:
            experiences = self.memory.sample()
            self.__learn(experiences, self.discount_factor)
            return True
        return False

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



class ReplayBuffer():
    # To store experience tuples
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        # create a new experience, and add to memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # randomly sample batch num of experience
        experiences = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
