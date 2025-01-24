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

# Agent to train a deep Q-network with discrete state input
class DiscreteStateAgent():
    def __init__(self, state_num, action_num, gamma):
        """Constructor

        Params
        ======
            state_num (int): number of discrete state
            action_num (int): number of discrete action
            gamma (float): discount factor
        """
        self.state_num = state_num
        self.action_num = action_num
        self.gamma = gamma

        # create the network
        self.network_local = model.DiscreteStateDQN(state_num, action_num).to(device)
        self.network_target = model.DiscreteStateDQN(state_num, action_num).to(device)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)

        # Experience Replay Queue
        self.memory = ReplayBuffer(int(1e5), BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

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
            self.__learn(experiences, self.gamma)
            return True
        return False


    def act(self, states, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(states)
        self.network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # use greedy method
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # randomly chose
            return random.choice(np.arange(self.action_num))


    def __learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # compute Q from the target network
        # "detach" so that the modification to output won't affect the network weight
        # this returns the action with the highest Q value for each state
        q_target_next_max = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_target_next_max * (1 - dones))

        # compute Q value from active updating network
        q_expected = self.network_local(states).gather(1, actions.long())

        # compute loss
        loss = F.mse_loss(q_targets, q_expected)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        TAU = 1e-3              # for soft update of target parameters
        self.__soft_update(self.network_local, self.network_target, TAU)


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

