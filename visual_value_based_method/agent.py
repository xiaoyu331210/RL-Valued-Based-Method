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

class VisualStateAgent():
    def __init__(self, action_num, gamma):
        self.action_num = action_num
        self.gamma = gamma

        # create the network
        self.network_local = model.VisualStateDQN(action_num).to(device)
        self.network_target = model.VisualStateDQN(action_num).to(device)

        # Experience replay queue
        self.memory = ReplayBuffer(int(1e5), BATCH_SIZE)

        self.optimizer = optim.Adam(self.network_local.parameters())

        self.t_step = 0
    
    def step(self, states, action, reward, next_states, done):
        # update memory with the current observations
        self.memory.add(states, action, reward, next_states, done)
        if len(self.memory) < BATCH_SIZE:
            return
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if 0 == self.t_step:
            experiences = self.memory.sample()
            self.__learn(experiences, self.gamma)
    
    def act(self, states, eps):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        states = torch.from_numpy(states).float().to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(states)
        self.network_local.train()

        # do spsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_num))
        

    def __learn(self, experienecs, gamma):
        # get indivdiual elements from experiences
        states, actions, rewards, next_states, dones = experienecs

        # get Q value from both target and current network
        q_target_next_max = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_target_next_max * (1 - dones))

        q_expected = self.network_local(states).gather(1, actions.long())

        # compute loss
        loss = F.mse_loss(q_targets, q_expected)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        TAU = 1e-3
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

    def add(self, states, action, reward, next_states, done):
        # create a new experience, and add to memory
        e = self.experience(states, action, reward, next_states, done)
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

    # def sample(self):
    #     # # randomly sample batch num of experience
    #     # experiences = random.sample(self.memory, self.batch_size)

    #     # Randomly sample an index
    #     experience_indices = random.sample(range(len(self.memory)), self.batch_size)
    #     states = np.zeros((self.batch_size, model.INPUT_IMAGE_NUM, model.IMAGE_SIZE, model.IMAGE_SIZE))
    #     # actions = np.zeros((self.batch_size))
    #     for i in range(self.batch_size):
    #         curr_idx = experience_indices[i]
    #         query_indices = list(range(curr_idx-model.INPUT_IMAGE_NUM+1, curr_idx+1))
    #         query_indices = [max(0, idx) for idx in query_indices]
    #         print("query ind: ", query_indices)
    #         curr_state = np.zeros((model.INPUT_IMAGE_NUM, model.IMAGE_SIZE, model.IMAGE_SIZE))
    #         print("curr state size: ", curr_state.shape)
    #         for j in range(model.INPUT_IMAGE_NUM):
    #             print(" query memory size: ", list(self.memory)[query_indices[j]].state.shape)
    #             curr_state[j] = list(self.memory)[query_indices[j]].state
    #         states[i] = curr_state

    #         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
    #         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    #         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    #         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)


    #     states = torch.from_numpy(states).float().to(device)
    #     actions = torch.from_numpy(actions).float().to(device)
    #     rewards = torch.from_numpy(rewards).float().to(device)
    #     next_states = torch.from_numpy(next_states).float().to(device)
    #     dones = torch.from_numpy(dones).float().to(device)

    #     # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    #     # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
    #     # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    #     # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    #     # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

