import model
from collections import namedtuple, deque
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


BATCH_SIZE = 32         # minibatch size
LR = 1e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network
REGULARIZATION = 1e-4   # regularization parameter
IMAGE_SIZE = [84, 84]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("device", device)

def augment_state(state_1, action_1, state_2, action_2, state_3):
    assert state_1.shape == state_2.shape
    assert state_2.shape == state_3.shape

    state_shape = [state_1.shape[-2], state_1.shape[-1]]
    action_1_data = np.ones((1,1,state_shape[0],state_shape[1]))*action_1
    action_2_data = np.ones((1,1,state_shape[0],state_shape[1]))*action_2
    # return np.concatenate((state_1, action_1_data, state_2, action_2_data, state_3), axis=1)
    return np.concatenate((state_1, state_2, state_3), axis=1)

class VisualStateAgent():
    def __init__(self, action_num, gamma):
        self.action_num = action_num
        self.gamma = gamma

        # create the network
        filter_channel=[16, 32]
        fc_hidden_layer=[64, 64]
        input_image_channel_num = 9

        self.network_local = model.VisualStateDQN(action_num, input_image_channel_num, IMAGE_SIZE, filter_channel, fc_hidden_layer).to(device)
        self.network_target = model.VisualStateDQN(action_num, input_image_channel_num, IMAGE_SIZE, filter_channel, fc_hidden_layer).to(device)

        # Experience replay queue
        self.memory = ReplayBuffer(int(1e5), BATCH_SIZE, IMAGE_SIZE)

        self.optimizer = optim.Adam(self.network_local.parameters())

        self.t_step = 0
    
    def update_experience(self, states, action, reward, next_states, done):
        # update memory with the current observations
        self.memory.add(states, action, reward, next_states, done)
        return
    
    def step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if 0 == self.t_step:
            experiences = self.memory.sample_augmented_experience()
            self.__learn(experiences, self.gamma)
    
    def act(self, states, eps):
        states = torch.from_numpy(states).float().to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(states)
        self.network_local.train()

        # do epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_num))

    def augment_state(self, state):
        # Augment the state to include previous observations and actions
        input_image_shape = self.memory.input_image_shape
        has_enough_experience = len(self.memory) >= 2

        curr_e = self.memory.experience(state, 0, 0, state, 0)

        prev_prev_e = self.memory.memory[-2] if has_enough_experience else curr_e
        prev_e = self.memory.memory[-1] if has_enough_experience else curr_e

        return augment_state(prev_prev_e.state, prev_prev_e.action, prev_e.state, prev_e.action, curr_e.state)

        # if len(self.memory) >= 2:
        #     prev_idx = len(self.memory) - 1
        #     prev_prev_idx = prev_idx - 1
        #     prev_e = self.memory.memory[prev_idx]
        #     prev_prev_e = self.memory.memory[prev_prev_idx]

        #     #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
        #     prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*prev_e.action
        #     prev_prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*prev_prev_e.action
        #     aug_state = np.concatenate((prev_prev_e.state, prev_prev_e_a, prev_e.state, prev_e_a, state), axis=1)
        # else:
        #     #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
        #     initial_action = 0
        #     prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*initial_action
        #     prev_prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*initial_action
        #     aug_state = np.concatenate((state, prev_prev_e_a, state, prev_e_a, state), axis=1)

        # return aug_state

    def __learn(self, experienecs, gamma):
        # get indivdiual elements from experiences
        states, actions, rewards, next_states, dones = experienecs

        # get Q value from both target and current network
        q_target_next_max = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * q_target_next_max * (1 - dones)

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
    def __init__(self, buffer_size, batch_size, image_size):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size
        self.input_image_shape = image_size

    def add(self, states, action, reward, next_states, done):
        # create a new experience, and add to memory
        e = self.experience(states, action, reward, next_states, done)
        self.memory.append(e)

    def sample_augmented_experience(self):
        """Randomly sample a batch of experiences from memory."""
        #Note: the experiences are store in the memory in chronoogical order

        aug_states = [] #augment state
        actions = []
        rewards = []
        aug_next_states = [] #augment next state
        dones = []
        ind = random.sample(range(len(self.memory)), self.batch_size)
        for idx in ind:
            #idx = 3+len(aug_states) #take experiences in order and in agent.step make sure 'len(self.memory) > BATCH_SIZE+5'
            e = self.memory[idx]
            experience_valid = e is not None and (idx - 2) >= 0 and (idx + 1) < len(self.memory)
            prev_e = self.memory[idx-1] if experience_valid else e
            prev_prev_e = self.memory[idx-2] if experience_valid else e
            next_e = self.memory[idx+1] if experience_valid else e

            #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
            # augment states
            aug_states.append(augment_state(prev_prev_e.state, prev_prev_e.action, prev_e.state, prev_e.action, e.state))
            actions.append(e.action)
            rewards.append(e.reward)
            # augment next states
            aug_next_states.append(augment_state(prev_e.state, prev_e.action, e.state, e.action, next_e.state))
            dones.append(e.done)

        #augment state is of shape Nx11x84x84
        aug_states = torch.from_numpy(np.vstack([s for s in aug_states])).float().to(device)
        actions = torch.from_numpy(np.vstack([a for a in actions])).long().to(device)
        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().to(device)
        aug_next_states = torch.from_numpy(np.vstack([ns for ns in aug_next_states])).float().to(device)
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(device)
  
        return (aug_states, actions, rewards, aug_next_states, dones)

    def __len__(self):
        return len(self.memory)
