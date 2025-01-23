import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# This class implements a simple value-based deep Q-network
class DiscreteStateDQN(nn.Module):
    def __init__(self, num_state_space, num_action_space):
        super(DiscreteStateDQN, self).__init__()
        self.fc1 = nn.Linear(num_state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_action_space)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# test = torch.rand((2, 3))
# print(test)
# action = torch.tensor([[0.],[1.]])
# action_int = action.int()
# print(action_int)
# print(test.gather(1, action_int))
# print(test.max(1)[0])
# print(test.max(1)[0].unsqueeze(0))
