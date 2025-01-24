import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A deep Q network with discrete input state
class DiscreteStateDQN(nn.Module):
    def __init__(self, num_state_space, num_action_space):
        """constructor
        Params
        ======
            num_state_space: number of discrete states
            num_action_space: number of discrete actions
        """
        super(DiscreteStateDQN, self).__init__()
        self.fc1 = nn.Linear(num_state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_action_space)

    def forward(self, states):
        """network forward function
        Params
        ======
            x: input tensor with size [batch, num_state_space]
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
