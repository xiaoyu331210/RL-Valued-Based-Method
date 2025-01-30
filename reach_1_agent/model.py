import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, state_num, action_num):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_num)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.tanh(x)

class Critic(nn.Module):
    """Some Information about Critic"""
    def __init__(self, state_num, action_num):
        super(Critic, self).__init__()
        self.state_fc = nn.Linear(state_num, 64)
        self.action_fc = nn.Linear(action_num, 64)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.action_fc.weight.data.uniform_(*hidden_init(self.action_fc))
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        state = F.relu(self.state_fc(state))
        action = F.relu(self.action_fc(action))

        combined = torch.cat([state, action], dim=1)
        combined = F.relu(self.fc1(combined))
        return self.fc2(combined)
