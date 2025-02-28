import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

ACTOR_HIDDEN_LAYER_1 = 400
ACTOR_HIDDEN_LAYER_2 = 300
class Actor(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, state_num, action_num):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_num, ACTOR_HIDDEN_LAYER_1)
        self.norm1 = nn.BatchNorm1d(ACTOR_HIDDEN_LAYER_1)
        self.fc2 = nn.Linear(ACTOR_HIDDEN_LAYER_1, ACTOR_HIDDEN_LAYER_2)
        self.norm2 = nn.BatchNorm1d(ACTOR_HIDDEN_LAYER_2)
        self.fc3 = nn.Linear(ACTOR_HIDDEN_LAYER_2, action_num)
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

CRITIC_HIDDEN_LAYER_1 = 400
CRITIC_HIDDEN_LAYER_2 = 300
class Critic(nn.Module):
    """Some Information about Critic"""
    def __init__(self, state_num, action_num):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_num, CRITIC_HIDDEN_LAYER_1)
        self.norm1 = nn.BatchNorm1d(CRITIC_HIDDEN_LAYER_1)
        self.fc2 = nn.Linear(CRITIC_HIDDEN_LAYER_1 + action_num, CRITIC_HIDDEN_LAYER_2)
        self.norm2 = nn.BatchNorm1d(CRITIC_HIDDEN_LAYER_2)
        self.fc3 = nn.Linear(CRITIC_HIDDEN_LAYER_2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(3e-4, 3e-4)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((x, action), dim=1)))
        return self.fc3(x)
