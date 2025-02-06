import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

ACTOR_HIDDEN_LAYER_1 = 256
ACTOR_HIDDEN_LAYER_2 = 128
class Actor(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, state_num, action_num, hidden_layer_1=ACTOR_HIDDEN_LAYER_1, hidden_layer_2=ACTOR_HIDDEN_LAYER_2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, action_num)
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

CRITIC_HIDDEN_LAYER_1 = 256
CRITIC_HIDDEN_LAYER_2 = 128
class Critic(nn.Module):
    """Some Information about Critic"""
    def __init__(self, state_num, action_num, hidden_layer_1=CRITIC_HIDDEN_LAYER_1, hidden_layer_2=CRITIC_HIDDEN_LAYER_2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1 + action_num, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((x, action), dim=1)))
        return self.fc3(x)

class MADDPGCritic(nn.Module):
    def __init__(self, state_num, action_num, agent_num, hidden_layer_1=CRITIC_HIDDEN_LAYER_1, hidden_layer_2=CRITIC_HIDDEN_LAYER_2):
        super(MADDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_num * agent_num, hidden_layer_1)
        self.fc2 = nn.Linear(action_num * agent_num + hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((x, action), dim=1)))
        return self.fc3(x)


# Actor Network
class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(MAPPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

# Centralized Critic Network
class MAPPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(MAPPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.net(x)

