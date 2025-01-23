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

HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32
class VisualStateDQN(nn.Module):
    def __init__(self, num_action_space):
        super(VisualStateDQN, self).__init__()
        # the input image should be [84 x 84]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, HIDDEN_LAYER_1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(HIDDEN_LAYER_1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(HIDDEN_LAYER_2))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(42*42*HIDDEN_LAYER_2, num_action_space)
        
    
    def forward(self, x):
        # The input state dimension [batch, height, width, channel]
        # print("shape start", x.shape)
        x = F.relu(self.conv1(x)) # [batch, 84, 84, 64]
        # print("shape after conv1", x.shape)
        x = F.relu(self.conv2(x)) # [batch, 84, 84, 64]
        # print("shape after conv2", x.shape)
        x = self.pool(x) # [batch, 42, 42, 128]
        # print("shape after pool", x.shape)
        x = x.view(x.size(0), -1) # [batch, ]
        # print("shape after flatten", x.shape)
        # print("fc1", self.fc1)
        return self.fc1(x)
        # x = F.relu(self.fc1(x))
        # # print("shape after fc1", x.shape)
        # x = F.relu(self.fc2(x))

        # # print("shape after fc2", x.shape)
        # return self.fc3(x)


# test = torch.rand((2, 3, 4, 5))
# print(test.view(2, 3, -1).shape)
# print(test)
# action = torch.tensor([[0.],[1.]])
# action_int = action.int()
# print(action_int)
# print(test.gather(1, action_int))
# print(test.max(1)[0])
# print(test.max(1)[0].unsqueeze(0))
