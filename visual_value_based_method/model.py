import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A deep Q netowrk that handles image inputs, and predict the Q value
# for each discrete action
HIDDEN_LAYER_1 = 32
HIDDEN_LAYER_2 = 64
IMAGE_SIZE = 84
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
        self.fc1 = nn.Linear((IMAGE_SIZE/2)*(IMAGE_SIZE/2)*HIDDEN_LAYER_2, num_action_space)

    def forward(self, x):
        """network forward function
        Params
        ======
            x: input tensor with size [batch, 1, 84, 84]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)
