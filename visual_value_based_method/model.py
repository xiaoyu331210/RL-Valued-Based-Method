import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A deep Q netowrk that handles image inputs, and predict the Q value
# for each discrete action
class VisualStateDQN(nn.Module):
    def __init__(self, num_action_space, input_channel, image_size=[84,84], filter_channel=[16, 32], fc_hidden_layer=[64, 64]):
        super(VisualStateDQN, self).__init__()
        self.input_channel = input_channel

        # the input image should be [84 x 84]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, filter_channel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_channel[0]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter_channel[0], filter_channel[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_channel[1]))
        
        self.fc1 = nn.Sequential(
            nn.Linear(int(image_size[0]/4)*int(image_size[1]/4)*filter_channel[1], fc_hidden_layer[0]),
            nn.BatchNorm1d(fc_hidden_layer[0]))
        self.fc2 = nn.Sequential(
            nn.Linear(fc_hidden_layer[0], fc_hidden_layer[1]),
            nn.BatchNorm1d(fc_hidden_layer[1]))
        self.fc3 = nn.Linear(fc_hidden_layer[1], num_action_space)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """network forward function
        Params
        ======
            x: input tensor with size [batch, input_image_channel, 84, 84]
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
