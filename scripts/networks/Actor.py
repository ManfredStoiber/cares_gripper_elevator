import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.weight_bias_init import weight_init


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Actor, self).__init__()

        self.hidden_size = [1024, 1024]

        self.h_linear_1 = nn.Linear(in_features=observation_size,    out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=num_actions)

        self.bn1 = nn.BatchNorm1d(self.hidden_size[0])
        self.bn2 = nn.BatchNorm1d(self.hidden_size[1])

        # self.apply(weight_init)
        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, state):
        x = F.relu(self.h_linear_1(state))
        x = self.bn1(x)
        x = F.relu(self.h_linear_2(x))
        x = self.bn2(x)
        x = torch.tanh(self.h_linear_3(x))
        return x
