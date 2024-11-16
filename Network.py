import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import gymnasium as gym


class Network(nn.Module):
    def __init__(self, state_size, action_size, seed = 42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)  #first is the nr of input variables and the second is the nr of neurons from the first layer
        self.fc2 = nn.Linear(64, 64)         # the nr 64 is obtained through many trials
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)