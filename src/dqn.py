import gym
import torch

from collections import deque
import random

import copy
import torch.autograd import Variable

env = gym.envs.make("MountainCar-v0")

class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.Relu(),
            torch.nn.Linear(n_hidden, n_action)
        )


        self.model_target = copy.deepcopy(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
