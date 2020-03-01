from gym import make
import numpy as np
import torch
import copy
import random
from collections import deque
from torch.autograd import Variable
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import os

BATCH_SIZE = 128
BETA = 69


class Exploration:
    def __init__(self, 
                 state_dim=2,
                 action_dim=3, 
                 hd_target=256,
                 hd_model=64, 
                 lr=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.target = nn.Sequential(nn.Linear(state_dim, hd_target), 
                                    nn.ReLU(), 
                                    nn.Linear(hd_target, hd_target),
                                    nn.ReLU(),
                                    nn.Linear(hd_target, action_dim))
        
        self.model = nn.Sequential(nn.Linear(state_dim, hd_model), 
                                    nn.ReLU(), 
                                    nn.Linear(hd_model, hd_model),
                                    nn.ReLU(),
                                    nn.Linear(hd_model, action_dim))
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_exploration_reward(self, state):
        state = torch.Tensor(state)
        y_target = self.target(state)
        y_pred = self.model(state)
        return torch.pow(y_target - y_pred, 2).sum(axis=-1)

        
    def update(self, state):
        state = torch.Tensor(state)
        loss = self.get_exploration_reward(state)
        self.optim.zero_grad()
        loss.sum().backward()
        self.optim.step()

class Agent:
    def __init__(self,
                 state_dim=2, 
                 action_dim=3, 
                 hd=128, 
                 exp_hd_target=256,
                 exp_hd_model=256,
                 exp_lr=1e-4, 
                 lr=1e-4,
                 gamma=0.9, 
                 eps=1,
                 eps_decay=0.96):
        self.gamma = gamma
        self.state_dim = state_dim # dimensionalite of state space
        self.action_dim = action_dim # count of available actions
        self.exploration = Exploration(state_dim, action_dim, exp_hd_target, exp_hd_model, exp_lr)
        self.memory = deque(maxlen=5000)
        
        self.model = nn.Sequential(nn.Linear(state_dim, hd), 
                                    nn.ReLU(), 
                                    nn.Linear(hd, hd),
                                    nn.ReLU(),
                                    nn.Linear(hd, action_dim))
        
        self.target = copy.deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.upd = 0
        self.eps = eps
        self.ed = eps_decay
        
    def act(self, state):
        if np.random.rand() < self.eps:
            res = np.random.randint(self.action_dim)
        else:
            state = torch.Tensor(state)
            res = int(torch.argmax(self.model(state).data))
        self.eps = max(0.01, self.eps * self.ed)
        return res


    def update(self, transition):
        self.memory.append(transition)
        state, action, next_state, reward, done = transition
        self.exploration.update(state)
        if len(self.memory) < BATCH_SIZE:
            return
        self.upd += 1
        state, action, next_state, reward, done = zip(*random.sample(self.memory, BATCH_SIZE))
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action)
        done = torch.tensor(done)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float) + BETA * self.exploration.get_exploration_reward(state) 
        Q1 = self.model(state)[np.arange(BATCH_SIZE), action]
        Q2 = reward + self.gamma * self.target(next_state).max(dim=1)[0].detach() * (~done)        
        loss = F.smooth_l1_loss(Q1, Q2)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
#         self.memory.clear()
        if self.upd >= 200:
            self.update_target()
            self.upd = 0

    
    def update_target(self):
        self.target = copy.deepcopy(self.model)

    def reset(self):
        pass