from abc import ABC, abstractmethod
from functools import partial
import random

import numpy as np
import gymnasium as gym
import torch
from tqdm import tqdm

from src.agent import Agent
from src.control import MonteCarloControl, QLearningControl
from src.q_functions import QTabular, QLinear

env = gym.make("FrozenLake-v1")

from collections import namedtuple
import numpy as np




class MLP(torch.nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(n_states, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 32, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(32, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, n_actions, bias=True))

    def forward(self, x):
        return self.model(x)

class FunctionApproximationNonLinear:
    def __init__(self, n_states, n_actions):

        self.model = MLP(n_states, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.accumulator = 0
        self.accum_s = []
        self.accum_a = []
        self.accum_y = []

        from ema_pytorch import EMA
        self.ema = EMA(self.model)

    def __call__(self, state, action):
        x = torch.tensor(state, dtype=torch.float)
        return self.ema(x)[action]

    def update(self, state, action, target, α=0.1):
        self.accum_s.append(state)
        self.accum_a.append(action)
        self.accum_y.append(target)

        self.accumulator += 1
        if self.accumulator % 100 == 0:

            x = torch.tensor(self.accum_s, dtype=torch.float)
            y = torch.tensor(self.accum_y, dtype=torch.float)
            a = torch.tensor(self.accum_a, dtype=torch.long)
            y_pred = self.model(x)[np.arange(100), a]
            loss = torch.nn.functional.l1_loss(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()

            self.accum_a = []
            self.accum_s = []
            self.accum_y = []


def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = gym.make('CartPole-v1')#, render_mode='human')
    q_function = QLinear(env.action_space.n, discrete_scale=scale)
    agent = Agent(q_function, N_0=N_0, scale=scale, n_actions=env.action_space.n)
    control = QLearningControl(env, agent, num_episodes=200_000, γ=gamma, discrete_scale=scale)
    ma_score = control.fit()
    env.close()
    return scale, N_0, ma_score, len(agent.N)


run(20, 5, gamma=1)

# with Pool(20) as p:
#     results = p.starmap(run, [(scale, N_0) for scale in [1, 3, 5, 7, 10, 20] for N_0 in [1, 2, 3, 5, 10, 20]])
#
#     for (scale, N_0, ma_score, exp) in results:
#         print(f"scale: {scale}, N_0: {N_0}, ma_score: {np.mean(ma_score):.2f}, exp: {exp}")
