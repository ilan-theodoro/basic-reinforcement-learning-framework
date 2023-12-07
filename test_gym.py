from abc import ABC, abstractmethod
from functools import partial
import random

import numpy as np
import gymnasium as gym
import torch
from tqdm import tqdm

from src.agent import Agent
from src.control import MonteCarloControl, QLearningControl
from src.q_functions import QTabular, QLinear, QDeep

env = gym.make("FrozenLake-v1")

from collections import namedtuple
import numpy as np

def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = gym.make('CartPole-v1')#, render_mode='human')
    q_function = QDeep(4, env.action_space.n, discrete_scale=scale)
    agent = Agent(q_function, N_0=N_0, scale=scale, n_actions=env.action_space.n)
    control = QLearningControl(env, agent, num_episodes=200_000, γ=gamma, discrete_scale=scale)
    ma_score = control.fit()
    env.close()
    return scale, N_0, ma_score, len(agent.N)


run(50, 5, gamma=0.95)

# with Pool(20) as p:
#     results = p.starmap(run, [(scale, N_0) for scale in [1, 3, 5, 7, 10, 20] for N_0 in [1, 2, 3, 5, 10, 20]])
#
#     for (scale, N_0, ma_score, exp) in results:
#         print(f"scale: {scale}, N_0: {N_0}, ma_score: {np.mean(ma_score):.2f}, exp: {exp}")
