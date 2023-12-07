import random

import numpy as np
import gymnasium as gym

from src.agent import Agent
from src.control import MonteCarloControl, QLearningControl
from src.q_functions import QTabular, QLinear, QDeep


def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = gym.make('CartPole-v1')#, render_mode='human')
    #q_function = #QDeep(4, env.action_space.n, discrete_scale=scale)
    q_function = QLinear(env.action_space.n, discrete_scale=scale)
    agent = Agent(q_function, N_0=N_0, n_actions=env.action_space.n)
    control = QLearningControl(env, agent, num_episodes=200_000, γ=gamma, discrete_scale=scale)
    ma_score = control.fit()
    env.close()
    return scale, N_0, ma_score, agent.q_function.states_explored


run(30, 20, gamma=0.9)