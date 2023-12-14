import random

import numpy as np
import gymnasium as gym

from src.agent import Agent
from src.control import MonteCarloControl, QLearningControl, SarsaLambdaControl
from src.environment import EnvironmentDiscretizer, BatchRenormalization
from src.q_functions import QTabular, QLinear, QDeep


def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = gym.make('CartPole-v1')
    #env
    env = EnvironmentDiscretizer(env, discrete_scale=1)
    #env = BatchRenormalization(env, env.observation_space.shape, discrete_scale=scale)#, render_mode='human')
    n_states = env.observation_space.shape[0]
    #q_function = QDeep(n_states, env.action_space.n, discrete_scale=scale)
    q_function = QLinear(n_actions=env.action_space.n, n_feat=n_states, discrete_scale=scale)
    agent = Agent(q_function, N_0=N_0, n_actions=env.action_space.n)
    control = SarsaLambdaControl(0.4, env, agent, num_episodes=200_000, γ=gamma)
    ma_score = control.fit()
    #a = np.array(q_function.state_history)
    #print(np.mean(a, axis=0))
    #print(np.std(a, axis=0))
    env.close()
    return scale, N_0, ma_score, agent.q_function.states_explored

# from multiprocessing import Pool
#
# if __name__ == '__main__':
#     # run grid-search
#     scale = [3, 5, 10, 15, 20, 25]
#     N_0 = [1, 5, 10, 15, 20, 25]
#     gamma = [0.9, 0.95, 0.99, 0.999]
#
#     with Pool(24) as p:
#         results = p.starmap(run, [(s, n, g) for s in scale for n in N_0 for g in gamma])
#
#     print(results)

# run single experiment
run(40, 5, gamma=0.2)