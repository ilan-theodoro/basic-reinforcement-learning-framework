import random

import numpy as np

from mo436_final_project.src.src.agent import Agent
from mo436_final_project.src.src.control import MonteCarloControl
from mo436_final_project.src.src.dqn import DQNFunction, DQNControl
from mo436_final_project.src.src.environment import EnvironmentNormalizer
from mo436_final_project.src.src.q_functions import QDeep

def run_dqn(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    #env = gym.make("CartPole-v1")
    env = EnvironmentNormalizer.from_gym('CartPole-v1')
    n_states = env.observation_space.shape[0]
    q_function = DQNFunction(batch_size=128, n_actions=env.action_space.n, n_feat=n_states, discrete_scale=scale)
    agent = Agent(q_function, N_0=N_0, n_actions=env.action_space.n)
    control = DQNControl(lr=0.0001, τ=0.005, env=env, agent=agent, num_episodes=200_000, γ=gamma, batch_size=128)
    ma_score = control.fit()
    env.close()
    return scale, N_0, ma_score, agent.q_function.states_explored


def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = EnvironmentNormalizer.from_gym('CartPole-v1')
    n_states = env.observation_space.shape[0]
    q_function = QDeep(batch_size=128, n_actions=env.action_space.n, n_feat=n_states, discrete_scale=scale)
    agent = Agent(q_function, N_0=N_0, n_actions=env.action_space.n)
    control = MonteCarloControl(env, agent, num_episodes=200_000, γ=gamma, batch_size=128)
    ma_score = control.fit()
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
run_dqn(
    100,
    10,
    gamma=0.99
)
