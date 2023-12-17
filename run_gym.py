"""Toy script."""
import random

import numpy as np

from rl_final_project.agent import Agent
from rl_final_project.control import MonteCarloControl
from rl_final_project.dqn import DQNControl
from rl_final_project.dqn import DQNFunction
from rl_final_project.environment import EnvironmentNormalizer
from rl_final_project.q_functions import QDeep


def run_dqn(scale: float, n0: float, gamma: float = 0.9) -> tuple:
    """Run DQN."""
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    # env = gym.make("CartPole-v1")
    env = EnvironmentNormalizer.from_gym("CartPole-v1")
    n_states = env.observation_space.shape[0]
    q_function = DQNFunction(
        batch_size=128,
        n_actions=env.action_space.n,
        n_feat=n_states,
        discrete_scale=scale,
    )
    agent = Agent(q_function, n0=n0, n_actions=env.action_space.n)
    control = DQNControl(
        lr=0.0001,
        tau=0.005,
        env=env,
        agent=agent,
        num_episodes=200_000,
        γ=gamma,
        batch_size=128,
    )
    ma_score = control.fit()
    env.close()
    return scale, n0, ma_score, agent.q_function.states_explored


def run(scale: float, n0: float, gamma: float = 0.9) -> tuple:
    """Run Classic Control Algorithm."""
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = EnvironmentNormalizer.from_gym("CartPole-v1")
    n_states = env.observation_space.shape[0]
    q_function = QDeep(
        batch_size=128,
        n_actions=env.action_space.n,
        n_feat=n_states,
        discrete_scale=scale,
    )
    agent = Agent(q_function, n0=n0, n_actions=env.action_space.n)
    control = MonteCarloControl(
        env, agent, num_episodes=200_000, gamma=gamma, batch_size=128
    )
    ma_score = control.fit()
    env.close()
    return scale, n0, ma_score, agent.q_function.states_explored


# from multiprocessing import Pool
#
# if __name__ == '__main__':
#     # run grid-search
#     scale = [3, 5, 10, 15, 20, 25]
#     N_0 = [1, 5, 10, 15, 20, 25]
#     gamma = [0.9, 0.95, 0.99, 0.999]
#
#     with Pool(24) as p:
#         results = p.starmap(run, [(s, n, g) for s in scale for n in N_0
#         for g in gamma])
#
#     print(results)

if __name__ == "__main__":
    # run single experiment
    run(10, 10, gamma=0.99)
