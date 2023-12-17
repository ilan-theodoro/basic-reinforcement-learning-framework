"""Test cases for the __main__ module."""
import random

import numpy as np

from rl_final_project.agent import Agent
from rl_final_project.dqn import DQNControl
from rl_final_project.dqn import DQNFunction
from rl_final_project.environment import EnvironmentNormalizer


# define test case for DQN
def test_dqn() -> None:
    """Test DQN."""
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = EnvironmentNormalizer.from_gym("CartPole-v1")
    n_states = env.observation_space.shape[0]
    q_function = DQNFunction(
        batch_size=128,
        n_actions=env.action_space.n,
        n_feat=n_states,
        discrete_scale=2,
    )
    agent = Agent(q_function, n0=1, n_actions=env.action_space.n)
    control = DQNControl(
        lr=0.0001,
        tau=0.005,
        env=env,
        agent=agent,
        num_episodes=1_000,
        γ=0.9,
        batch_size=128,
    )
    ma_score = control.fit()
    env.close()

    assert ma_score > 20
