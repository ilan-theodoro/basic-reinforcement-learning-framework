"""Test DQN."""
import random

import numpy as np
import pytest

from rl_final_project.agent import Agent
from rl_final_project.dqn import DQNControl
from rl_final_project.dqn import DQNFunction
from rl_final_project.environment import EnvironmentNormalizer


@pytest.mark.usefixtures("available_envs")
def test_dqn(available_envs: list) -> None:
    """Test DQN."""
    scale = 1
    n0 = 1
    gamma = 0.9

    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    # env = gym.make("CartPole-v1")
    env = EnvironmentNormalizer.from_gym("CartPole-v1")
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    n_states = env.observation_space.shape[0]
    q_function = DQNFunction(
        batch_size=128,
        n_actions=env.action_space.shape[0],
        n_feat=n_states,
        discrete_scale=scale,
    )
    agent = Agent(q_function, n0=n0, n_actions=env.action_space.shape[0])
    control = DQNControl(
        lr=0.0001,
        tau=0.005,
        env=env,
        agent=agent,
        num_episodes=100,
        gamma=gamma,
        batch_size=16,
    )
    control.fit()
    env.close()
