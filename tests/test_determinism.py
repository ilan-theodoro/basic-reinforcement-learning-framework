"""Test determinism of the Monte Carlo control algorithm."""
import random

import gymnasium
import numpy as np
import pytest

from rl_final_project.agent import Agent
from rl_final_project.control import MonteCarloControl
from rl_final_project.environment import EnvironmentNormalizer
from rl_final_project.q_functions import QLinear


@pytest.mark.usefixtures("available_envs")
def test_determinism_monte_carlo(available_envs: list) -> None:
    """Test Monte Carlo control determinism."""

    def run(env: gymnasium.Env) -> tuple:
        np.random.seed(0)
        random.seed(0)
        env = EnvironmentNormalizer(env)
        n_states = env.observation_space.shape[0]
        q_function = QLinear(
            n_actions=env.action_space.n,
            n_feat=n_states,
            discrete_scale=5,
        )
        agent = Agent(q_function, n_actions=env.action_space.n)
        control = MonteCarloControl(
            env, agent, num_episodes=10, gamma=0.9, batch_size=4
        )
        rewards = np.asarray(control.fit())

        assert isinstance(control.q_function, QLinear)

        return control.q_function.weights, rewards

    for env in available_envs:
        first_weights, first_rewards = run(env)
        second_weights, second_rewards = run(env)

        assert np.all(first_weights == second_weights)
        assert np.all(first_rewards == second_rewards)
