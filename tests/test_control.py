"""Test cases for the control module."""
import pytest

from rl_final_project.agent import Agent
from rl_final_project.control import MonteCarloControl
from rl_final_project.control import QLearningControl
from rl_final_project.control import SarsaLambdaControl
from rl_final_project.environment import EnvironmentNormalizer
from rl_final_project.q_functions import QLinear


@pytest.mark.usefixtures("available_envs")
def test_qlearning_control(available_envs: list) -> None:
    """Test q-learning control."""
    for env in available_envs:
        env = EnvironmentNormalizer(env)
        n_states = env.observation_space.shape[0]
        q_function = QLinear(
            n_actions=env.action_space.n,
            n_feat=n_states,
            discrete_scale=2,
        )
        agent = Agent(q_function, n0=1, n_actions=env.action_space.n)
        control = QLearningControl(
            env, agent, num_episodes=10, gamma=0.9, batch_size=4
        )
        control.fit()


@pytest.mark.usefixtures("available_envs")
def test_mc_control(available_envs: list) -> None:
    """Test Monte Carlo control."""
    for env in available_envs:
        env = EnvironmentNormalizer(env)
        n_states = env.observation_space.shape[0]
        q_function = QLinear(
            n_actions=env.action_space.n,
            n_feat=n_states,
            discrete_scale=2,
        )
        agent = Agent(q_function, n0=1, n_actions=env.action_space.n)
        control = MonteCarloControl(
            env, agent, num_episodes=10, gamma=0.9, batch_size=4
        )
        control.fit()


@pytest.mark.usefixtures("available_envs")
def test_sarsa_lambda(available_envs: list) -> None:
    """Test SARSA(lambda) control."""
    for env in available_envs:
        env = EnvironmentNormalizer(env)
        n_states = env.observation_space.shape[0]
        q_function = QLinear(
            n_actions=env.action_space.n,
            n_feat=n_states,
            discrete_scale=2,
        )
        agent = Agent(q_function, n0=1, n_actions=env.action_space.n)
        control = SarsaLambdaControl(
            0.8, env, agent, num_episodes=10, gamma=0.9, batch_size=4
        )
        control.fit()
