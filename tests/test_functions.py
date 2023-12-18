"""Test Q value functions implementations."""
import pytest

from rl_final_project.agent import Agent
from rl_final_project.control import MonteCarloControl
from rl_final_project.environment import EnvironmentNormalizer
from rl_final_project.q_functions import QDeep
from rl_final_project.q_functions import QTabular


@pytest.mark.usefixtures("available_envs")
def test_function_tabular(available_envs: list) -> None:
    """Test Monte Carlo control with tabular Q."""
    for env in available_envs:
        env = EnvironmentNormalizer(env)
        n_states = env.observation_space.shape[0]
        q_function = QTabular(
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
def test_function_deep(available_envs: list) -> None:
    """Test Monte Carlo control with deep Q VFA."""
    for env in available_envs:
        env = EnvironmentNormalizer(env)
        n_states = env.observation_space.shape[0]
        q_function = QDeep(
            n_actions=env.action_space.n,
            n_feat=n_states,
            discrete_scale=2,
        )
        agent = Agent(q_function, n0=1, n_actions=env.action_space.n)
        control = MonteCarloControl(
            env, agent, num_episodes=10, gamma=0.9, batch_size=4
        )
        control.fit()
