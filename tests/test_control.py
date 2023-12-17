"""Test cases for the control module."""
import gymnasium as gym
import pytest

from rl_final_project.agent import Agent
from rl_final_project.control import MonteCarloControl
from rl_final_project.control import QLearningControl
from rl_final_project.control import SarsaLambdaControl


@pytest.fixture()
def control_algorithms_instances(env: gym.Env, agent: Agent) -> list:
    """Return a list of control algorithms instances."""
    return [
        MonteCarloControl(env, agent),
        SarsaLambdaControl(0.8, env, agent),
        QLearningControl(env, agent),
    ]
