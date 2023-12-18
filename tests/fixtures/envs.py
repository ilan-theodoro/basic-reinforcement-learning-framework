"""Fixtures for testing gym environments."""
from typing import List

import gymnasium as gym
import pytest
from gymnasium import Env


@pytest.fixture()
def available_envs() -> List[Env]:
    """Return a gym environment."""
    return [
        gym.make("CartPole-v1"),
        gym.make("MountainCar-v0"),
        gym.make("Acrobot-v1"),
    ]
