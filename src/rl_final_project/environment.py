"""This module contains the environment tools used in the project."""
from typing import Any
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from numpy import ndarray


def _normalize(obs: np.ndarray) -> np.ndarray:
    """Normalize the state according to a sigmoid function."""
    obs = 1 / (1 + np.exp(-obs)) - 0.5
    return obs


class EnvironmentNormalizer(gym.Env):
    """Normalized environment.

    This class is used to normalize the state of an environment according to a
    sigmoid function. It scales any observation value-range to the interval
    [-1, 1].
    """

    def __init__(self, env: gym.Env) -> None:
        """Creates a new normalized environment.

        :param env: the environment to be normalized.
        """
        self.env = env
        self.reset = self.env.reset
        self.close = self.env.close
        self.render = self.env.render
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=self.env.observation_space.shape,
            dtype=np.int32,
        )
        self.scale = 1 / (1 + np.exp(-self.env.observation_space.high)) - 0.5

    def step(
        self, action: int
    ) -> tuple[ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Normalized step evaluation.

        Captures the step of the environment and normalize the observation.
        :param action: the action to be taken.
        :returns: a tuple containing the normalized observation, the reward,
            a boolean indicating if the episode is terminated, a boolean
            indicating if the episode is truncated and a dictionary containing
            extra information.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return _normalize(obs), reward, terminated, truncated, info

    @staticmethod
    def from_gym(name_id: str) -> gym.Env:
        """Create a new normalized environment from gym.

        :param name_id: name of the environment in gym.
        :returns: a new normalized environment.
        """
        return EnvironmentNormalizer(gym.make(name_id))
