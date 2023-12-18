"""This module contains the environment tools used in the project."""
import gymnasium as gym
import numpy as np
from scipy.special import expit as sigmoid


class EnvironmentNormalizer(gym.ObservationWrapper):
    """Normalized environment.

    This class is used to normalize the state of an environment according to a
    sigmoid function. It scales any observation value-range to the interval
    [-1, 1].
    """

    def __init__(self, env: gym.Env) -> None:
        """Creates a new normalized environment.

        :param env: the environment to be normalized.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.scale_right = sigmoid(self.env.observation_space.high) - 0.5
            self.scale_left = sigmoid(self.env.observation_space.low) - 0.5
            self.mean = (self.scale_right + self.scale_left) / 2

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalized observation.

        :param observation: the observation to be normalized.
        :returns: the normalized observation.
        """
        return self._normalize(observation)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize the state according to a sigmoid function."""
        obs = sigmoid(obs) - 0.5
        obs = (obs - self.mean) / (self.scale_right - self.mean)
        return obs

    @staticmethod
    def from_gym(name_id: str) -> gym.Env:
        """Create a new normalized environment from gym.

        :param name_id: name of the environment in gym.
        :returns: a new normalized environment.
        """
        return EnvironmentNormalizer(gym.make(name_id))
