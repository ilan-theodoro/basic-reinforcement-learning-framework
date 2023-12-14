import gymnasium as gym
import numpy as np


def _discretize(obs):
    """discretize the state"""
    obs = (1 / (1 + np.exp(-obs)) - 0.5)
    return obs


class EnvironmentNormalizer(gym.Env):
    def __init__(self, env):
        self.env = env
        self.reset = self.env.reset
        self.close = self.env.close
        self.render = self.env.render
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                                shape=self.env.observation_space.shape, dtype=np.int32)
        self.scale = 1 / (1 + np.exp(-self.env.observation_space.high)) - 0.5

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return _discretize(obs), reward, terminated, truncated, info

    @staticmethod
    def from_gym(name_id):
        return EnvironmentNormalizer(gym.make(name_id))
