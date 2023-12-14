import gymnasium as gym
import numpy as np


class BatchRenormalization:
    def __init__(self, num_features, batch_size=1, momentum=0.01, affine=True, eps=1e-3, discrete_scale=10):
        self.running_μ = np.zeros(num_features)
        self.running_σ = np.ones(num_features)
        self.num_features = num_features
        self.batch_size = batch_size
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.n_batches_tracked = 0
        self.weight = np.ones(num_features)
        self.bias = np.zeros(num_features)
        self.step = 0

        self.training = True # TODO change it

    def observation(self, obs):
        assert len(obs.shape) == 2

        if self.training:
            batch_μ = np.mean(obs, axis=0)
            batch_σ = np.std(obs, axis=0, ddof=1) + self.eps

            r = np.clip(batch_σ / self.running_σ, 1 / self.rmax, self.rmax)
            d = np.clip((batch_μ - self.running_μ) / self.running_σ, -self.dmax, self.dmax)

            x = (obs - batch_μ) / batch_σ * r + d

            self.running_μ += self.momentum * (batch_μ - self.running_μ)
            self.running_σ += self.momentum * (batch_σ - self.running_σ)
            self.n_batches_tracked += 1
        else:
            x = (obs - self.running_μ) / self.running_σ
        # if self.affine:
        #     x = x * self.weight + self.bias
        return np.round(x * self.discrete_scale)

    @property
    def rmax(self):
        return np.clip(2 / 35000 * self.num_batches_tracked + 25 / 35, 1.0, 3.0)

    @property
    def dmax(self):
        return np.clip(5 / 20000 * self.num_batches_tracked - 25 / 20, 0.0, 5.0)


class EnvironmentDiscretizer(gym.Env):

    def __init__(self, env, discrete_scale=10):
        self.env = env
        self.discrete_scale = discrete_scale
        self.reset = self.env.reset
        self.close = self.env.close
        self.render = self.env.render
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=-self.discrete_scale, high=self.discrete_scale,
                                                shape=self.env.observation_space.shape, dtype=np.int32)
        self.scale = 1 / (1 + np.exp(-self.env.observation_space.high)) - 0.5

    def _discretize(self, obs):
        """discretize the state"""
        obs = (1 / (1 + np.exp(-obs)) - 0.5) * self.discrete_scale # (sigmoid(x) - 0.5) * scale
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._discretize(obs), reward, terminated, truncated, info
