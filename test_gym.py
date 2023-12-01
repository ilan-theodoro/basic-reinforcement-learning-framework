from functools import partial
import random

import numpy as np
import gymnasium as gym
from tqdm import tqdm

env = gym.make("FrozenLake-v1")

from collections import namedtuple
import numpy as np

class AbstractControl:
    def __init__(self, env, agent, num_episodes=1000, γ=0.9, discrete_scale=40):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.discrete_scale = discrete_scale
        self.γ = γ
        self.α_t = lambda st: 1 / self.N[st[0]][st[1]]

        self.Q = agent.Q
        self.N = agent.N

    def _discretize(self, state):
        """discretize the state"""
        state = (state * self.discrete_scale).astype(int)
        return tuple(state)


class MonteCarloControl(AbstractControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self):
        episodes_rewards = []

        pbar = tqdm(range(self.num_episodes))
        for i_episode in pbar:
            state, _ = self.env.reset(seed=i_episode)
            returns = []
            total_reward = 0
            while True:
                if not np.issubdtype(state.dtype, np.integer):
                    state = self._discretize(state)
                action = self.agent.act(state)

                new_state, reward, done, _, info = self.env.step(action)
                returns.append((state, action, reward))
                total_reward += reward

                if done:
                    self.update(returns)

                    episodes_rewards.append(total_reward)
                    moving_average = np.mean(episodes_rewards[-100:])

                    pbar.set_description(f"Total reward for episode {i_episode:3d}: {total_reward}, "
                                         f"moving average: {moving_average:.2f}, states explored: {len(self.N)}",
                                         refresh=False)
                    break
                state = new_state

        return np.mean(episodes_rewards[-(self.num_episodes//10):])

    def update(self, returns):
        """Feedback the agent with the returns"""
        G_t = 0
        for t, (state, action, reward) in reversed(list(enumerate(returns))):
            G_t = self.γ * G_t + reward

            # Initialize Q(s,a) function for the given pair (s,a)
            if state not in self.Q.keys():
                self.Q[state] = {action: G_t}
                self.N[state] = {-1: 1, action: 1}
            elif action not in self.Q[state]:
                self.Q[state][action] = G_t
                self.N[state][action] = 1
            else:
                # Get the learning rate
                α = self.α_t((state, action))
                # Increment the counter N(s)
                self.N[state][-1] += α
                # Increment the counter N(s,a)
                self.N[state][action] += α
                # Update the mean for the action-value function Q(s,a)
                self.Q[state][action] += α * (G_t - self.Q[state][action]) / (self.N[state][action])


class QLearningControl(AbstractControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self):
        episodes_rewards = []

        pbar = tqdm(range(self.num_episodes))
        for i_episode in pbar:
            state, _ = self.env.reset(seed=i_episode)
            returns = []
            total_reward = 0
            while True:
                if not np.issubdtype(state.dtype, np.integer):
                    state = self._discretize(state)
                action = self.agent.act(state)

                new_state, reward, done, _, info = self.env.step(action)
                returns.append((state, action, reward))
                total_reward += reward
                self.update(state, action, reward, new_state)

                if done:
                    episodes_rewards.append(total_reward)
                    moving_average = np.mean(episodes_rewards[-100:])

                    pbar.set_description(f"Total reward for episode {i_episode:3d}: {total_reward}, "
                                         f"moving average: {moving_average:.2f}, states explored: {len(self.N)}",
                                         refresh=False)
                    break
                state = new_state

        return np.mean(episodes_rewards[-(self.num_episodes//10):])

    def update(self, S, A, R, S_prime):
        """Feedback the agent with the returns"""
        if not np.issubdtype(S_prime.dtype, np.integer):
            S_prime = self._discretize(S_prime)

        # Compute the max Q(s',a')
        max_q = 0
        if S_prime in self.Q.keys():
            max_q = max(self.Q[S_prime].values())

        # Initialize Q(s,a) function for the given pair (s,a)
        if S not in self.Q.keys():
            self.Q[S] = {A: R + self.γ * max_q}
            self.N[S] = {-1: 1, A: 1}
        elif A not in self.Q[S]:
            self.Q[S][A] = R + self.γ * max_q
            self.N[S][A] = 1
        else:
            # Get the learning rate
            α = self.α_t((S, A))
            # Increment the counter N(s)
            self.N[S][-1] += α
            # Increment the counter N(s,a)
            self.N[S][A] += α
            # Update the mean for the action-value function Q(s,a)
            self.Q[S][A] += α * (R + self.γ * max_q - self.Q[S][A])

class Agent:
    def __init__(self, N_0=12.0):
        self.Q = {}
        self.N = {}
        self.ϵ_t = lambda s: N_0 / (N_0 + self.N[s][-1])

    def act(self, state):
        best_reward = 0
        action = 0
        t = np.random.uniform()

        if self.Q.get(state) is not None:
            for a, r in self.Q[state].items():
                if best_reward < r:
                    best_reward = r
                    action = a

        # ϵ-greedy strategy to choose the action
        if best_reward > 0 and t > self.ϵ_t(state):
            return action
        else:
            return np.random.randint(2)


from multiprocessing import Pool


def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    agent = Agent(N_0=N_0)
    env = gym.make('CartPole-v1')  # , render_mode='human')
    mccontrol = QLearningControl(env, agent, num_episodes=20_000, γ=gamma, discrete_scale=scale)
    ma_score = mccontrol.fit()
    env.close()
    return scale, N_0, ma_score, len(agent.N)


run(30, 5, gamma=0.99)

# with Pool(20) as p:
#     results = p.starmap(run, [(scale, N_0) for scale in [1, 3, 5, 7, 10, 20] for N_0 in [1, 2, 3, 5, 10, 20]])
#
#     for (scale, N_0, ma_score, exp) in results:
#         print(f"scale: {scale}, N_0: {N_0}, ma_score: {np.mean(ma_score):.2f}, exp: {exp}")
