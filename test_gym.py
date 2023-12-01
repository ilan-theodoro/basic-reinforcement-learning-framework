from functools import partial

import numpy as np
import gymnasium as gym
from tqdm import tqdm

env = gym.make("FrozenLake-v1")

from collections import namedtuple
import numpy as np


class MonteCarloControl:
    def __init__(self, env, agent, num_episodes=1000, γ=0.9, discrete_scale=40):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.discrete_scale = discrete_scale
        self.γ = γ

        self.Q = agent.Q
        self.N = agent.N

    def _discretize(self, state):
        """discretize the state"""
        state = (state * self.discrete_scale).astype(int)
        return tuple(state)

    def fit(self):
        episodes_rewards = []

        pbar = tqdm(range(self.num_episodes))
        for i_episode in pbar:
            state, _ = self.env.reset()
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

        return np.mean(episodes_rewards[-100:])

    def update(self, returns):
        """Feedback the agent with the returns"""
        G_t = 0
        for t, (state, action, reward) in reversed(list(enumerate(returns))):
            G_t = self.γ * G_t + reward

            # Initialize the Q(s,a) function for the given pair (s,a)
            if state not in self.Q.keys():
                self.Q[state] = {action: 0}
                self.N[state] = {-1: 0, action: 0}
            elif action not in self.Q[state]:
                self.Q[state][action] = 0
                self.N[state][action] = 0

            # Increment the counter N(s)
            self.N[state][-1] += 1
            # Increment the counter N(s,a)
            self.N[state][action] += 1
            # Update the mean for the action-value function Q(s,a)
            self.Q[state][action] += (G_t - self.Q[state][action]) / (self.N[state][action])


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


def run(scale, N_0):
    agent = Agent(N_0=N_0)
    env = gym.make('CartPole-v1')  # , render_mode='human')
    mccontrol = MonteCarloControl(env, agent, num_episodes=20_000, γ=1)
    ma_score = mccontrol.fit()
    env.close()
    return scale, N_0, ma_score, len(agent.N)


run(7, 1)

# with Pool(20) as p:
#     results = p.starmap(run, [(scale, N_0) for scale in [1, 3, 5, 7, 10, 20] for N_0 in [1, 2, 3, 5, 10, 20]])
#
#     for (scale, N_0, ma_score, exp) in results:
#         print(f"scale: {scale}, N_0: {N_0}, ma_score: {np.mean(ma_score):.2f}, exp: {exp}")
