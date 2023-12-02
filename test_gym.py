from functools import partial
import random

import numpy as np
import gymnasium as gym
import torch
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
            _state, _ = self.env.reset(seed=i_episode)
            returns = []
            total_reward = 0
            while True:
                if not np.issubdtype(_state.dtype, np.integer):
                    state = self._discretize(_state)
                action = self.agent.act(_state, state)

                new_state, reward, done, truncated, info = self.env.step(action)
                returns.append((_state, state, action, reward))
                total_reward += reward

                if done or truncated:
                    self.update(returns)

                    episodes_rewards.append(total_reward)
                    moving_average = np.mean(episodes_rewards[-100:])

                    pbar.set_description(f"Total reward for episode {i_episode:3d}: {total_reward}, "
                                         f"moving average: {moving_average:.2f}, states explored: {len(self.N)}",
                                         refresh=False)
                    break
                #state = new_state
                _state = new_state

        return np.mean(episodes_rewards[-(self.num_episodes//10):])

    def update(self, returns):
        """Feedback the agent with the returns"""
        G_t = 0
        for t, (und_state, state, action, reward) in reversed(list(enumerate(returns))):
            G_t = self.γ * G_t + reward

            # Initialize Q(s,a) function for the given pair (s,a)
            if state not in self.N.keys():
                self.Q[state] = {action: G_t}
                self.N[state] = {-1: 1, action: 1}
            elif action not in self.Q[state]:
                self.Q[state][action] = G_t
                self.N[state][action] = 1
            else:
                # Get the learning rate
                α = self.α_t((state, action))
                # Increment the counter N(s)
                self.N[state][-1] += 1
                # Increment the counter N(s,a)
                self.N[state][action] += 1
                # Update the mean for the action-value function Q(s,a)
            self.agent.Q_hat.update(und_state, action, G_t, 0.001)
                #self.Q[state][action] += α * (G_t - self.Q[state][action]) / (self.N[state][action])


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

class FunctionApproximationNonLinear:
    def __init__(self, n_states, n_actions):
        self.model = torch.nn.Sequential(torch.nn.Linear(n_states, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 32, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(32, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, n_actions, bias=True))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.accumulator = 0
        self.accum_s = []
        self.accum_a = []
        self.accum_y = []

        from ema_pytorch import EMA
        self.ema = EMA(self.model)

    def __call__(self, state, action):
        x = torch.tensor(state, dtype=torch.float)
        return self.ema(x)[action]

    def update(self, state, action, target, α=0.1):
        self.accum_s.append(state)
        self.accum_a.append(action)
        self.accum_y.append(target)

        self.accumulator += 1
        if self.accumulator % 100 == 0:

            x = torch.tensor(self.accum_s, dtype=torch.float)
            y = torch.tensor(self.accum_y, dtype=torch.float)
            a = torch.tensor(self.accum_a, dtype=torch.long)
            y_pred = self.model(x)[np.arange(100), a]
            loss = torch.nn.functional.l1_loss(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()

            self.accum_a = []
            self.accum_s = []
            self.accum_y = []




class FunctionApproximation:
    def __init__(self, n_states, n_actions):
        self.weights = np.zeros((n_actions, n_states))

    def __call__(self, state, action):
        features = np.zeros((self.weights.shape[1]))
        #features[-1] = action * 2 - 1

        features[:len(state)] = state

        return self.weights[action].T @ features


    def update(self, state, action, target, α=0.1):
        features = np.zeros((self.weights.shape[1]))
        features[:len(state)] = state

        with open("dataset.txt", "a") as f:
            for s in features:
                f.write(f"{s}, ")
            f.write(f"{action*2 - 1}, {target}\n")

        pred = self.weights[action].T @ features
        #print(np.abs(pred - target))
        #before = self.weights.T @ features - target
        self.weights[action] += α * 0.1 * (target - pred) * features
        #print(f"before: {before}, after: {self.weights.T @ features - target}")
        #print("")

    def gradient(self, *_):
        return self.weights

class Agent:
    def __init__(self, N_0=12.0, n_actions=2, use_function_approximation=False):
        self.Q = {}
        self.N = {}
        self.ϵ_t = lambda s: N_0 / (N_0 + self.N[s][-1]) if s in self.N else 1
        self.use_function_approximation = use_function_approximation
        self.Q_hat = FunctionApproximationNonLinear(4, n_actions)
        self.n_actions = n_actions
        from linear_regressor import Model
        self.model = Model()
        self.model.load_state_dict(torch.load("model.pt"))

    def act(self, state, disc_state=None):
        """Choose an action based on the current state"""
        if self.use_function_approximation:
            return self._act_function_approximation(state, disc_state)
        else:
            return self._act(state)

    def _act_function_approximation(self, state, disc_state):
        best_reward = 0
        action = 0
        t = np.random.uniform()

        for a in range(self.n_actions):
            # if a == 0:
            #     r = self.model.linear_2(torch.tensor(state, dtype=torch.float)).item()
            # else:
            #     r = self.model.linear_1(torch.tensor(state, dtype=torch.float)).item()
            r = self.Q_hat(state, a)
            if best_reward < r:
                best_reward = r
                action = a

        # ϵ-greedy strategy to choose the action
        if best_reward > 0 and t > self.ϵ_t(disc_state):
            return action
        else:
            return np.random.randint(self.n_actions)

    def _act(self, state):
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
            return np.random.randint(self.n_actions)


from multiprocessing import Pool


def run(scale, N_0, gamma=0.9):
    # set deterministic random seed
    np.random.seed(0)
    random.seed(0)
    env = gym.make('CartPole-v1')#, render_mode='human')
    agent = Agent(N_0=N_0, use_function_approximation=True, n_actions=env.action_space.n)
    mccontrol = MonteCarloControl(env, agent, num_episodes=200_000, γ=gamma, discrete_scale=scale)
    ma_score = mccontrol.fit()
    print(agent.Q_hat.weights)
    env.close()
    return scale, N_0, ma_score, len(agent.N)


run(20, 5, gamma=1)

# with Pool(20) as p:
#     results = p.starmap(run, [(scale, N_0) for scale in [1, 3, 5, 7, 10, 20] for N_0 in [1, 2, 3, 5, 10, 20]])
#
#     for (scale, N_0, ma_score, exp) in results:
#         print(f"scale: {scale}, N_0: {N_0}, ma_score: {np.mean(ma_score):.2f}, exp: {exp}")
