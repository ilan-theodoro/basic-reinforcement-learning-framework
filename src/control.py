from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm


class AbstractControl(ABC):
    def __init__(self, env, agent, num_episodes=1000, γ=0.9, discrete_scale=40):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.discrete_scale = discrete_scale
        self.q_function = agent.q_function
        self.γ = γ
        self.α_t = lambda s, a: 1 / self.q_function.N(s, a) if self.q_function.N(s, a) > 0 else 1

    def fit(self):
        episodes_rewards = []

        pbar = tqdm(range(self.num_episodes))
        for i_episode in pbar:
            state, _ = self.env.reset(seed=i_episode)
            returns = []
            total_reward = 0
            while True:
                action = self.agent.act(state)

                state_prime, reward, done, truncated, info = self.env.step(action)
                returns.append((state, action, reward))
                total_reward += reward

                self.update_on_step(state, action, reward, state_prime)

                if done or truncated:
                    self.update_on_episode_end(returns)

                    episodes_rewards.append(total_reward)
                    moving_average = np.mean(episodes_rewards[-100:])

                    pbar.set_description(f"Total reward for episode {i_episode:3d}: {total_reward}, "
                                         f"moving average: {moving_average:.2f}, states explored: {self.q_function.states_explored}",
                                         refresh=False)
                    break

                state = state_prime

        return np.mean(episodes_rewards[-(self.num_episodes//10):])

    @abstractmethod
    def update_on_step(self, state, action, reward, state_prime):
        pass

    @abstractmethod
    def update_on_episode_end(self, returns):
        pass


class MonteCarloControl(AbstractControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_on_step(self, *_):
        pass

    def update_on_episode_end(self, returns):
        """Feedback the agent with the returns"""
        G_t = 0
        for t, (state, action, reward) in reversed(list(enumerate(returns))):
            G_t = self.γ * G_t + reward

            predicted = self.q_function(state, action)
            # Update the mean for the action-value function Q(s,a)
            self.q_function.update(state, action, G_t, predicted, self.α_t(state, action))


class QLearningControl(AbstractControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_on_step(self, S, A, R, S_prime):
        """Feedback the agent with the returns"""
        # Compute the max Q(s',a')
        _, max_q = self.q_function.q_max(S_prime)

        expected = R + self.γ * max_q
        predicted = self.q_function(S, A)
        self.q_function.update(S, A, expected, predicted, self.α_t(S, A))

    def update_on_episode_end(self, *_):
        pass