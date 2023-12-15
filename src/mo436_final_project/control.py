from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm

from src.mo436_final_project.agent import Agent
from src.mo436_final_project.memory import ReplayMemory
from src.mo436_final_project.q_functions import QLinear, QTabular


class AbstractControl(ABC):
    """Abstract class for control algorithms"""

    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        replay_capacity: int = 10000,
        num_episodes: int = 1000,
        γ: float = 0.9,
        batch_size: int = 32,
    ) -> None:
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.q_function = agent.q_function
        self.γ = γ
        self.α_t: Callable[[np.ndarray, int], float] = (
            lambda s, a: 1 / self.q_function.N(s, a)
            if self.q_function.N(s, a) > 0
            else 1
        )
        self.memory = ReplayMemory(replay_capacity, batch_size)

    def fit(self) -> float:
        episodes_rewards = []

        pbar = tqdm(range(self.num_episodes))
        for i_episode in pbar:
            state, _ = self.env.reset(seed=i_episode)
            action = self.agent.act(state, current_epoch=i_episode)
            returns = []
            total_reward = 0
            self.reset()
            while True:
                state_prime, reward, done, truncated, info = self.env.step(action)
                returns.append((state, action, reward))
                total_reward += reward

                action_prime = self.agent.act(state_prime, current_epoch=i_episode)
                loss_updated = self.update_on_step(
                    state,
                    action,
                    reward,
                    state_prime if not done else None,
                    action_prime,
                    done,
                )
                if loss_updated is not None:
                    pbar.set_postfix(loss=loss_updated)

                if done or truncated:
                    loss_updated = self.update_on_episode_end(returns)
                    if loss_updated is not None:
                        pbar.set_postfix(loss=loss_updated)

                    episodes_rewards.append(total_reward)
                    moving_average = np.mean(episodes_rewards[-100:])

                    pbar.set_description(
                        f"Total reward for episode {i_episode:3d}: {total_reward}, "
                        f"moving average: {moving_average:.2f}, states explored: {self.q_function.states_explored}",
                        refresh=False,
                    )
                    break

                state = state_prime
                action = action_prime

        return np.mean(episodes_rewards[-(self.num_episodes // 10) :])

    @abstractmethod
    def update_on_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        state_prime: np.ndarray,
        action_prime: int,
        done: bool,
    ) -> Optional[float]:
        pass

    @abstractmethod
    def update_on_episode_end(self, returns: list) -> Optional[float]:
        pass

    def reset(self) -> None:
        pass

    def optimize(self) -> Optional[float]:
        if self.memory.batch_ready:
            batch = self.memory.sample()
            loss = 0
            updated_batch = []
            for b in batch:
                s, a, y_callable, _, α = b
                if isinstance(y_callable, Callable):
                    y = y_callable()
                else:
                    y = y_callable
                updated_batch.append((s, a, y, _, α))

            for ub in updated_batch:
                _loss = self.q_function.update(*ub)
                if _loss is not None:
                    loss += _loss
            return loss / len(batch)
        else:
            return None


class MonteCarloControl(AbstractControl):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def update_on_step(self, *_: Any) -> Optional[float]:
        pass

    def update_on_episode_end(self, returns: list) -> Optional[float]:
        """Feedback the agent with the returns"""
        G_t = 0
        for t, (state, action, reward) in reversed(list(enumerate(returns))):
            G_t = self.γ * G_t + reward

            predicted = self.q_function(state, action)
            # Update the mean for the action-value function Q(s,a)
            self.memory.push(state, action, G_t, predicted, self.α_t(state, action))

        return self.optimize()


class QLearningControl(AbstractControl):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def update_on_step(
        self, S: np.ndarray, A: int, R: float, S_prime: np.ndarray, _: Any, done: bool
    ) -> Optional[float]:
        """Feedback the agent with the returns"""
        # Compute the max Q(s',a')
        max_q = partial(self.q_function.q_max, S_prime)

        def expected(R: float, done: bool, max_q: Callable) -> Callable[[], float]:
            return lambda: R + self.γ * max_q()[1] if not done else R

        self.memory.push(S, A, expected(R, done, max_q), None, self.α_t(S, A))

        return self.optimize()

    def update_on_episode_end(self, *_: Any) -> None:
        pass


class SarsaLambdaControl(AbstractControl):
    def __init__(self, λ: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.λ = λ
        self.e: Dict[Tuple, float] = {}
        self.q_old = 0
        self.z = np.zeros(self.q_function.n_feat)

    def update_on_step(
        self,
        S: np.ndarray,
        A: int,
        R: float,
        S_prime: np.ndarray,
        A_prime: int,
        done: bool,
    ) -> None:
        """Feedback the agent with the returns"""

        # Compute the max Q(s',a')
        def q_prime_instantiation(S_prime: np.ndarray, A_prime: int) -> Callable:
            return (
                lambda: self.q_function(S_prime, A_prime) if S_prime is not None else 0
            )

        q_prime = q_prime_instantiation(S_prime, A_prime)

        def q_instantiation(S: np.ndarray, A: int) -> Callable[[], np.ndarray | Tensor]:
            return lambda: self.q_function(S, A)

        q = q_instantiation(S, A)

        δ = lambda: R + self.γ * q_prime() - q()

        s = self.q_function._preprocess_state(S)
        idx = self.q_function._index(s, A)
        α = self.α_t(S, A)

        # self.memory.push(S, A, q_prime, q, δ, self.α_t(S, A))

        if isinstance(self.q_function, QLinear):
            x = np.asarray(S)
            self.z = (
                self.γ * self.λ * self.z + (1 - α * self.γ * self.λ * self.z.T @ x) * x
            )
            self.q_function.weights[A] += (
                α * 0.1 * (δ() + q() - self.q_old) * self.z - α * (q() - self.q_old) * x
            )
            self.q_function.q_tabular.n[idx] += 1
            self.q_old = q_prime()
        elif isinstance(self.q_function, QTabular):
            self.memory.push(S, A, q_prime, q, δ, self.α_t(S, A))
            self.optimize()
        else:
            raise NotImplementedError

    def optimize(self) -> None:
        if isinstance(self.q_function, QTabular):
            if self.memory.batch_ready:
                batch = self.memory.sample()
                updated_batch = []
                for b in batch:
                    s, a, q_prime, q, δ, α = b
                    updated_batch.append((s, a, q_prime(), q(), δ(), α))

                for ub in updated_batch:
                    s, a, q_prime, q, δ, α = ub
                    s = self.q_function._preprocess_state(s)
                    idx = self.q_function._index(s, a)

                    if idx not in self.e:
                        self.e[idx] = 0
                    self.e[idx] += 1
                    for idx, e in self.e.items():
                        self.q_function.n[idx] += α * e
                        self.q_function.q[idx] += α * δ * e
                        self.e[idx] *= self.γ * self.λ

        return None

    def update_on_episode_end(self, *_: Any) -> None:
        pass

    def reset(self) -> None:
        self.q_old = 0
        self.z = np.zeros(self.q_function.n_feat)
