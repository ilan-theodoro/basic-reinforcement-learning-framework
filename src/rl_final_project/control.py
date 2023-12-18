"""Control algorithms for reinforcement learning."""
from abc import ABC
from abc import abstractmethod
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import gymnasium as gym
import numpy as np
from torch import Tensor

from rl_final_project.agent import Agent
from rl_final_project.memory import ReplayMemory
from rl_final_project.q_functions import QLinear
from rl_final_project.q_functions import QTabular


try:
    # Attempt to import the IPython module
    from IPython import get_ipython

    # Check if the code is running in a Jupyter notebook
    if "IPKernelApp" in get_ipython().config:
        # Import the tqdm notebook module
        from tqdm.notebook import tqdm
except Exception:
    # If it fails, import the terminal version of tqdm
    from tqdm import tqdm


class AbstractControl(ABC):
    """Abstract class for control algorithms."""

    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        replay_capacity: int = 10000,
        num_episodes: int = 1000,
        gamma: float = 0.9,
        batch_size: int = 32,
    ) -> None:
        """Default constructor for the AbstractControl class.

        :param env: The gym environment.
        :param agent: The agent.
        :param replay_capacity: The replay capacity for the replay memory.
        :param num_episodes: The number of episodes in which the algorithm will
         learn.
        :param gamma: The discount factor.
        :param batch_size: The batch size for the replay memory.
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.q_function = agent.q_function
        self.γ = gamma
        self.α_t: Callable[[np.ndarray, int], float] = (
            lambda s, a: 1 / self.q_function.n(s, a)
            if self.q_function.n(s, a) > 0
            else 1
        )
        self.memory = ReplayMemory(replay_capacity, batch_size)

    def fit(self) -> float:
        """Fitting loop for the control algorithm.

        It is built according to a generic reinforcement learning loop that
        iterates over the episodes and the steps of each episode. The children
        classes must implement the update_on_step and update_on_episode_end
        methods, in which they can control how the agent learns.

        :return: The mean of the last 10% of the episodes.
        """
        episodes_rewards = []

        pbar = tqdm(range(self.num_episodes))
        for i_episode in pbar:
            state, _ = self.env.reset(seed=i_episode)
            action = self.agent.act(state, current_episode=i_episode)
            returns = []
            total_reward = 0.0
            self.reset()
            while True:
                state_prime, reward, done, truncated, info = self.env.step(
                    action
                )
                reward = float(reward)
                returns.append((state, action, reward))
                total_reward += reward

                action_prime = self.agent.act(
                    state_prime, current_episode=i_episode
                )
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
                        f"Total reward for episode {i_episode:3d}: "
                        f"{total_reward}, moving average: {moving_average:.2f},"
                        f" states explored: {self.q_function.states_explored}",
                        refresh=False,
                    )
                    break

                state = state_prime
                action = action_prime

        return float(np.mean(episodes_rewards[-(self.num_episodes // 10) :]))

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
        """Abstract method to update the agent on a step."""
        pass

    @abstractmethod
    def update_on_episode_end(self, returns: list) -> Optional[float]:
        """Abstract method to update the agent on an episode end."""
        pass

    def reset(self) -> None:  # noqa: B027
        """Reset the control algorithm."""
        pass

    def optimize(self) -> Optional[float]:
        """Generic optimization of the policy network.

        It follows the rule that generally the optimization goal is designed
        to minimize the difference between the expected and the predicted
        value. It is performed on a batch of samples from the replay memory.

        :return: The loss of the optimization step. If no optimization step was
         performed, returns None.
        """
        if self.memory.batch_ready:
            batch = self.memory.sample()
            loss = 0
            updated_batch = []
            for b in batch:
                s, a, y_callable, _, α = b
                if isinstance(y_callable, Callable):  # type: ignore
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
    """Monte Carlo control algorithm implementation."""

    def update_on_step(self, *_: Any) -> Optional[float]:
        """Dummy method to avoid abstraction unimplemented error."""
        pass

    def update_on_episode_end(self, returns: list) -> Optional[float]:
        """Feedback the agent with the returns."""
        gt = 0
        for state, action, reward in reversed(returns):
            gt = self.γ * gt + reward

            # Update the mean for the action-value function Q(s,a)
            self.memory.push(state, action, gt, None, self.α_t(state, action))

        return self.optimize()


class QLearningControl(AbstractControl):
    """Q-Learning control algorithm implementation."""

    def update_on_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        _: Any,
        done: bool,
    ) -> Optional[float]:
        """Feedback the agent with the transitions on a determined step.

        :param state: The current state
        :param action: The action taken
        :param reward: The reward received
        :param next_state: The next state
        :param _: ignored
        :param done: Whether the episode is over

        :return: The loss of the optimization step. If no optimization step was
         performed, returns None.
        """
        # Compute the max Q(s',a')
        max_q = partial(self.q_function.q_max, next_state)

        # Compute a dynamic expected value according to the done flag and the
        # max Q(s',a')
        def expected(
            r: float, done: bool, max_q: Callable
        ) -> Callable[[], float]:
            return lambda: r + self.γ * max_q()[1] if not done else r

        self.memory.push(
            state,
            action,
            expected(reward, done, max_q),
            None,
            self.α_t(state, action),
        )

        return self.optimize()

    def update_on_episode_end(self, *_: Any) -> None:
        """Dummy method to avoid abstraction unimplemented error."""
        pass


class SarsaLambdaControl(AbstractControl):
    """Sarsa control algorithm implementation."""

    def __init__(self, lambda_factor: float, *args: Any, **kwargs: Any) -> None:
        """Constructor for the Sarsa control algorithm.

        :param lambda_factor: The lambda factor for the eligibility trace.
        :param args: remaining arguments from :AbstractControl:
         `rl_final_project.AbstractControl.__init__`
        :param kwargs: remaining keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.λ = lambda_factor
        self.e: Dict[Tuple, float] = {}
        self.q_old = 0
        self.z = np.zeros(self.q_function.n_feat)

    def update_on_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
    ) -> None:
        """Feedback the agent with the returns."""

        def q_prime_instantiation(
            next_state: np.ndarray, next_action: int
        ) -> Callable:
            return (
                lambda: self.q_function(next_state, next_action)
                if next_state is not None
                else 0
            )

        q_prime = q_prime_instantiation(next_state, next_action)

        def q_instantiation(
            s: np.ndarray, a: int
        ) -> Callable[[], Union[float, np.ndarray, Tensor]]:
            return lambda: self.q_function(s, a)

        q = q_instantiation(state, action)

        def δ_instantiation(
            r: float, q_prime_call: Callable, q_call: Callable
        ) -> Callable[[], float]:
            return lambda: r + self.γ * q_prime_call() - q_call()

        δ = δ_instantiation(reward, q_prime, q)

        s = self.q_function._preprocess_state(state)
        idx = self.q_function._index(s, action)
        α = self.α_t(state, action)

        # self.memory.push(S, A, q_prime, q, δ, self.α_t(S, A))

        if isinstance(self.q_function, QLinear):
            x = np.asarray(state)
            self.z = (
                self.γ * self.λ * self.z
                + (1 - α * self.γ * self.λ * self.z.T @ x) * x
            )
            self.q_function.weights[:, action] += (
                α * 0.1 * (δ() + q() - self.q_old) * self.z
                - α * (q() - self.q_old) * x
            )
            self.q_function.q_tabular._n[idx] += 1
            self.q_function.q_tabular.count_non_zero += 1
            self.q_old = q_prime()
        elif isinstance(self.q_function, QTabular):
            self.memory.push(
                state, action, q_prime, q, δ, self.α_t(state, action)
            )
            self.optimize()
        else:
            raise NotImplementedError

    def optimize(self) -> None:
        """Optimize the policy network.

        It is applied the tabular version of the algorithm.
        """
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
                        self.q_function._n[idx] += α * e
                        self.q_function._q[idx] += α * δ * e
                        self.e[idx] *= self.γ * self.λ

        return None

    def update_on_episode_end(self, *_: Any) -> None:
        """Dummy method to avoid abstraction unimplemented error."""
        pass

    def reset(self) -> None:
        """Reset the control algorithm."""
        self.q_old = 0
        self.z = np.zeros(self.q_function.n_feat)
