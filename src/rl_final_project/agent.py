"""Agent class definition."""
from typing import Callable
from typing import Union

import numpy as np
import torch

from rl_final_project.q_functions import QAbstractApproximation
from rl_final_project.q_functions import QTabular


class Agent:
    """Agent class based on a ϵ-greedy policy."""

    def __init__(
        self,
        q_function: Union[QTabular, QAbstractApproximation],
        n0: float = 12.0,
        n_actions: int = 2,
        eps_greedy_function: str = "dqn",
    ) -> None:
        """Agent class.

        Define an agent that chooses an action based on the current state.
        The action is chosen using an epsilon-greedy strategy.

        :param q_function: Q function used to choose the action.
        :param n0: parameter used in the epsilon-greedy function.
        :param n_actions: number of actions available in the environment.
        :param eps_greedy_function: string that defines the epsilon-greedy
         function. The options are:
         - "s": epsilon is a function of the number of times the state has
         been visited. Defined as N_0 / (N_0 + N(s)), where N(s) is the number
         of times the state s has been visited.
         - "t": epsilon is a function of the number of episodes. Defined as
         N_0 / (N_0 + t), where t is the current episode.
         - "dqn": epsilon is a function of the  number of steps as used in
         the DQN paper. Defined as 0.05 + 0.85 * exp(-steps_done / 1000).
        """
        self.q_function = q_function

        self.ϵ: Callable[[np.ndarray, int], float]

        if eps_greedy_function == "s":
            self.ϵ = lambda s, t: n0 / (n0 + self.q_function.n(s))  # type: ignore
        elif eps_greedy_function == "t":
            self.ϵ = lambda s, t: n0 / (n0 + t)  # type: ignore
        elif eps_greedy_function == "dqn":
            self.ϵ = lambda *_: 0.05 + 0.85 * np.exp(  # type: ignore
                -self.steps_done / 1000
            )
        else:
            raise ValueError("Unknown epsilon function")

        self.n_actions = n_actions
        self.steps_done = 0

    def act(self, state: np.ndarray, current_episode: int) -> int:
        """Agent action.

        Choose an action based on the current state.
        The action is chosen using an epsilon-greedy strategy.

        :param state:
            current state.
        :param current_episode:
            current episode.
        :return:
            action chosen.
        """
        with torch.no_grad():
            action, best_reward = self.q_function.q_max(state)

        self.steps_done += 1
        # ϵ-greedy strategy to choose the action
        t = np.random.uniform()
        if t > self.ϵ(state, current_episode):
            return action
        else:
            return np.random.randint(self.n_actions)
