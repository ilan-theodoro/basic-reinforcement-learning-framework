from typing import Any, Callable, Union

import numpy as np
import torch

from src.mo436_final_project.q_functions import (QAbstractApproximation,
                                                 QTabular)


class Agent:
    def __init__(
        self,
        q_function: Union[QTabular, QAbstractApproximation],
        N_0: float = 12.0,
        n_actions: int = 2,
        ϵ_func: str = "dqn",
    ) -> None:
        self.q_function = q_function

        if ϵ_func == "s":
            self.ϵ: Callable[[np.ndarray, int], float] = lambda s, t: N_0 / (  # type: ignore
                N_0 + self.q_function.N(s)
            )
        elif ϵ_func == "t":
            self.ϵ: Callable[[np.ndarray, int], float] = lambda s, t: N_0 / (N_0 + t)  # type: ignore
        elif ϵ_func == "d":
            self.ϵ: Callable[[np.ndarray, int], float] = lambda s, t: max(  # type: ignore
                0.1, 0.9 - t / 1000
            )
        elif ϵ_func == "dqn":
            self.ϵ: Callable[[Any], float] = lambda *_: 0.05 + 0.85 * np.exp(  # type: ignore
                -self.steps_done / 1000
            )
        else:
            raise ValueError("Unknown epsilon function")

        self.n_actions = n_actions
        self.steps_done = 0

    def act(self, state: np.ndarray, current_epoch: int = 1) -> int:
        """Choose an action based on the current state"""
        with torch.no_grad():
            action, best_reward = self.q_function.q_max(state)

        self.steps_done += 1
        # ϵ-greedy strategy to choose the action
        t = np.random.uniform()
        if t > self.ϵ(state, current_epoch):
            return action
        else:
            return np.random.randint(self.n_actions)
