import random
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, List, Optional, Union

import numpy as np
import torch
from ema_pytorch import EMA


class QTabular:
    def __init__(self, n_actions: int, n_feat: int, discrete_scale: int = 40) -> None:
        assert discrete_scale % 2 == 0, (
            "discrete_scale must be even, just to prevent possibles bugs that I am too "
            "lazy to check whether they really exist"
        )
        self.q = np.zeros((*[discrete_scale] * n_feat, n_actions), dtype=np.float32)
        self.n = np.zeros((*[discrete_scale] * n_feat, n_actions), dtype=np.float32)

        self.n_actions = n_actions
        self.n_feat = n_feat
        self.count_non_zero = 0
        self.discrete_scale = discrete_scale

    def __call__(self, state: np.ndarray, action: int) -> None:
        state = self._preprocess_state(state)
        idx = self._index(state, action)
        return self.q[idx]

    @property
    def states_explored(self) -> int:
        return self.count_non_zero

    def q_max(self, state: np.ndarray) -> tuple[int, float]:
        state = self._preprocess_state(state)
        idx_state = self._index(state)

        maximal_value = -np.inf
        maximal_set = []
        for action in range(self.n_actions):
            if maximal_value < self.q[idx_state][action]:
                maximal_value = self.q[idx_state][action]
                maximal_set = [action]
            elif maximal_value == self.q[idx_state][action]:
                maximal_set.append(action)

        action = random.choice(maximal_set)

        return action, maximal_value

    def update(
        self,
        state: np.ndarray,
        action: int,
        expected: float,
        predicted: float,
        α: float,
    ) -> None:
        state = self._preprocess_state(state)
        idx = self._index(state, action)
        if self.n[idx] == 0:
            self.count_non_zero += 1
        self.n[idx] += α
        self.q[idx] += α * (expected - predicted)

    def N(self, state: np.ndarray, action: Optional[int] = None) -> float:
        state = self._preprocess_state(state)
        idx = self._index(state)
        return np.sum(self.n[idx]) if action is None else self.n[idx][action]

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        if not np.issubdtype(state.dtype, np.integer):
            state = self._discretize(state)

        return state

    def _index(self, state: np.ndarray, action: Optional[int] = None) -> tuple:
        state_idx = state + self.discrete_scale // 2
        if action is None:
            idx = tuple(state_idx)
        else:
            idx = tuple(state_idx) + (action,)
        return idx

    def _discretize(self, state: np.ndarray) -> np.ndarray:
        """discretize the state"""
        state = (state * self.discrete_scale).astype(int)
        return state


class QAbstractApproximation(ABC):
    def __init__(self, n_actions: int, n_feat: int, discrete_scale: int = 40) -> None:
        self.n_actions = n_actions
        self.n_feat = n_feat
        self.q_tabular = QTabular(n_actions, n_feat, discrete_scale)

    @abstractmethod
    def __call__(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError

    def q_max(self, state: np.ndarray) -> tuple[int, float]:
        values = self(state)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().astype(np.int32)
        maximal_value = values.max()
        maximal_set = np.argwhere(values == maximal_value).flatten()
        action = random.choice(maximal_set)

        return action, maximal_value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.q_tabular, name)


class QLinear(QAbstractApproximation):
    def __init__(self, base_lr: float = 0.0001, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.weights = np.zeros((self.n_actions, 4))

    def __call__(self, state: np.ndarray, action: Optional[int] = None) -> np.ndarray:
        x = np.asarray(state)
        y = self.weights.T @ x
        return y[action] if action is not None else y

    def update(
        self,
        state: np.ndarray,
        action: int,
        expected: float,
        predicted: float,
        α: float,
    ) -> None:
        self.q_tabular.update(state, action, expected, predicted, α)
        α *= self.base_lr

        # predicted = self(state, action)
        assert predicted == self(state, action)

        # y_pred = self(state, action)
        self.weights[action] += α * (expected - predicted) * np.asarray(state)


class MLP(torch.nn.Module):
    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_states, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QDeep(QAbstractApproximation):
    def __init__(self, batch_size: int = 32, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = MLP(self.n_feat, self.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.accumulator = 0
        self.accum_s: List[np.ndarray] = []
        self.accum_a: List[int] = []
        self.accum_y: List[float] = []
        self.batch_size = batch_size

        self.ema = EMA(self.model)

    def __call__(self, state: np.ndarray, action: Optional[int] = None) -> torch.Tensor:
        x = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        y = self.ema(x)[0]
        return y[action] if action is not None else y

    def update(
        self, state: np.ndarray, action: int, expected: float, _: Any, α: float = 0.1
    ) -> Optional[float]:
        self.q_tabular.update(state, action, expected, _, α)
        self.accum_s.append(state)
        self.accum_a.append(action)
        self.accum_y.append(expected)

        self.accumulator += 1
        if self.accumulator % self.batch_size == 0:
            x = torch.tensor(self.accum_s, dtype=torch.float)
            y = torch.tensor(self.accum_y, dtype=torch.float)
            a = torch.tensor(self.accum_a, dtype=torch.long)
            y_pred = self.model(x)[np.arange(self.batch_size), a]
            loss = torch.nn.functional.mse_loss(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()

            self.accum_a = []
            self.accum_s = []
            self.accum_y = []

            return loss.detach().item()
        return None
