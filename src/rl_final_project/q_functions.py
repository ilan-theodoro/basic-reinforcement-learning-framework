"""Q functions implementations for control algorithms.

author: Ilan Theodoro.
date: December/2023.
"""
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from ema_pytorch import EMA


class QTabular:
    """Discretized tabular Q state-action-value function."""

    def __init__(
        self, n_actions: int, n_feat: int, discrete_scale: float = 40.0
    ) -> None:
        """Define a tabular Q function.

        :param n_actions: number of actions available in the environment.
        :param n_feat: number of features in the state.
        :param discrete_scale: scale in which the state will be discretized. The
         state will be multiplied by this number and then rounded to the nearest
         integer.
        """
        self._q = np.zeros(
            (*[discrete_scale * 2 + 1] * n_feat, n_actions), dtype=np.float32
        )
        self._n = np.zeros(
            (*[discrete_scale * 2 + 1] * n_feat, n_actions), dtype=np.float32
        )

        self.n_actions = n_actions
        self.n_feat = n_feat
        self.count_non_zero = 0
        self.discrete_scale = discrete_scale

    def __call__(self, state: np.ndarray, action: int) -> float:
        """Call function to get the Q value of a state-action pair."""
        state = self._preprocess_state(state)
        idx = self._index(state, action)
        return self._q[idx].item()

    @property
    def states_explored(self) -> int:
        """Number of states explored."""
        return self.count_non_zero

    def q_max(self, state: np.ndarray) -> tuple[int, float]:
        """Maximal Q value for a given state.

        Get the action and Q value for the maximal Q value for a given
        state.

        :param state: state to be evaluated.
        :return: action and Q value for the maximal Q value for a given state.
        """
        state = self._preprocess_state(state)
        idx_state = self._index(state)

        maximal_value = -np.inf
        maximal_set = []
        for action in range(self.n_actions):
            if maximal_value < self._q[idx_state][action]:
                maximal_value = self._q[idx_state][action]
                maximal_set = [action]
            elif maximal_value == self._q[idx_state][action]:
                maximal_set.append(action)

        action = np.random.choice(maximal_set)

        return action, maximal_value

    def update(
        self,
        state: np.ndarray,
        action: int,
        expected: float,
        _: float,
        α: float,
    ) -> None:
        """Update the Q value for a given state-action pair.

        It is a classical update rule for a tabular Q function. The update rule
        follows the principle of minimizing the error between the expected and
        the predicted value.

        :param state: state associated.
        :param action: action associated.
        :param expected: expected value for the evaluated policy.
        :param _: ignored.
        :param α: learning rate.
        """
        predicted = self(state, action)
        state = self._preprocess_state(state)
        idx = self._index(state, action)
        if self._n[idx] == 0:
            self.count_non_zero += 1
        self._n[idx] += α
        self._q[idx] += α * (expected - predicted)

    def n(self, state: np.ndarray, action: Optional[int] = None) -> float:
        """Number of times a state or a state-action pair was visited.

        :param state: state to be evaluated.
        :param action: action to be evaluated. If None, returns the number of
         times the state was visited.
        :return: number of times a state or a state-action pair was visited.
        """
        state = self._preprocess_state(state)
        idx = self._index(state)
        return np.sum(self._n[idx]) if action is None else self._n[idx][action]

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Preprocess the state to be used in the tabular Q function."""
        if not np.issubdtype(state.dtype, np.integer):
            state = self._discretize(state)

        return state

    def _index(self, state: np.ndarray, action: Optional[int] = None) -> tuple:
        """Get the index of a state or a state-action pair.

        It is solved according to the injective mapping of a state the index
        in the Q table.

        :param state: state to be evaluated.
        :param action: action to be evaluated. If None, returns the index of the
         state.
        :return: index of a state or a state-action pair.
        """
        state_idx = state + self.discrete_scale
        if any(state_idx < 0):
            raise ValueError("Index is negative")
        if action is None:
            idx = tuple(state_idx)
        else:
            idx = tuple(state_idx) + (action,)
        return idx

    def _discretize(self, state: np.ndarray) -> np.ndarray:
        """Discretize the state.

        :param state: state to be discretized. It must be a numpy array.
        """
        state = (state * self.discrete_scale).astype(int)
        return state


class QAbstractApproximation(ABC):
    """Abstract class for Q function approximations."""

    def __init__(
        self, n_actions: int, n_feat: int, discrete_scale: int = 40
    ) -> None:
        """Define generic Q function approximation.

        It includes a tabular Q function in cases where discrete evaluation is
        necessary.

        :param n_actions: number of actions available in the environment.
        :param n_feat: number of features in the state.
        :param discrete_scale: scale in which the state will be discretized. The
         state will be multiplied by this number and then rounded to the nearest
         integer.
        """
        self.n_actions = n_actions
        self.n_feat = n_feat
        self.q_tabular = QTabular(n_actions, n_feat, discrete_scale)

    @abstractmethod
    def __call__(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Call function to get the Q value of a state-action pair."""
        raise NotImplementedError

    def q_max(self, state: np.ndarray) -> tuple[int, float]:
        """Maximal Q value for a given state.

        Get the action and Q value for the maximal Q value for a given state.

        :param state: state to be evaluated.
        :return: action and Q value for the maximal Q value for a given state.
        """
        values = self(state)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().astype(np.int32)
        maximal_value = values.max()
        maximal_set = np.argwhere(values == maximal_value).flatten()
        action = np.random.choice(maximal_set)

        return action, maximal_value.item()

    def __getattr__(self, name: str) -> Any:
        """Get missing attributes from the tabular Q function."""
        return getattr(self.q_tabular, name)


class QLinear(QAbstractApproximation):
    """Linear Q function value approximation."""

    def __init__(
        self, base_lr: float = 0.0001, *args: Any, **kwargs: Any
    ) -> None:
        """Define a linear Q function approximation.

        It is an unbiased linear approximation of the Q function. It is defined
        as Q(s) = W^T x(s), where W is the weight vector mapping the state
        space to the action space and x(s) is the input sample from the state
        space.

        :param base_lr: base learning rate to scale the learning rate of the
         control algorithm.
        :param args: arguments to be passed to the QAbstractApproximation.
        :param kwargs: keyword arguments to be passed to the
         QAbstractApproximation.
        """
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.weights = np.zeros((self.n_feat, self.n_actions))

    def __call__(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> np.ndarray:
        """Call function to get the Q value of a state-action pair."""
        x = np.asarray(state)
        y = self.weights.T @ x
        return y[action] if action is not None else y

    def update(
        self,
        state: np.ndarray,
        action: int,
        expected: float,
        _: float,
        α: float,
    ) -> None:
        """Update the Q linear approximation for a given state-action pair.

        It is a classical update rule for a linear Q function approximation. The
        update rule follows the principle of minimizing the error between the
        expected and the predicted value.

        :param state: state associated.
        :param action: action associated.
        :param expected: expected value for the evaluated policy.
        :param _: ignored
        :param α: learning rate.
        """
        predicted = self(state, action).item()
        self.q_tabular.update(state, action, expected, predicted, α)
        α *= self.base_lr

        self.weights[:, action] += (
            α * (expected - predicted) * np.asarray(state)
        )


class MLP(torch.nn.Module):
    """Non-linear approximation torch MLP module."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        """Define an MLP for the DQN.

        :param n_states: number of states.
        :param n_actions: number of actions.
        """
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_states, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        return self.model(x)


class QDeep(QAbstractApproximation):
    """Non-linear Q function approximation."""

    def __init__(self, batch_size: int = 32, *args: Any, **kwargs: Any) -> None:
        """Define a non-linear Q function approximation.

        :param batch_size: batch size to be used in the training of the neural
         network.
        :param args: arguments to be passed to the QAbstractApproximation.
        :param kwargs: keyword arguments to be passed to the
         QAbstractApproximation.
        """
        super().__init__(*args, **kwargs)

        self.model = MLP(self.n_feat, self.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.accumulator = 0
        self.accum_s: List[np.ndarray] = []
        self.accum_a: List[int] = []
        self.accum_y: List[float] = []
        self.batch_size = batch_size

        self.ema = EMA(self.model)

    def __call__(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> torch.Tensor:
        """Call function to get the Q value of a state-action pair."""
        x = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        y = self.ema(x)[0]
        return y[action] if action is not None else y

    def update(
        self,
        state: np.ndarray,
        action: int,
        expected: float,
        _: Any,
        α: float,
    ) -> Optional[float]:
        """Try to update the non-linear Q function approximation.

        'Try' in the sense that it will only update the neural network if the
        batch size is reached. Otherwise, it will accumulate the values and
        return None.

        :param state: state to be updated.
        :param action: action to be updated.
        :param expected: expected value for the evaluated policy.
        :param _: not used.
        :param α: learning rate from the control algorithm. Currently it is
         ignored to optimized. But it remains as a TODO to be implemented.
        :return: loss of the neural network if the batch size is reached.
        """
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
