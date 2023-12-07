import random

import numpy as np
from abc import ABC, abstractmethod

import torch


class QTabular:
    def __init__(self, n_actions, discrete_scale=40):
        self._Q = {}
        self._N = {}
        self.n_actions = n_actions
        self.discrete_scale = discrete_scale

    def __call__(self, state, action):
        state = self._preprocess_state(state)
        return self._Q[state][action]

    @property
    def states_explored(self):
        return len(self._N)

    def q_max(self, state):
        state = self._preprocess_state(state)

        maximal_value = -np.inf
        maximal_set = []
        for action in range(self.n_actions):
            if maximal_value < self._Q[state][action]:
                maximal_value = self._Q[state][action]
                maximal_set = [action]
            elif maximal_value == self._Q[state][action]:
                maximal_set.append(action)

        action = random.choice(maximal_set)

        return action, maximal_value

    def update(self, state, action, expected, _, α):
        state = self._preprocess_state(state)
        self._N[state][-1] += α
        self._N[state][action] += α
        self._Q[state][action] += α * expected

    def N(self, state, action=-1):
        state = self._preprocess_state(state)
        return self._N[state][action]

    def _preprocess_state(self, state):
        if not np.issubdtype(state.dtype, np.integer):
            state = self._discretize(state)
        if state not in self._Q:
            self._Q[state] = {action: 0 for action in range(self.n_actions)}
            self._N[state] = {action: 0 for action in range(-1, self.n_actions)}

        return state

    def _discretize(self, state):
        """discretize the state"""
        state = (state * self.discrete_scale).astype(int)
        return tuple(state)


class QAbstractApproximation(ABC):
    def __init__(self, n_actions, discrete_scale=40):
        self.n_actions = n_actions
        self.q_tabular = QTabular(n_actions, discrete_scale)

    @abstractmethod
    def __call__(self, state, action):
        pass

    @property
    def states_explored(self):
        return self.q_tabular.states_explored

    def N(self, state, action=-1):
        return self.q_tabular.N(state, action)

    def q_max(self, state):
        maximal_value = -np.inf
        maximal_set = []
        for action in range(self.n_actions):
            q_value = self(state, action)
            if maximal_value < q_value:
                maximal_value = q_value
                maximal_set = [action]
            elif maximal_value == q_value:
                maximal_set.append(action)

        action = random.choice(maximal_set)

        return action, maximal_value


class QLinear(QAbstractApproximation):
    def __init__(self, n_actions, base_lr=0.0001, **kwargs):
        super().__init__(n_actions, **kwargs)
        self.base_lr = base_lr
        self.weights = np.zeros((self.n_actions, 4))

    def __call__(self, state, action):
        x = np.asarray(state)
        return self.weights[action].T @ x

    def update(self, state, action, expected, predicted, α):
        self.q_tabular.update(state, action, 0, None, α)
        α *= self.base_lr

        assert predicted == self(state, action)

        #y_pred = self(state, action)
        self.weights[action] += α * (expected - predicted) * np.asarray(state)


class MLP(torch.nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(n_states, 16, bias=True), torch.nn.ReLU(),
                                         torch.nn.Linear(16, 32, bias=True), torch.nn.ReLU(),
                                         torch.nn.Linear(32, 16, bias=True), torch.nn.ReLU(),
                                         torch.nn.Linear(16, n_actions, bias=True))

    def forward(self, x):
        return self.model(x)

class QDeep(QAbstractApproximation):
    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_actions, **kwargs)

        self.model = MLP(n_states, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.accumulator = 0
        self.accum_s = []
        self.accum_a = []
        self.accum_y = []

        from ema_pytorch import EMA
        self.ema = EMA(self.model)

    def __call__(self, state, action):
        x = torch.tensor(state, dtype=torch.float)
        return self.ema(x)[action]

    def update(self, state, action, expected, _, α=0.1):
        self.q_tabular.update(state, action, 0, None, α)
        self.accum_s.append(state)
        self.accum_a.append(action)
        self.accum_y.append(expected)

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
