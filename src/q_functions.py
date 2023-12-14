import random

import numpy as np
from abc import ABC, abstractmethod

import torch


class QTabular:
    def __init__(self, n_actions, n_feat, discrete_scale=40):
        assert discrete_scale % 2 == 0, ("discrete_scale must be even, just to prevent possibles bugs that I am too "
                                         "lazy to check whether they really exist")
        self.q = np.zeros((*[discrete_scale] * n_feat, n_actions), dtype=np.float32)
        self.n = np.zeros((*[discrete_scale] * n_feat, n_actions), dtype=np.float32)

        self.n_actions = n_actions
        self.n_feat = n_feat
        self.discrete_scale = discrete_scale

    def __call__(self, state, action):
        state = self._preprocess_state(state)
        idx = self._index(state, action)
        return self.q[idx]

    @property
    def states_explored(self):
        return np.count_nonzero(self.n)

    def q_max(self, state):
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

    def update(self, state, action, expected, predicted, α):
        state = self._preprocess_state(state)
        idx = self._index(state, action)
        self.n[idx] += α
        self.q[idx] += α * (expected - predicted)

    def N(self, state, action=None):
        state = self._preprocess_state(state)
        idx = self._index(state)
        return np.sum(self.n[idx]) if action is None else self.n[idx][action]

    def _preprocess_state(self, state):
        if not np.issubdtype(state.dtype, np.integer):
            state = self._discretize(state)
        #if state not in self._Q:
        #    self._Q[state] = {action: 0 for action in range(self.n_actions)}
       #     self._N[state] = {action: 0 for action in range(-1, self.n_actions)}

        return state

    def _index(self, state, action=None):
        state_idx = state + self.discrete_scale // 2
        if action is None:
            idx = tuple(state_idx)
        else:
            idx = tuple(state_idx) + (action,)
        return idx

    def _discretize(self, state):
        """discretize the state"""
        state = (state * self.discrete_scale).astype(int)
        return state


class QAbstractApproximation(ABC):
    def __init__(self, n_actions, n_feat, discrete_scale=40):
        self.n_actions = n_actions
        self.n_feat = n_feat
        self.q_tabular = QTabular(n_actions, n_feat, discrete_scale)

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

    def __getattr__(self, name):
        return getattr(self.q_tabular, name)


class QLinear(QAbstractApproximation):
    def __init__(self, base_lr=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.weights = np.zeros((self.n_actions, 4))

    def __call__(self, state, action):
        x = np.asarray(state)
        return self.weights[action].T @ x

    def update(self, state, action, expected, predicted, α):
        self.q_tabular.update(state, action, expected, predicted, α)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.accumulator = 0
        self.accum_s = []
        self.accum_a = []
        self.accum_y = []

        from ema_pytorch import EMA
        self.ema = EMA(self.model)

    def __call__(self, state, action):
        x = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        y = self.ema(x)[0, action]
        return y

    def update(self, state, action, expected, _, α=0.1):
        self.q_tabular.update(state, action, expected, None, α)
        self.accum_s.append(state)
        self.accum_a.append(action)
        self.accum_y.append(expected)

        self.accumulator += 1
        if self.accumulator % 100 == 0:
            x = torch.tensor(self.accum_s, dtype=torch.float)
            y = torch.tensor(self.accum_y, dtype=torch.float)
            a = torch.tensor(self.accum_a, dtype=torch.long)
            y_pred = self.model(x)[np.arange(100), a]
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
