from numbers import Number

import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn
import torch.nn.functional as F

from src.mo436_final_project.control import AbstractControl
from src.mo436_final_project.q_functions import QAbstractApproximation


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNFunction(QAbstractApproximation):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQN(self.n_feat, self.n_actions).to(self.device)
        self.target = DQN(self.n_feat, self.n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

    @dispatch((torch.Tensor, np.ndarray), Number)
    def __call__(self, state, action):
        return self(state)[action]

    @dispatch((torch.Tensor, np.ndarray))
    def __call__(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.policy(state)


class DQNControl(AbstractControl):
    def __init__(self, lr, τ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.q_function.device
        self.policy = self.q_function.policy
        self.target = self.q_function.target
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, amsgrad=True, weight_decay=1e-5)
        self.τ = τ
        self.criteria = nn.SmoothL1Loss()

    def update_on_episode_end(self, returns):
        pass

    def update_on_step(self, S, A, R, S_prime, A_prime, done):
        self.memory.push(S, A, S_prime, R)

        loss = self.optimize()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.τ + target_net_state_dict[key] * (1 - self.τ)
        self.target.load_state_dict(target_net_state_dict)

        return loss

    def optimize(self):
        if self.memory.batch_ready:
            batch = self.memory.sample()

            S = []
            A = []
            R = []
            S_next = []
            for b in batch:
                s, a, s_prime, r = b
                S.append(s)
                A.append(a)
                R.append(r)
                S_next.append(s_prime)

            S = torch.tensor(S, dtype=torch.float32, device=self.device)
            A = torch.tensor(A, dtype=torch.int64, device=self.device).unsqueeze(0)
            R = torch.tensor(R, dtype=torch.float32, device=self.device)

            all_indices = torch.arange(self.memory.batch_size, device=self.device)
            state_action_values = self.policy(S)[all_indices, A].squeeze(0).unsqueeze(1)

            non_final_mask = torch.tensor([False if s is None else True for s in S_next],
                                          device=self.device, dtype=torch.bool)
            non_final_next_states = torch.tensor([s for s in S_next if s is not None]).to(self.device)
            next_state_values = torch.zeros(self.memory.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.γ) + R
            loss = self.criteria(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            self.optimizer.step()

            return loss.detach().item()

        return None
