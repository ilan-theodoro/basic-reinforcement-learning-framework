"""Deep Q Network implementation.

The code is based on the following sources:
- https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
from typing import Any
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_final_project.control import AbstractControl
from rl_final_project.q_functions import QAbstractApproximation


class DQN(nn.Module):
    """Deep Q Network Torch Module."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        """Deep Q Network.

        Implements a simple MLP with 2 hidden layers with 128 neurons each.

        :param n_observations: Number of observations
        :param n_actions: Number of actions
        """
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNFunction(QAbstractApproximation):
    """Deep Q Network Q Function Approximation."""

    def __init__(self, batch_size: int, *args: Any, **kwargs: Any) -> None:
        """Deep Q Network Q Function Approximation.

        :param batch_size: Batch size
        :param args: Remaining arguments for the QAbstractApproximation
        :param kwargs: Remaining keyword arguments for the
         QAbstractApproximation
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.policy = DQN(self.n_feat, self.n_actions).to(self.device)
        self.target = DQN(self.n_feat, self.n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

    def __call__(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass of the network.

        It converts the state to a torch tensor and passes it to the network.
        If an action is provided, it returns the Q value for that action.

        :param state: The state to evaluate
        :param action: The action to evaluate
        :return: The Q value for the state-action pair or the Q values for all
         actions if no action is provided
        """
        state_tsr = torch.tensor(state, dtype=torch.float32, device=self.device)
        if action is not None:
            return self.policy(state_tsr)[action]
        else:
            return self.policy(state_tsr)


class DQNControl(AbstractControl):
    """DQN Control algorithm.

    Implements the DQN algorithm as described in the paper:
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(
        self, lr: float, tau: float, *args: Any, **kwargs: Any
    ) -> None:
        """Constructor for the DQN control algorithm.

        :param lr: Learning rate
        :param tau: Soft update parameter
        :param args: remaining arguments
        :param kwargs: remaining keyword arguments
        """
        super().__init__(*args, **kwargs)
        if not isinstance(self.q_function, DQNFunction):
            raise ValueError("Q function must be a DQN")
        self.device = self.q_function.device
        self.policy = self.q_function.policy
        self.target = self.q_function.target
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=lr, amsgrad=True
        )
        self.τ = tau
        self.criteria = nn.SmoothL1Loss()

    def update_on_episode_end(self, returns: list) -> Optional[float]:
        """Feedback the agent with the returns."""
        pass

    def update_on_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
    ) -> Optional[float]:
        """Feedback the agent with the transitions on a determined step.

        :param state: The current state
        :param action: The action taken
        :param reward: The reward received
        :param next_state: The next state
        :param next_action: The next action
        :param done: Whether the episode is over
        :return: The loss of the optimization step.
        """
        self.memory.push(state, action, next_state, reward)

        loss = self.optimize()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.τ + target_net_state_dict[key] * (1 - self.τ)
        self.target.load_state_dict(target_net_state_dict)

        return loss

    def optimize(self) -> Optional[float]:
        """Optimize the policy network.

        :return: The loss of the optimization step. If no optimization step was
         performed, returns None.
        """
        if self.memory.batch_ready:
            batch = self.memory.sample()

            s_lst = []
            a_lst = []
            r_lst = []
            s_next_lst = []
            for b in batch:
                sb, sa, sb_prime, rb = b
                s_lst.append(sb)
                a_lst.append(sa)
                r_lst.append(rb)
                s_next_lst.append(sb_prime)

            s = torch.tensor(s_lst, dtype=torch.float32, device=self.device)
            a = torch.tensor(
                a_lst, dtype=torch.int64, device=self.device
            ).unsqueeze(0)
            r = torch.tensor(r_lst, dtype=torch.float32, device=self.device)

            all_indices = torch.arange(
                self.memory.batch_size, device=self.device
            )
            state_action_values = (
                self.policy(s)[all_indices, a].squeeze(0).unsqueeze(1)
            )

            non_final_mask = torch.tensor(
                [False if s is None else True for s in s_next_lst],
                device=self.device,
                dtype=torch.bool,
            )
            non_final_next_states = torch.tensor(
                [s for s in s_next_lst if s is not None]
            ).to(self.device)
            next_state_values = torch.zeros(
                self.memory.batch_size, device=self.device
            )
            with torch.no_grad():
                next_state_values[non_final_mask] = (
                    self.target(non_final_next_states).max(1).values
                )

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.γ) + r
            loss = self.criteria(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            self.optimizer.step()

            return loss.detach().item()

        return None
