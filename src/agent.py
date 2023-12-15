import numpy as np
import torch
from scipy import ndimage


class Agent:
    def __init__(self, q_function, N_0=12.0, n_actions=2, ϵ_func="dqn"):
        self.q_function = q_function

        if ϵ_func == "s":
            self.ϵ = lambda s, t: N_0 / (N_0 + self.q_function.N(s))
        elif ϵ_func == "t":
            self.ϵ = lambda s, t: N_0 / (N_0 + t)
        elif ϵ_func == "d":
            self.ϵ = lambda s, t: max(0.1, 0.9 - t / 1000)
        elif ϵ_func == "dqn":
            self.ϵ = lambda *_: 0.05 + 0.85 * np.exp(-self.steps_done / 1000)
        else:
            raise ValueError("Unknown epsilon function")

        self.n_actions = n_actions
        self.steps_done = 0

    def act(self, state, current_epoch=1):
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