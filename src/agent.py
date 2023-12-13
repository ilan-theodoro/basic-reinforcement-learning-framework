import numpy as np
from scipy import ndimage


class Agent:
    def __init__(self, q_function, N_0=12.0, n_actions=2):
        self.q_function = q_function

        self.ϵ_s = lambda s: N_0 / (N_0 + self.q_function.N(s))
        self.ϵ_t = lambda t: N_0 / (N_0 + t)
        self.ϵ_d = lambda t: max(0.1, 1 - t / 50000)
        self.n_actions = n_actions

    def act(self, state, current_epoch=1):
        """Choose an action based on the current state"""
        action, best_reward = self.q_function.q_max(state)

        # ϵ-greedy strategy to choose the action
        t = np.random.uniform()
        if best_reward > 0 and t > self.ϵ_s(state):
            return action
        else:
            return np.random.randint(self.n_actions)