import numpy as np


class Agent:
    def __init__(self, q_function, N_0=12.0, n_actions=2):
        self.q_function = q_function

        self.ϵ_t = lambda s: N_0 / (N_0 + self.q_function.N(s))
        self.n_actions = n_actions

    def act(self, state):
        """Choose an action based on the current state"""
        action, best_reward = self.q_function.q_max(state)

        # ϵ-greedy strategy to choose the action
        t = np.random.uniform()
        if best_reward > 0 and t > self.ϵ_t(state):
            return action
        else:
            return np.random.randint(self.n_actions)