import numpy as np
import gym
env = gym.make("FrozenLake-v1")


np.random.seed(0)

class MonteCarloControl:
    def __init__(self, num_episodes=1000):
        self.env = MNISTEnvironment()
        self.Q = None
        self.returns = None
        self.π = None
        self.γ = 0.9
        self.ε = 0.1
        self.α = 0.1
        self.num_episodes = num_episodes

    def run(self):
        self.reset()
        for i in range(self.num_episodes):
            state_action_pairs = self.run_episode()
            G = self.compute_returns(state_action_pairs)
            self.update_policy(G)

        π_star = np.argmax(self.Q, axis=2)

        return self.Q, π_star

    # def update(self, state, action, reward):
    #     self.returns[state[0], state[1], action] += reward
    #     self.Q[state[0], state[1], action] = self.returns[state[0], state[1], action] / self.π[state[0], state[1], action]
    #
    def run_episode(self):
        self.env.reset()

        done = False
        state = self.env.current_position
        action = self.π[state]

        state_action_pairs = [(state, action, 0)]

        while not done:
            state, reward, done = self.env.step(action)
            if not done:
                action = self.π[state]
                state_action_pairs.append((state, action, reward))
            else:
                state_action_pairs.append((state, None, reward))

        return state_action_pairs

    def compute_returns(self, state_action_pairs):
        G = 0
        for t, (state, action, reward) in enumerate(state_action_pairs):

            G = self.γ * G + reward
            self.returns[state[0], state[1], action] += G

    def reset(self):
        self.env.reset()
        self.π = np.zeros(self.env.n_states)
        self.returns = np.zeros((self.env.n_states, self.env.n_actions))
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))