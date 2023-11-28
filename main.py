import numpy as np
import torchvision as tv

np.random.seed(0)

class MNISTEnvironment:
    def __init__(self, σ_stochasticity=0.1):
        self.dataset = tv.datasets.MNIST("./data/", download=True)
        self.state = np.zeros((28, 28))
        self.image = np.zeros((28, 28))
        self.reward = np.zeros((28, 28))
        self.current_position = np.array([0, 0])
        self.reset()
        self.σ = σ_stochasticity

    @property
    def n_states(self):
        return 28 * 28

    @property
    def n_actions(self):
        return 5

    def reset(self):
        # choose one image
        self.image = self.dataset[np.random.randint(0, len(self.dataset))][0].numpy()
        self.state = np.zeros((28, 28))
        self.reward = (self.image / 255) * 2 - 1

        # get a random position in the image that is inside the top-25 brightest pixels
        # TODO make 25 a parameter
        top_25 = np.argsort(self.image.flatten())[-25:]
        top_25 = np.unravel_index(top_25, self.image.shape)
        i = np.random.randint(0, len(top_25[0]))
        self.current_position = np.array([top_25[0][i], top_25[1][i]])

    def get_position(self, x, y):
        μ = np.array([x, y])
        sample = np.random.multivariate_normal(μ, np.eye(2) * self.σ, 1)[0]
        sample = np.where(sample < 0, 27, sample)
        sample = np.where(sample > 27, 0, sample)
        return sample.astype(int)

    def eval(self):
        # return (1 - mse) * 100
        return 100 * (1 - np.mean((self.state - self.image) ** 2))

    def step(self, action):
        pos = self.current_position

        reward = 0
        done = False

        if action == 0:
            pos[0] -= 1
        elif action == 1:
            pos[0] += 1
        elif action == 2:
            pos[1] -= 1
        elif action == 3:
            pos[1] += 1
        elif action == 4:
            done = True
            reward += self.eval()
        else:
            raise ValueError(f"action {action} not recognized")
        self.current_position = self.get_position(*pos)

        state = self.current_position
        reward += self.reward[*self.current_position]

        return state, reward, done

class MonteCarloControl:
    def __init__(self, num_episodes=1000):
        self.env = MNISTEnvironment()
        self.Q = np.zeros((28, 28, 4))
        self.returns = np.zeros((28, 28, 4))
        self.π = np.zeros(self.env.n_states)
        self.γ = 0.9
        self.ε = 0.1
        self.α = 0.1
        self.num_episodes = num_episodes

    def run(self):
        self.reset()
        for i in range(self.num_episodes):
            state, action, reward = self.run_episode()

    # def update(self, state, action, reward):
    #     self.returns[state[0], state[1], action] += reward
    #     self.Q[state[0], state[1], action] = self.returns[state[0], state[1], action] / self.π[state[0], state[1], action]
    #
    def run_episode(self):
        self.env.reset()

        state_action_pairs = []

        done = False
        state = self.env.current_position
        action = self.π[state]
        while not done:
            state, reward, done = self.env.step(action)
            action = self.π[state]
            state_action_pairs.append((state, action, reward))

        return state_action_pairs

    def reset(self):
        self.env.reset()
        self.π = np.zeros((28, 28, 4))
        self.returns = np.zeros((28, 28, 4))
        self.Q = np.zeros((28, 28, 4))