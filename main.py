import numpy as np
import torchvision as tv

np.random.seed(0)

class MNISTEnvironment:
    def __init__(self, σ_stochasticity=0.1):
        self.dataset = tv.datasets.MNIST("./data/", download=True)
        self.state = np.zeros((28, 28))
        self.image = np.zeros((28, 28))
        self.reward = np.zeros((28, 28))
        self.current_position = (0, 0)
        self.reset()
        self.σ = σ_stochasticity

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
        self.current_position = top_25[0][i], top_25[1][i]

    def get_position(self, x, y):
        μ = np.array([x, y])
        sample = np.random.multivariate_normal(μ, np.eye(2) * self.σ, 1)[0]
        sample = np.clip(sample, 0, 27)
        return sample.astype(int)

    def eval(self):
        # return (1 - mse) * 100
        return 100 * (1 - np.mean((self.state - self.image) ** 2))

    def step(self, action):
        if action == 0:
            self.state = (self.state[0] - 1, self.state[1])
        elif action == 1:
            self.state = (self.state[0] + 1, self.state[1])
        elif action == 2:
            self.state = (self.state[0], self.state[1] - 1)
        elif action == 3:
            self.state = (self.state[0], self.state[1] + 1)
        else:
            raise ValueError("Invalid action")
        self.state = (max(0, min(self.size - 1, self.state[0])),
                      max(0, min(self.size - 1, self.state[1])))
        done = self.state == self.goal
        reward = 1 if done else 0
        return self.state, reward, done