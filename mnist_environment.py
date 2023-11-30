import numpy as np
import torchvision as tv

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
        reward += self.reward[tuple(self.current_position)]

        return state, reward, done