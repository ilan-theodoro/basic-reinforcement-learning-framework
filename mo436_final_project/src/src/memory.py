from collections import deque
from random import sample


class ReplayMemory(object):
    def __init__(self, capacity, batch_size=32):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.items = 0

    def push(self, *args):
        """Save a transition"""
        self.memory.append(args)
        self.items += 1

    @property
    def batch_ready(self):
        #return self.items >= self.batch_size
        return len(self.memory) >= self.batch_size

    def sample(self):
        self.items -= self.batch_size
        return sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)