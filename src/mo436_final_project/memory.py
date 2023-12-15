from collections import deque
from random import sample
from typing import Any


class ReplayMemory:
    def __init__(self, capacity: int, batch_size: int = 32) -> None:
        self.memory: deque = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.items = 0

    def push(self, *args: Any) -> None:
        """Save a transition"""
        self.memory.append(args)
        self.items += 1

    @property
    def batch_ready(self) -> bool:
        # return self.items >= self.batch_size
        return len(self.memory) >= self.batch_size

    def sample(self) -> list:
        self.items -= self.batch_size
        return sample(self.memory, self.batch_size)

    def __len__(self) -> int:
        return len(self.memory)
