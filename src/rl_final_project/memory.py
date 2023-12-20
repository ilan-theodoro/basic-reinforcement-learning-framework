"""Replay Memory implementation as described in DQN paper.

author: Ilan Theodoro.
date: December/2023.
"""
from collections import deque
from random import sample
from typing import Any
from typing import Optional


class ReplayMemory:
    """Replay memory as described in the DQN paper."""

    def __init__(self, capacity: int, batch_size: Optional[int]) -> None:
        """Initialize the replay memory.

        :param capacity: The maximum number of transitions to be saved.
        :param batch_size: The batch size to be sampled.
        :param consume_and_release: Whether to consume and release the memory.
        """
        self.memory: deque = deque([], maxlen=capacity)
        if batch_size is None:
            self.consume_and_release = True
            self.batch_size = 1
        else:
            self.consume_and_release = False
            self.batch_size = batch_size

    def push(self, *args: Any) -> None:
        """Save a transition.

        :param args: The transition to be saved.
        """
        self.memory.append(args)

    @property
    def batch_ready(self) -> bool:
        """Whether the batch is ready.

        :return: Whether the batch is ready.
        """
        return len(self.memory) >= self.batch_size

    def sample(self) -> list:
        """Sample a batch according to the batch size.

        :return: A batch of transitions.
        """
        if not self.batch_ready:
            raise RuntimeError(
                "The batch is not ready. There are not enough " "transitions."
            )
        if self.consume_and_release:
            ret = [self.memory.popleft() for _ in range(self.batch_size)]
            return ret
        else:
            return sample(self.memory, self.batch_size)
