from collections import deque
from typing import Deque, Tuple

class ReplayBuffer:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.buf: Deque[Tuple] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, transition: Tuple) -> None:
        self.buf.append(transition)

    def sample(self, k: int):
        import random
        k = min(k, len(self.buf))
        return random.sample(self.buf, k)
