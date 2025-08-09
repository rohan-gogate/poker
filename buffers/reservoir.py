from __future__ import annotations
import random
from typing import Any, List

class Reservoir:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.cap = capacity
        self.n_seen = 0
        self.data: List[Any] = []

    def __len__(self) -> int:
        return len(self.data)

    def add(self, item: Any) -> None:
        self.n_seen += 1
        if len(self.data) < self.cap:
            self.data.append(item)
            return
        j = random.randint(1, self.n_seen)
        if j <= self.cap:
            self.data[j - 1] = item

    def sample(self, k: int) -> List[Any]:
        k = min(k, len(self.data))
        return random.sample(self.data, k)