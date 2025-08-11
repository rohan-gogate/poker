import numpy as np

def regret_matching(advantages: np.ndarray) -> np.ndarray:
    pos = np.maximum(advantages, 0.0)
    s = pos.sum()
    if s <= 0:
        p = np.zeros_like(pos, dtype=np.float32)
        p[int(np.argmax(advantages))] = 1.0
        return p
    return (pos / s).astype(np.float32)