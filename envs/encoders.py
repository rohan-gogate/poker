from typing import Dict, List
import numpy as np

LEDUC_RANKS = 3
LEDUC_DECK = LEDUC_RANKS * 2


def one_hot(n: int, idx: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def encode_infoset(
    private_rank: int,
    public_rank: int | None,
    to_act: int,
    round_index: int,
    raises_this_round: int,
    max_raises: int,
    pot: int,
) -> np.ndarray:
    """Domain-independent-ish vector:
    - private card rank (one-hot of size 3)
    - public card rank incl. 'none' (one-hot size 4)
    - player to act (one-hot size 2)
    - round (preflop/flop: one-hot size 2)
    - raises so far (one-hot up to max_raises+1)
    - pot size bucketed coarse (4 buckets)
    """
    priv = one_hot(LEDUC_RANKS, private_rank)
    pub = one_hot(LEDUC_RANKS + 1, public_rank if public_rank is not None else LEDUC_RANKS)
    actor = one_hot(2, to_act)
    rnd = one_hot(2, round_index)
    raises = one_hot(max_raises + 1, min(raises_this_round, max_raises))
    
    bucket = 0
    if pot >= 5:
        bucket = 3
    elif pot >= 3:
        bucket = 2
    elif pot >= 1:
        bucket = 1
    pot_enc = one_hot(4, bucket)
    return np.concatenate([priv, pub, actor, rnd, raises, pot_enc]).astype(np.float32)