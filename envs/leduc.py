from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import sys, os

from encoders import LEDUC_RANKS, LEDUC_DECK, encode_infoset

FOLD, CALL_CHECK, RAISE = 0, 1, 2

@dataclass
class LeducState:
    deck: List[int]
    private: List[int]         
    public: Optional[int]     
    to_act: int
    to_call: int               
    round_index: int           
    raises_this_round: int
    max_raises: int
    bet_size: int             
    pot: int
    last_action: Optional[int] 
    history: List[int]
    stacks: List[int]
    terminal: bool
    winner: Optional[int]

class LeducEnv:
    """Classic 2-player Leduc Hold'em with raise cap (2 per round) and bet size doubling on round 2.
    Cards: 3 ranks x 2 suits => represent hand/public by rank only.
    """
    def __init__(self, max_raises: int = 2, starting_stack: int = 99, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.max_raises = max_raises
        self.starting_stack = starting_stack
        self.reset()

    def reset(self) -> LeducState:
        deck = [r for r in range(LEDUC_RANKS) for _ in range(2)]  
        self.rng.shuffle(deck)
        private = [deck.pop(), deck.pop()]
        state = LeducState(
            deck=deck,
            private=private,
            public=None,
            to_act=0,
            to_call=0,
            round_index=0,
            raises_this_round=0,
            max_raises=self.max_raises,
            bet_size=1,
            pot=2,  
            last_action=None,
            history=[],
            stacks=[self.starting_stack, self.starting_stack],
            terminal=False,
            winner=None,
        )
        self.state = state
        return state

    def legal_actions(self) -> List[int]:
        s = self.state
        acts = [FOLD, CALL_CHECK]
        if s.raises_this_round < s.max_raises:
            acts.append(RAISE)
        return acts

    def infoset(self, player: int) -> np.ndarray:
        s = self.state
        return encode_infoset(
            private_rank=s.private[player],
            public_rank=s.public,
            to_act=s.to_act,
            round_index=s.round_index,
            raises_this_round=s.raises_this_round,
            max_raises=s.max_raises,
            pot=s.pot,
        )

    def step(self, action: int):
        assert not self.state.terminal
        assert action in self.legal_actions()
        s = self.state
        p, opp = s.to_act, 1 - s.to_act

        if action == FOLD:
            s.terminal = True
            s.winner = opp
            return self._finalize(), self._rewards(), True, {"terminal": "fold"}

        if action == RAISE:
            s.raises_this_round += 1
            s.to_call = s.bet_size
            s.pot += s.bet_size
            s.history.append(RAISE)
            s.to_act = opp
            return s, (0, 0), False, {}

        if action == CALL_CHECK:
            s.history.append(CALL_CHECK)
            if s.to_call > 0:
                s.pot += s.to_call
                s.to_call = 0

            if len(s.history) >= 2:
                last2 = s.history[-2:]
                if last2 == [CALL_CHECK, CALL_CHECK] or last2 == [RAISE, CALL_CHECK]:
                    if s.round_index == 0:
                        s.public = s.deck.pop()
                        s.round_index = 1
                        s.bet_size = 2
                        s.raises_this_round = 0
                        s.to_call = 0
                        s.history.clear()
                        s.to_act = 0
                        return s, (0, 0), False, {"round": 1}
                    else:
                        s.terminal = True
                        s.winner = self._showdown_winner()
                        return self._finalize(), self._rewards(), True, {"terminal": "showdown"}

            s.to_act = opp
            return s, (0, 0), False, {}

    def _showdown_winner(self) -> Optional[int]:
        s = self.state
        p_pub = s.public
        ranks = s.private
        score = [0, 0]
        for i in (0, 1):
            if ranks[i] == p_pub:
                score[i] = 2 * (ranks[i] + 1)  
            else:
                score[i] = 1 * (ranks[i] + 1)  
        if score[0] > score[1]:
            return 0
        if score[1] > score[0]:
            return 1
        return None  

    def _finalize(self) -> LeducState:
        return self.state

    def _rewards(self) -> Tuple[int, int]:
        s = self.state
        if s.winner is None:
            r = s.pot // 2
            return (r, r)
        return (s.pot, 0) if s.winner == 0 else (0, s.pot)


if __name__ == "__main__":
    env = LeducEnv(seed=7)
    done = False
    while not done:
        obs = env.infoset(env.state.to_act)
        a = random.choice(env.legal_actions())
        _, rew, done, info = env.step(a)
    print("terminal:", env.state.winner, "pot=", env.state.pot, "rew=", rew)
