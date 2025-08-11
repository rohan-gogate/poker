from __future__ import annotations
import os
import sys
import random
from typing import Optional, Tuple, Iterable

# --- path shim so this works even when run directly ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

# Absolute imports (now resolvable thanks to the path shim)
from pokerlab.buffers.replay import ReplayBuffer
from pokerlab.buffers.reservoir import Reservoir
from pokerlab.algos.nfsp.QNetwork import QNetwork
from pokerlab.algos.nfsp.policy_network import PolicyNetwork

FOLD, CALL_CHECK, RAISE = 0, 1, 2


class NFSPAgent:
    """Minimal NFSP agent (Q-best-response + supervised avg policy)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        device: str = "cpu",
        mrl_capacity: int = 50_000,
        msl_capacity: int = 200_000,
        gamma: float = 1.0,
        lr_q: float = 1e-3,
        lr_pi: float = 1e-3,
        epsilon_start: float = 0.12,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 50_000,
        eta: float = 0.1,
        target_update: int = 1000,
        seed: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.eta = eta
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.rng = random.Random(seed)

        # Memories
        self.mrl = ReplayBuffer(mrl_capacity)
        self.msl = Reservoir(msl_capacity)

        # Networks
        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_tgt = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.pi = PolicyNetwork(obs_dim, n_actions).to(self.device)

        # Optimizers
        self.opt_q = optim.Adam(self.q.parameters(), lr=lr_q)
        self.opt_pi = optim.Adam(self.pi.parameters(), lr=lr_pi)
        self.target_update = target_update

    # -------- utilities --------
    def _choose_random_legal(self, legal: Iterable[int]) -> int:
        legal = list(legal)
        return legal[self.rng.randrange(len(legal))]

    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / float(self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def _mask_illegal_q(self, q: torch.Tensor, legal: Iterable[int]) -> torch.Tensor:
        mask = torch.full_like(q, -1e9)
        idx = torch.tensor(list(legal), dtype=torch.long, device=q.device)
        mask[idx] = 0.0
        return q + mask

    def _mask_illegal_probs(self, logits: torch.Tensor, legal: Iterable[int]) -> np.ndarray:
        probs = torch.softmax(logits, dim=-1)
        m = torch.zeros_like(probs)
        idx = torch.tensor(list(legal), dtype=torch.long, device=probs.device)
        m[idx] = 1.0
        probs = (probs * m)
        # if all zero (shouldn't happen), fall back to uniform over legal
        if float(probs.sum()) <= 0.0:
            probs = m / m.sum()
        else:
            probs = probs / probs.sum()
        return probs.cpu().numpy()

    # -------- acting --------
    def act(
        self,
        obs: np.ndarray,
        legal_actions: Iterable[int],
        as_player: int,
        use_best_response: Optional[bool] = None,
    ) -> Tuple[int, bool]:
        if use_best_response is None:
            use_best_response = self.rng.random() < self.eta
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if use_best_response:
            # ε-greedy over legal actions
            if self.rng.random() < self.epsilon():
                a = self._choose_random_legal(legal_actions)
            else:
                with torch.no_grad():
                    q = self.q(obs_t)[0]
                    q_masked = self._mask_illegal_q(q, legal_actions)
                    a = int(torch.argmax(q_masked).item())
            return a, True
        else:
            with torch.no_grad():
                logits = self.pi(obs_t)[0]
                probs = self._mask_illegal_probs(logits, legal_actions)
            a = int(np.random.choice(self.n_actions, p=probs))
            return a, False

    # -------- storage --------
    def store_transition(self, s, a, r, s_next, done):
        self.mrl.add((s, a, r, s_next, done))

    def store_supervised(self, s, a):
        self.msl.add((np.asarray(s, dtype=np.float32), int(a)))

    # -------- training --------
    def train_q(self, batch_size: int = 256):
        if len(self.mrl) < batch_size:
            return 0.0
        batch = self.mrl.sample(batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack(s2), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.q(s2)
            next_a = torch.argmax(next_q, dim=1)
            next_q_tgt = self.q_tgt(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            y = r + (1.0 - d) * self.gamma * next_q_tgt

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt_q.step()

        if self.step_count % self.target_update == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def train_pi(self, batch_size: int = 256):
        if len(self.msl) < batch_size:
            return 0.0
        batch = self.msl.sample(batch_size)
        s, a = zip(*batch)
        s = torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        logits = self.pi(s)
        loss = F.cross_entropy(logits, a)
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_pi.step()
        return float(loss.item())
