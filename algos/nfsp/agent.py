from __future__ import annotations
import random
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os
print(os.getcwd())

# from buffers.replay import ReplayBuffer
# from buffers.reservoir import Reservoir
# from algos.nfsp.QNetwork import QNetwork
# from algos.nfsp.policy_network import PolicyNetwork

# FOLD, CALL_CHECK, RAISE = 0, 1, 2

# class NFSPAgent:
#     """Minimal NFSP agent for Leduc.

#     - Two memories: MRL (replay) for off-policy Q-learning; MSL (reservoir) for supervised average-policy learning.
#     - Two networks: Q-network (best response) and Policy-network (average strategy).
#     - Action selection: anticipatory dynamics with mixing parameter eta.
#     """
#     def __init__(
#         self,
#         obs_dim: int,
#         n_actions: int = 3,
#         device: str = "cpu",
#         mrl_capacity: int = 50_000,
#         msl_capacity: int = 200_000,
#         gamma: float = 1.0,
#         lr_q: float = 1e-3,
#         lr_pi: float = 1e-3,
#         epsilon_start: float = 0.12,
#         epsilon_final: float = 0.01,
#         epsilon_decay: int = 50_000,
#         eta: float = 0.1,
#         target_update: int = 1000,
#         seed: Optional[int] = None,
#     ):
#         self.device = torch.device(device)
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.eta = eta
#         self.epsilon_start = epsilon_start
#         self.epsilon_final = epsilon_final
#         self.epsilon_decay = epsilon_decay
#         self.step_count = 0
#         self.rng = random.Random(seed)

#         self.mrl = ReplayBuffer(mrl_capacity)
#         self.msl = Reservoir(msl_capacity)

#         self.q = QNetwork(obs_dim, n_actions).to(self.device)
#         self.q_tgt = QNetwork(obs_dim, n_actions).to(self.device)
#         self.q_tgt.load_state_dict(self.q.state_dict())
#         self.pi = PolicyNetwork(obs_dim, n_actions).to(self.device)

#         self.opt_q = optim.Adam(self.q.parameters(), lr=lr_q)
#         self.opt_pi = optim.Adam(self.pi.parameters(), lr=lr_pi)
#         self.target_update = target_update

#     def epsilon(self) -> float:
#         frac = min(1.0, self.step_count / float(self.epsilon_decay))
#         return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

#     def act(self, obs: np.ndarray, as_player: int, use_best_response: Optional[bool] = None) -> Tuple[int, bool]:
#         """Return (action, used_br_branch?).
#         If use_best_response is None, pick BR vs AVG with anticipatory mixing (eta).
#         """
#         if use_best_response is None:
#             use_best_response = (self.rng.random() < self.eta)

#         obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
#         if use_best_response:
#             if self.rng.random() < self.epsilon():
#                 a = self.rng.randrange(self.n_actions)
#             else:
#                 with torch.no_grad():
#                     q = self.q(obs_t)[0]
#                     a = int(torch.argmax(q).item())
#             return a, True
#         else:
#             with torch.no_grad():
#                 logits = self.pi(obs_t)[0]
#                 probs = torch.softmax(logits, dim=-1).cpu().numpy()
#             a = int(np.random.choice(self.n_actions, p=probs/np.sum(probs)))
#             return a, False

#     def store_transition(self, s, a, r, s_next, done):
#         self.mrl.add((s, a, r, s_next, done))

#     def store_supervised(self, s, a):
#         self.msl.add((s.astype(np.float32), int(a)))

#     def train_q(self, batch_size: int = 256):
#         if len(self.mrl) < batch_size:
#             return 0.0
#         import random as _r
#         batch = self.mrl.sample(batch_size)
#         s, a, r, s2, d = zip(*batch)
#         s = torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device)
#         a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
#         r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
#         s2 = torch.as_tensor(np.stack(s2), dtype=torch.float32, device=self.device)
#         d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

#         with torch.no_grad():
#             next_q = self.q(s2)
#             next_a = torch.argmax(next_q, dim=1)
#             next_q_tgt = self.q_tgt(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
#             y = r + (1.0 - d) * self.gamma * next_q_tgt

#         q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
#         loss = F.smooth_l1_loss(q_sa, y)

#         self.opt_q.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
#         self.opt_q.step()

#         if self.step_count % self.target_update == 0:
#             self.q_tgt.load_state_dict(self.q.state_dict())
#         return float(loss.item())

#     def train_pi(self, batch_size: int = 256):
#         if len(self.msl) < batch_size:
#             return 0.0
#         batch = self.msl.sample(batch_size)
#         s, a = zip(*batch)
#         s = torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device)
#         a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
#         logits = self.pi(s)
#         loss = F.cross_entropy(logits, a)
#         self.opt_pi.zero_grad(set_to_none=True)
#         loss.backward()
#         self.opt_pi.step()
#         return float(loss.item())