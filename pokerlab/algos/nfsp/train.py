from __future__ import annotations
import sys
import os
import numpy as np
import torch

# Ensure the project root is on sys.path so 'pokerlab' can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from pokerlab.envs.leduc import LeducEnv
    from pokerlab.algos.nfsp.agent import NFSPAgent
except ImportError as e:
    raise ImportError(
        f"Failed to import NFSPAgent or LeducEnv. Ensure that '__init__.py' files exist in all package folders and run from the parent of 'pokerlab'. Original error: {e}"
    )


def play_episode(env: LeducEnv, a0: NFSPAgent, a1: NFSPAgent, train: bool = True):
    env.reset()
    done = False
    while not done:
        p = env.state.to_act
        agent = a0 if p == 0 else a1
        obs = env.infoset(p)
        legal = env.legal_actions()
        act, used_br = agent.act(obs, legal_actions=legal, as_player=p)

        # Step env
        _, rew, done, _ = env.step(act)

        if train:
            r = 0.0
            if done:
                r = float(rew[p])
            next_obs = env.infoset(p) if not done else np.zeros_like(obs)
            agent.store_transition(obs, act, r, next_obs, float(done))
            if used_br:
                agent.store_supervised(obs, act)

        a0.step_count += 1
        a1.step_count += 1
    return rew


def train_nfsp(episodes: int = 10_000, device: str = "cpu"):
    env = LeducEnv(seed=123)
    obs_dim = env.infoset(0).shape[0]
    a0 = NFSPAgent(obs_dim, device=device, seed=1)
    a1 = NFSPAgent(obs_dim, device=device, seed=2)

    for ep in range(1, episodes + 1):
        rew = play_episode(env, a0, a1, train=True)
        q_loss0 = a0.train_q(256)
        q_loss1 = a1.train_q(256)
        pi_loss0 = a0.train_pi(256)
        pi_loss1 = a1.train_pi(256)

        if ep % 200 == 0:
            print(
                f"ep {ep:5d} | q_loss: {q_loss0:.3f}/{q_loss1:.3f} | "
                f"pi_loss: {pi_loss0:.3f}/{pi_loss1:.3f} | last_rew: {rew}"
            )
    return a0, a1


if __name__ == "__main__":
    train_nfsp(episodes=2000, device="cpu")