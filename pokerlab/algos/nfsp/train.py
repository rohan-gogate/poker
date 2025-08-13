from __future__ import annotations
import sys, os
import numpy as np
from typing import Optional, Tuple
# --- path shim ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pokerlab.envs.leduc import LeducEnv
from pokerlab.algos.nfsp.agent import NFSPAgent

def _zero_like_obs(obs):
    try:
        return np.zeros_like(obs)
    except Exception:
        return None

# --- Helper: print buffers if the agent exposes sizes/capacity ---
def debug_print_replay_sizes(agent, tag: str = ""):
    def _fmt(x):
        return "?" if x is None else str(x)
    mrl_n = getattr(agent, "mrl_size", None)
    mrl_cap = getattr(agent, "mrl_capacity", None)
    msl_n = getattr(agent, "msl_size", None)
    msl_cap = getattr(agent, "msl_capacity", None)
    print(f"[buffers{(' ' + tag) if tag else ''}] MRL { _fmt(mrl_n) }/{ _fmt(mrl_cap) } | MSL { _fmt(msl_n) }/{ _fmt(msl_cap) }")




def play_episode(env, a0, a1, train: bool = True, force_policy: bool = False) -> Tuple[int, Tuple[float,float]]:
    env.reset()
    done = False
    last = {0: {"obs": None, "act": None}, 1: {"obs": None, "act": None}}

    while not done:
        p = env.state.to_act
        agent = a0 if p == 0 else a1
        obs = env.infoset(p)
        legal = env.legal_actions()
        use_br: Optional[bool] = None if not force_policy else False
        act, used_br = agent.act(obs, legal_actions=legal, as_player=p, use_best_response=use_br)

        if train:
            last[p]["obs"], last[p]["act"] = obs, act

        _, rew, done, _ = env.step(act)

        if train:
            if not done:
                nxt = env.infoset(p)
                agent.store_transition(obs, act, 0.0, nxt, 0.0)
                if used_br:
                    agent.store_supervised(obs, act)
            else:
                signed = float(rew[0]) - float(rew[1])
                r_p = signed if p == 0 else -signed
                r_opp = -r_p
                zero = _zero_like_obs(obs)

                agent.store_transition(obs, act, r_p, zero, 1.0)
                if used_br:
                    agent.store_supervised(obs, act)

                opp = 1 - p
                opp_agent = a0 if opp == 0 else a1
                if last[opp]["obs"] is not None and last[opp]["act"] is not None:
                    opp_agent.store_transition(last[opp]["obs"], last[opp]["act"], r_opp, zero, 1.0)

    signed = float(rew[0]) - float(rew[1])
    return 1, (signed, -signed)


def evaluate_avg_policy(env, agent0, agent1, hands: int = 2000, seed: Optional[int] = None) -> Tuple[float,float]:
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    total0 = 0.0
    total1 = 0.0
    for _ in range(hands):
        env.reset()
        done = False
        while not done:
            p = env.state.to_act
            agent = agent0 if p == 0 else agent1
            obs = env.infoset(p)
            legal = env.legal_actions()
            act, _ = agent.act(obs, legal_actions=legal, as_player=p, use_best_response=False)
            _, rew, done, _ = env.step(act)
        signed = float(rew[0]) - float(rew[1])
        total0 += signed
        total1 += -signed
    return total0 / hands, total1 


def evaluate_proxy_br(env, avg_agent, br_agent, avg_player: int, hands: int = 2000, seed: Optional[int] = None) -> float:
    assert avg_player in (0, 1)
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    total = 0.0
    for _ in range(hands):
        env.reset()
        done = False
        while not done:
            p = env.state.to_act
            if p == avg_player:
                obs = env.infoset(p)
                legal = env.legal_actions()
                act, _ = avg_agent.act(obs, legal_actions=legal, as_player=p, use_best_response=False)
            else:
                obs = env.infoset(p)
                legal = env.legal_actions()
                act, _ = br_agent.act(obs, legal_actions=legal, as_player=p, use_best_response=True)
            _, rew, done, _ = env.step(act)
        signed = float(rew[0]) - float(rew[1])
        total += signed if (1 - avg_player) == 0 else -signed
    return total / hands


# keep higher anticipatory longer so the SL buffer fills earlier
WARMUP_DECAY_EP = 10_000


def maybe_configure_agent(agent: NFSPAgent):
    """Best-effort toggles on agent if methods exist: larger MSL, Huber loss, target net, grad clip."""
    # 1) Increase supervised memory (MSL)
    for name in ("resize_msl", "set_msl_capacity", "set_sl_capacity"):
        if hasattr(agent, name):
            try:
                getattr(agent, name)(50_000)
                break
            except Exception:
                pass
    # 2) Switch Q to Huber loss if supported
    if hasattr(agent, "set_q_loss"):
        try:
            agent.set_q_loss("huber")
        except Exception:
            pass
    # 3) Enable target network soft updates if supported
    if hasattr(agent, "enable_target"):
        try:
            agent.enable_target(tau=0.01)
        except Exception:
            pass
    # 4) Gradient clipping if supported
    if hasattr(agent, "set_grad_clip"):
        try:
            agent.set_grad_clip(1.0)
        except Exception:
            pass


def train_nfsp(episodes: int = 20_000, device: str = "cpu"):
    env = LeducEnv(seed=123)
    obs_dim = env.infoset(0).shape[0]

    a0 = NFSPAgent(obs_dim, device=device, seed=1)
    a1 = NFSPAgent(obs_dim, device=device, seed=2)

    # Optional: configure agent internals if supported by your implementation
    maybe_configure_agent(a0)
    maybe_configure_agent(a1)

    # Warmup anticipatory parameter: keep BR sampling higher until 10k
    a0.eta = 0.30
    a1.eta = 0.30

    for ep in range(1, episodes + 1):
        rew = play_episode(env, a0, a1, train=True)

        # a few small updates per episode for stability
        q_loss0 = q_loss1 = pi_loss0 = pi_loss1 = 0.0
        for _ in range(4):  # 4 mini-passes
            q_loss0 = a0.train_q(128)
            q_loss1 = a1.train_q(128)
            pi_loss0 = a0.train_pi(32)   # smaller batch so pi starts moving earlier
            pi_loss1 = a1.train_pi(32)

        # decay anticipatory parameter after longer warmup
        if ep == WARMUP_DECAY_EP:
            a0.eta = 0.10
            a1.eta = 0.10

        if ep % 200 == 0:
            print(
                f"ep {ep:5d} | mrl {len(a0.mrl)}/{len(a1.mrl)} msl {len(a0.msl)}/{len(a1.msl)} | "
                f"q_loss: {q_loss0:.3f}/{q_loss1:.3f} | pi_loss: {pi_loss0:.3f}/{pi_loss1:.3f} | last_rew: {rew}"
            )

        # periodic policy-only eval (does not touch buffers)
        if ep % 500 == 0:
            avg0, avg1 = evaluate_avg_policy(env, a0, a1, hands=2000)
            print(f"  [eval avg-policy over 2k hands] EV p0: {avg0:.3f} | p1: {avg1:.3f}")

        # proxy BR eval: BR vs average policy for each side
        if ep % 1000 == 0:
            br_vs_p0 = evaluate_proxy_br(env, avg_agent=a0, br_agent=a1, avg_player=0, hands=4000)
            br_vs_p1 = evaluate_proxy_br(env, avg_agent=a1, br_agent=a0, avg_player=1, hands=4000)
            print(f"  [proxy BR] EV exploit p0: {br_vs_p0:.3f} | exploit p1: {br_vs_p1:.3f}")

    return a0, a1


if __name__ == "__main__":
    train_nfsp(episodes=20_000, device="cpu")
