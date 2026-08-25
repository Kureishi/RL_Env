"""DQN-style agent driving AutoRefineGymEnv through the gymnasium API only.

SPEC.md 21.1: proves the §15 adapter contract end-to-end with an
external-library-style agent — NumPy only, no other dependencies, deterministic
given a seed (G2). The agent touches the environment *only* through the
gymnasium surface (`reset(seed)`, `step(action)`, `observation_space`,
`action_space`) and imports nothing from autorefine beyond the adapter
(`Budget` for the CLI args). No target network is used — a deliberate
simplification for a 2-layer net on a small, non-stationary MDP.

Run:  python examples/gym_dqn.py [--task sine-v1] [--experiments 5] [--seed 7]
"""
from __future__ import annotations

import argparse

import numpy as np


class DQNAgent:
    """2-layer Q-network + experience replay + epsilon-greedy (NumPy only)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 7,
        hidden: int = 64,
        lr: float = 2e-3,
        buffer_size: int = 512,
        batch_size: int = 4,  # small: episodes are short (one step per experiment)
        eps_start: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.8,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        s1 = 1.0 / np.sqrt(obs_dim)
        s2 = 1.0 / np.sqrt(hidden)
        self.w1 = self.rng.normal(0.0, s1, (obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = self.rng.normal(0.0, s2, (hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        self.buffer: list[tuple] = []
        self.buffer_size = int(buffer_size)
        self.batch = int(batch_size)
        self.lr = lr
        self.eps_start, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.t = 0
        self.updates = 0

    def epsilon(self) -> float:
        return max(self.eps_min, self.eps_start * (self.eps_decay ** self.t))

    def q(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(obs @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def act(self, obs: np.ndarray) -> int:
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q(obs)))

    def store(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))
        if len(self.buffer) > self.buffer_size:
            del self.buffer[: len(self.buffer) - self.buffer_size]

    def update(self) -> bool:
        """One replay-batch MSE update (hand-written backprop). True if run."""
        if len(self.buffer) < self.batch:
            return False
        idx = self.rng.integers(0, len(self.buffer), size=self.batch)
        n = self.batch
        x = np.stack([self.buffer[i][0] for i in idx])
        a = np.asarray([self.buffer[i][1] for i in idx], dtype=np.intp)
        r = np.asarray([self.buffer[i][2] for i in idx], dtype=np.float64)
        done = np.asarray([self.buffer[i][4] for i in idx], dtype=np.float64)
        nq = np.stack([self.buffer[i][3] for i in idx])
        target = r + (1.0 - done) * np.max(self.q(nq), axis=1)
        h = np.tanh(x @ self.w1 + self.b1)
        qv = h @ self.w2 + self.b2
        diff = qv[np.arange(n), a] - target
        dq = np.zeros_like(qv)
        dq[np.arange(n), a] = diff / n
        dw2 = h.T @ dq
        db2 = dq.sum(axis=0)
        dh = (dq @ self.w2.T) * (1.0 - h * h)
        dw1 = x.T @ dh
        db1 = dh.sum(axis=0)
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.updates += 1
        return True

    def step_learning(self, obs, action, reward, next_obs, done) -> None:
        """Record a transition and (if the buffer is full enough) update."""
        self.t += 1
        self.store(obs, action, reward, next_obs, done)
        self.update()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="sine-v1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--experiments", type=int, default=5)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    from autorefine import Budget
    from autorefine.gym import AutoRefineGymEnv  # the only autorefine imports

    env = AutoRefineGymEnv(
        task=args.task, seed=args.seed,
        budget=Budget(args.experiments, 600, 30), runs_dir=args.runs_dir,
    )
    obs, _ = env.reset(seed=args.seed)
    agent = DQNAgent(obs.shape[0], env.action_space.n, seed=args.seed)
    total = 0.0
    steps = 0
    while True:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        total += reward
        steps += 1
        agent.step_learning(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated:
            break
    print(f"episode  : {steps} steps, total reward {total:+.4f}")
    print(f"agent    : {agent.updates} replay updates, buffer {len(agent.buffer)}")
    print(f"runs dir : {args.runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
