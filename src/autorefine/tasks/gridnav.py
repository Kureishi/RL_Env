"""GridNavV1: tabular navigation policy task (SPEC.md 15).

A 6x6 toroidal grid: the model maps a one-hot cell state (36,) to one of
4 directions {0: up, 1: down, 2: left, 3: right} (edges wrap). An episode
ends when the agent reaches the fixed goal cell (success) or times out
(failure). Deterministic given the seed (SPEC.md G2, S3).

Score (holdout/gen): 100 * success_rate + 25 * (1 - mean_steps / max_steps)
(reward both reliability and efficiency; max 125).
"""
from __future__ import annotations

import zlib

import numpy as np

WIDTH = 6
HEIGHT = 6
N_CELLS = WIDTH * HEIGHT
GOAL = WIDTH * (HEIGHT - 1) + (WIDTH - 1)  # bottom-right corner cell
ACTIONS = (0, 1, 2, 3)  # up, down, left, right


class GridNavV1:
    name = "gridnav-v1"
    state_dim = N_CELLS
    n_outputs = 4
    head = "softmax"
    max_steps = 20  # shortest torus path is <= 6 steps; 20x margin
    default_dataset_size = 60  # SPEC.md 20.3 (episodes; v1 legacy value)

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def _split_rng(self, split: str) -> np.random.Generator:
        seq = np.random.SeedSequence([self.seed, zlib.crc32(split.encode("utf-8"))])
        return np.random.default_rng(seq)

    # --- geometry -------------------------------------------------------------
    @staticmethod
    def _decode(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cell = np.argmax(np.atleast_2d(states), axis=1)
        return cell // WIDTH, cell % WIDTH

    # --- Task protocol ----------------------------------------------------------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        """Seed-derived start cells (uniform, excluding the goal), one-hot (n, 36)."""
        rng = self._split_rng(split)
        cells = rng.integers(0, N_CELLS, n)
        cells = np.where(cells == GOAL, (GOAL + 1) % N_CELLS, cells)
        out = np.zeros((n, N_CELLS), dtype=np.float64)
        out[np.arange(n), cells] = 1.0
        return out

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # one-hot inputs need no transform

    def act(self, logits: np.ndarray) -> np.ndarray:
        """logits (n, 4) -> direction index (n,)."""
        return logits.argmax(axis=1).astype(np.int64)

    def step_vec(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, c = self._decode(states)
        a = actions.astype(np.int64)
        r2 = np.where(a == 0, (r - 1) % HEIGHT, r)
        r2 = np.where(a == 1, (r + 1) % HEIGHT, r2)
        c2 = np.where(a == 2, (c - 1) % WIDTH, c)
        c2 = np.where(a == 3, (c + 1) % WIDTH, c2)
        nxt = np.zeros((len(states), N_CELLS), dtype=np.float64)
        cells = r2 * WIDTH + c2
        nxt[np.arange(len(states)), cells] = 1.0
        terminated = cells == GOAL
        return nxt, terminated

    def terminal_success(self, states: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Success = reached the goal before the timeout."""
        return steps < self.max_steps

    def reference_action(self, states: np.ndarray) -> np.ndarray:
        """Move along the coordinate that reduces torus distance the most."""
        r, c = self._decode(states)
        gr, gc = GOAL // WIDTH, GOAL % WIDTH
        d = (gr - r) % HEIGHT
        dr = np.where(d <= HEIGHT // 2, d, d - HEIGHT)
        d = (gc - c) % WIDTH
        dc = np.where(d <= WIDTH // 2, d, d - WIDTH)
        a = np.zeros(len(r), dtype=np.int64)
        vertical = (np.abs(dr) >= np.abs(dc)) & (dr != 0)
        a[vertical & (dr < 0)] = 0   # up
        a[vertical & (dr > 0)] = 1   # down
        horizontal = ~vertical & (dc != 0)
        a[horizontal & (dc < 0)] = 2  # left
        a[horizontal & (dc > 0)] = 3  # right
        return a

    def make_dataset(self, n_episodes: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """(states, action_labels): reference rollouts from the train split plus
        every non-goal cell with its confident reference action — a dense,
        unambiguous tabular policy target."""
        X: list[np.ndarray] = []
        A: list[np.ndarray] = []

        ics = self.initial_conditions("train", n_episodes)
        for ic in ics:
            st = ic[None, :]
            for _ in range(self.max_steps):
                act = int(self.reference_action(st)[0])
                X.append(st)
                A.append(np.array([act]))
                nxt, term = self.step_vec(st, np.array([act]))
                st = nxt[0]
                if term[0]:
                    break

        all_cells = np.eye(N_CELLS, dtype=np.float64)
        mask = np.arange(N_CELLS) != GOAL
        X.append(all_cells[mask])
        A.append(self.reference_action(all_cells[mask]))

        X_all = np.vstack(X)
        A_all = np.concatenate(A).astype(np.int64)
        return X_all, A_all

    # --- scoring ----------------------------------------------------------------
    def score(self, model, split: str, n: int) -> float:
        """100 * success_rate + 25 * (1 - mean_steps / max_steps); higher is better."""
        from ..evaluator import run_episodes

        stats = run_episodes(self, model, split, n)
        return (
            100.0 * stats["success_rate"]
            + 25.0 * (1.0 - stats["mean_steps"] / self.max_steps)
        )
