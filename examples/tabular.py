"""Worked example: "I have tabular data, I don't know which model to fit".

The user story (SPEC.md 21.5): given a classification table, find a model that
reaches a final accuracy > 95% on held-out data — without knowing up front
whether linear, MLP, or trees are the right family. AutoRefine's answer:

  1. wrap the data in the minimal Task protocol (`make_dataset` + `score`),
  2. register it (in-process here; via an entry point from another package),
  3. let the improver loop search family + hyperparameters against a holdout
     split the improver never trains on,
  4. gate on YOUR target from `final_best_score` — the loop maximizes the
     validated score, it does not know about the 95% bar.

The task below is synthetic (XOR of quadrants: linear models sit at ~50,
trees are weak, a tuned MLP reaches ~99 — the same family gradient that makes
the search informative). To use your real data, replace `_points()` with a
deterministic load/split of your CSV and keep the protocol.

Deterministic given the seed (G2). Run:
  python examples/tabular.py [--seed 7] [--experiments 20] [--target 95.0]
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np


class TabularV1:
    """2-D tabular classification: label = XOR of the two quadrants.

    Non-linear by construction (linear models provably cannot separate the
    four quadrant regions), clean labels (ceiling 100), unit-scaled inputs."""

    name = "tabular-v1"
    state_dim = 2
    head = "softmax"
    n_outputs = 2
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    default_dataset_size = 1024  # SPEC.md 20.3 (points)

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def _split_rng(self, split: str) -> np.random.Generator:
        seq = np.random.SeedSequence([self.seed, zlib.crc32(split.encode("utf-8"))])
        return np.random.default_rng(seq)

    def _points(self, split: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(x (n,2) in [-1,1]^2, y (n,)) — quadrant XOR, balanced by symmetry.

        For real tabular data this is where your (deterministic) CSV load and
        train/holdout split would go; `make_dataset` and `score` stay the same.
        """
        rng = self._split_rng(split)
        x = rng.uniform(-1.0, 1.0, (n, 2))
        y = ((x[:, 0] > 0.0) == (x[:, 1] > 0.0)).astype(np.int64)
        return x, y

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        return self._points(split, n)[0]

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # unit-scaled inputs need no transform

    def make_dataset(self, n_points: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """Train split: what the improver is allowed to see (SPEC.md 15)."""
        return self._points("train", n_points)

    def score(self, model, split: str, n: int) -> float:
        """100 * accuracy on fresh seed-derived points of `split`."""
        x, y = self._points(split, n)
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        return float(100.0 * (pred == y).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--experiments", type=int, default=20)
    ap.add_argument("--target", type=float, default=95.0,
                    help="your acceptance bar on final_best_score (score points)")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    # --- 1. register the task -------------------------------------------------
    # In-process (this example): one line.
    from autorefine.tasks import TASKS
    TASKS["tabular-v1"] = TabularV1
    # From another package instead, no import needed (SPEC.md 21.3):
    #   [project.entry-points."autorefine.tasks"]
    #   tabular-v1 = "my_pkg.tabular:TabularV1"
    #   (check: `python -m autorefine plugins list`)

    # --- 2. run the autonomous improvement loop --------------------------------
    from autorefine import AutoRefineEnv, BanditPolicy, Budget
    env = AutoRefineEnv(
        task="tabular-v1", seed=args.seed,
        budget=Budget(args.experiments, 900, 30), runs_dir=args.runs_dir,
    )
    state = env.reset()
    print(f"task     : tabular-v1 (seed {args.seed}, {args.experiments} experiments)")
    print(f"baseline : {state['baseline_score']:.2f}  (default spec, no tuning)")

    policy = BanditPolicy(seed=args.seed)
    while not env.done:
        state, reward, done, info = env.step(policy.propose(state))
        mark = "+" if info.get("accepted") else " "
        score = info.get("candidate_score")
        score_s = f"{score:8.2f}" if isinstance(score, (int, float)) else "  duplicate"
        print(f" {mark} exp {env.bm.used_experiments:3d}  score {score_s}  reward {reward:+.4f}")

    # --- 3. gate on the user's target -------------------------------------------
    final = env.best_score
    print(f"\nfinal best: {final:.2f}  (target {args.target:.1f})")
    if final >= args.target:
        print(f"PASS: final accuracy {final:.2f} >= {args.target:.1f} on held-out data")
        print("the loop never trained on these points; gen_gap in summary.json is the "
              "overfit guard")
    else:
        print(f"MISS: {final:.2f} < {args.target:.1f} — the search space or budget is "
              "exhausted for this task")
        print("next: raise --experiments, or extend the spec space / model families "
              "(README 'Extending')")
    print(f"\nrun dir  : {env.run_dir}")
    print("follow-up: python -m autorefine report --run <run_dir> --plot")
    print("           python -m autorefine report --run <run_dir> --json  | jq .final_best_score")
    print("           python -m autorefine eval   --run <run_dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
