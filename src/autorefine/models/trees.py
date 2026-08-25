"""Bagged decision-tree ensemble (SPEC.md 15, model families).

NumPy-only CART (entropy / variance splits), deterministic given a seed.
Classification leaves hold class-probability vectors; regression leaves hold
the target mean. `forward(x)` has the same contract as MLP.forward:
(n, n_out) — so the evaluator and tasks treat both families uniformly.
Checkpoints are plain float/int arrays (np.load with allow_pickle=False).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _min_samples_leaf(n: int) -> int:
    return int(min(max(2, n // 200), 500))


def _best_split(
    X: np.ndarray, y: np.ndarray, min_leaf: int, head: str
) -> tuple[int, float, float] | None:
    """Best (feature, threshold, weighted_impurity) minimizing impurity."""
    n, d = X.shape
    if head != "softmax":
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
    best: tuple[int, int, float] | None = None  # (feature, split position, impurity)
    for f in range(d):
        order = np.argsort(X[:, f], kind="stable")
        xs = X[order, f]
        change = np.empty(n, dtype=bool)
        change[0] = False
        change[1:] = xs[1:] != xs[:-1]
        if not change[1:].any():
            continue
        if head == "softmax":
            # classification: cumulative class counts -> weighted entropy
            y = np.asarray(y, dtype=np.int64).reshape(n)
            C = int(y.max()) + 1
            counts = np.zeros((n + 1, C), dtype=np.int64)
            np.add.at(counts, (np.arange(n) + 1, y), 1)
            counts = counts.cumsum(axis=0)
            # weighted entropy of both sides of every split position, vectorized
            p = np.arange(n + 1)[:, None]
            q = (n - np.arange(n + 1))[:, None]
            total = counts[-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                pl = counts / np.maximum(p, 1)
                pr = (total[None, :] - counts) / np.maximum(q, 1)
                hl = -(np.where(pl > 0, pl * np.log(pl), 0.0)).sum(axis=1)
                hr = -(np.where(pr > 0, pr * np.log(pr), 0.0)).sum(axis=1)
            wr = (p[:, 0] * hl + q[:, 0] * hr) / n
        else:
            # regression: cumulative sums -> weighted variance of each side
            zero = np.zeros((1,) + y.shape[1:])
            s = np.concatenate([zero, np.cumsum(y, axis=0)])          # (n+1, out)
            s2 = np.concatenate([zero, np.cumsum(y * y, axis=0)])
            p = np.arange(n + 1)[:, None]
            q = (n - np.arange(n + 1))[:, None]
            sum_r = s[-1] - s
            ssq_r = s2[-1] - s2
            vl = np.maximum(s2 - s * s / np.maximum(p, 1), 0.0) / np.maximum(p, 1)
            vr = np.maximum(ssq_r - sum_r * sum_r / np.maximum(q, 1), 0.0) / np.maximum(q, 1)
            wr = ((p * vl + q * vr) / n).ravel()
        # valid split positions p in 1..n-1: boundary between distinct values
        # and both sides satisfy the minimum leaf size
        ps = np.arange(1, n)
        valid = change[ps] & (ps >= min_leaf) & (n - ps >= min_leaf)
        if not valid.any():
            continue
        cand = np.where(valid, wr[ps], np.inf)
        j = int(np.argmin(cand))
        cj = cand[j]
        if not np.isfinite(cj):
            continue
        pos = int(ps[j])  # split position: pos elements left, n - pos right
        if best is None or cand[j] < best[2]:
            best = (int(f), pos, float(cand[j]))
    if best is None:
        return None
    f, pos, imp = best
    # re-derive the boundary between the pos-th and (pos+1)-th sorted values
    # for that feature (the argmin was taken across features)
    order = np.argsort(X[:, f], kind="stable")
    xs = X[order, f]
    thr = float(0.5 * (xs[pos - 1] + xs[pos]))
    return int(f), thr, imp


def _fit_tree(
    X: np.ndarray, y: np.ndarray, max_depth: int, min_leaf: int, n_out: int,
    head: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recursive CART; returns flat node arrays (feature, threshold, left,
    right, leaf_value (k, n_out)). Leaves have left == right == -1."""
    y = np.asarray(y)
    if head == "mse":
        y = y.reshape(len(y), n_out)
    feat: list[int] = []
    thr: list[float] = []
    left: list[int] = []
    right: list[int] = []
    leafv: list[np.ndarray] = []

    def _leaf_value(idx: np.ndarray) -> np.ndarray:
        if head == "mse":
            return y[idx].mean(axis=0)
        v = np.zeros(n_out)
        for c in range(n_out):
            v[c] = (y[idx] == c).mean()
        return v

    def grow(idx: np.ndarray, depth: int) -> None:
        i = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        leafv.append(_leaf_value(idx))
        if depth >= max_depth or idx.size < 2 * min_leaf:
            return
        split = _best_split(X[idx], y[idx], min_leaf, head)
        if split is None:
            return
        f, t, _imp = split
        mask = X[idx, f] < t
        grow(idx[mask], depth + 1)
        grow(idx[~mask], depth + 1)
        # overwrite the placeholder with an internal node (children just added)
        feat[i] = int(f)
        thr[i] = t
        left[i] = i + 1
        right[i] = i + 2

    grow(np.arange(X.shape[0]), 0)
    return (
        np.array(feat, dtype=np.int64),
        np.array(thr, dtype=np.float64),
        np.array(left, dtype=np.int64),
        np.array(right, dtype=np.int64),
        np.stack(leafv, axis=0),
    )


class _Tree:
    def __init__(self, feat, thr, left, right, leafv) -> None:
        self.feat = feat
        self.thr = thr
        self.left = left
        self.right = right
        self.leafv = leafv

    def predict_values(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, self.leafv.shape[1]))
        idx = np.zeros(n, dtype=np.int64)
        done = np.zeros(n, dtype=bool)
        while not done.all():
            active = np.flatnonzero(~done)
            node = idx[active]
            is_leaf = self.left[node] < 0
            la = active[is_leaf]
            out[la] = self.leafv[node[is_leaf]]
            done[la] = True
            ia = active[~is_leaf]
            if ia.size:
                n2 = idx[ia]
                idx[ia] = np.where(
                    X[ia, self.feat[n2]] < self.thr[n2], self.left[n2], self.right[n2]
                )
        return out


class TreeEnsemble:
    """Bagged CART ensemble. forward(x) -> (n, n_out) averaged leaf values."""

    def __init__(
        self,
        n_out: int,
        n_trees: int,
        max_depth: int,
        head: str,
        seed: int,
    ) -> None:
        self.n_out = int(n_out)
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.head = head
        self.seed = int(seed)
        self.trees: list[_Tree] = []

    def fit(self, X: np.ndarray, y: np.ndarray, input_noise: float = 0.0) -> "TreeEnsemble":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        min_leaf = _min_samples_leaf(n)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n)  # bootstrap bag
            xb = X[idx]
            if input_noise > 0.0:
                xb = xb + rng.normal(0.0, input_noise, xb.shape)
            yb = y[idx]
            feat, thr, left, right, leafv = _fit_tree(
                xb, yb, self.max_depth, min_leaf, self.n_out, self.head,
            )
            self.trees.append(_Tree(feat, thr, left, right, leafv))
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("TreeEnsemble is not fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        acc = np.zeros((x.shape[0], self.n_out))
        for t in self.trees:
            acc += t.predict_values(x)
        return acc / len(self.trees)

    # --- (de)serialization: plain arrays only (allow_pickle=False) ----------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {
            "n_trees": np.array(len(self.trees)),
            "n_out": np.array(self.n_out),
            "max_depth": np.array(self.max_depth),
            "head": np.array(self.head),
            "seed": np.array(self.seed),
        }
        for i, t in enumerate(self.trees):
            arrays[f"t{i}_feature"] = t.feat
            arrays[f"t{i}_threshold"] = t.thr
            arrays[f"t{i}_left"] = t.left
            arrays[f"t{i}_right"] = t.right
            arrays[f"t{i}_leaf_value"] = t.leafv
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "TreeEnsemble":
        with np.load(path, allow_pickle=False) as z:
            n_trees = int(z["n_trees"])
            n_out = int(z["n_out"])
            max_depth = int(z["max_depth"])
            head = str(z["head"])
            seed = int(z["seed"])
            trees = [
                _Tree(
                    z[f"t{i}_feature"].astype(np.int64),
                    z[f"t{i}_threshold"].astype(np.float64),
                    z[f"t{i}_left"].astype(np.int64),
                    z[f"t{i}_right"].astype(np.int64),
                    z[f"t{i}_leaf_value"].astype(np.float64),
                )
                for i in range(n_trees)
            ]
        e = cls(n_out=n_out, n_trees=n_trees, max_depth=max_depth, head=head, seed=seed)
        e.trees = trees
        return e


# SPEC.md 19.2: fixed boost shrinkage (learning rate) for v0.5. A dedicated
# spec field for the boost family is a documented follow-up (the improver tunes
# it indirectly via n_trees / depth, as it does for the bagged `tree` family).
BOOST_SHRINK = 0.1


class BoostingEnsemble:
    """Gradient-boosted (residual) CART ensemble (SPEC.md 19.2).

    A stronger answer than the bagged `TreeEnsemble`: instead of averaging
    independent (each weak) trees, each round fits one regression CART per
    output column to the *residual* of the current model, shrunk by
    `BOOST_SHRINK`. The base estimate is the target mean (the class prior for a
    softmax head). `forward(x)` returns `(n, n_out)` exactly like the other
    families, so the evaluator/tasks treat it uniformly. Deterministic given a
    seed (G2); checkpoints are plain arrays (np.load allow_pickle=False).
    """

    def __init__(
        self,
        n_out: int,
        n_trees: int,
        max_depth: int,
        head: str,
        seed: int,
        shrink: float = BOOST_SHRINK,
    ) -> None:
        self.n_out = int(n_out)
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.head = head
        self.seed = int(seed)
        self.shrink = float(shrink)
        self.base = np.zeros(n_out)
        self.rounds: list[list[_Tree]] = []

    def fit(self, X: np.ndarray, y: np.ndarray, input_noise: float = 0.0) -> "BoostingEnsemble":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        min_leaf = _min_samples_leaf(n)
        if self.head == "softmax":
            target = np.zeros((n, self.n_out))
            target[np.arange(n), y.reshape(-1).astype(np.int64)] = 1.0
        else:
            target = y.reshape(n, self.n_out).astype(np.float64)
        self.base = target.mean(axis=0)
        pred = np.tile(self.base, (n, 1))
        rng = np.random.default_rng(self.seed)
        self.rounds = []
        for _t in range(self.n_trees):
            idx = rng.integers(0, n, n)  # bootstrap bag (SPEC.md 15, 19.2)
            xb = X[idx]
            if input_noise > 0.0:
                xb = xb + rng.normal(0.0, input_noise, xb.shape)
            col_trees = []
            for c in range(self.n_out):
                residual = target[idx, c] - pred[idx, c]  # functional gradient
                feat, thr, left, right, leafv = _fit_tree(
                    xb, residual.reshape(-1, 1), self.max_depth, min_leaf, 1, "mse",
                )
                col_trees.append(_Tree(feat, thr, left, right, leafv))
            self.rounds.append(col_trees)
            upd = np.zeros((n, self.n_out))
            for c in range(self.n_out):
                upd[:, c] = col_trees[c].predict_values(X).reshape(n)
            pred += self.shrink * upd
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        out = np.tile(self.base, (x.shape[0], 1))
        for col_trees in self.rounds:
            for c in range(self.n_out):
                out[:, c] += self.shrink * col_trees[c].predict_values(x).reshape(x.shape[0])
        return out

    # --- (de)serialization: plain arrays only (allow_pickle=False) ----------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {
            "n_out": np.array(self.n_out),
            "n_trees": np.array(self.n_trees),
            "max_depth": np.array(self.max_depth),
            "head": np.array(self.head),
            "seed": np.array(self.seed),
            "shrink": np.array(self.shrink),
            "base": self.base,
            "n_rounds": np.array(len(self.rounds)),
        }
        for t, col_trees in enumerate(self.rounds):
            for c in range(self.n_out):
                tt = col_trees[c]
                arrays[f"t{t}c{c}_feature"] = tt.feat
                arrays[f"t{t}c{c}_threshold"] = tt.thr
                arrays[f"t{t}c{c}_left"] = tt.left
                arrays[f"t{t}c{c}_right"] = tt.right
                arrays[f"t{t}c{c}_leaf_value"] = tt.leafv
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "BoostingEnsemble":
        with np.load(path, allow_pickle=False) as z:
            n_out = int(z["n_out"])
            n_trees = int(z["n_trees"])
            max_depth = int(z["max_depth"])
            head = str(z["head"])
            seed = int(z["seed"])
            shrink = float(z["shrink"]) if "shrink" in z.files else BOOST_SHRINK
            base = z["base"].astype(np.float64)
            n_rounds = int(z["n_rounds"])
            rounds: list[list[_Tree]] = []
            for t in range(n_rounds):
                col_trees = []
                for c in range(n_out):
                    col_trees.append(_Tree(
                        z[f"t{t}c{c}_feature"].astype(np.int64),
                        z[f"t{t}c{c}_threshold"].astype(np.float64),
                        z[f"t{t}c{c}_left"].astype(np.int64),
                        z[f"t{t}c{c}_right"].astype(np.int64),
                        z[f"t{t}c{c}_leaf_value"].astype(np.float64),
                    ))
                rounds.append(col_trees)
        e = cls(n_out=n_out, n_trees=n_trees, max_depth=max_depth, head=head, seed=seed, shrink=shrink)
        e.base = base
        e.rounds = rounds
        return e
