"""GraphTask: a graph dataset (``nodes.csv`` + ``edges.csv``) as a
fitting task (SPEC.md 92.5, v0.78, A82).

The node-classification modality: one directory with
  * ``nodes.csv`` — one row per node; a label column (``label``/``target``/
    ``y``/``class``, else the last) and the numeric feature columns
    (an optional ``id`` column names the nodes);
  * ``edges.csv`` — two columns of node references (row indices, or ids
    when ``nodes.csv`` carries an id column).

The feature of a node is its own features **mean-aggregated over its
1-hop neighborhood** (92.5.1 — itself plus its neighbours, pure NumPy,
deterministic given the files, G2). The rows are the nodes: the §22.1
head inference over the node labels, the seed split, train-only
standardization, and the §22.1 score. There is no ``holdout_errors`` —
a node is not a displayable media file (92.5.2).
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np

from .base import Task
from .media import resolve_labels, split_indices

_LABEL_HINTS = ("label", "target", "y", "class")
_ID_HINTS = ("id", "node", "node_id", "index", "idx")
_EDGE_HINTS = ("source", "src", "from", "u")
_EDGE_HINTS2 = ("target", "dst", "to", "v")


def _is_num(s: str) -> bool:
    try:
        float(s)
    except (TypeError, ValueError):
        return False
    return True


def _read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = _csv.reader(fh)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        rows = [[(c or "").strip() for c in raw] for raw in reader
                if any((c or "").strip() for c in raw)]
    return header, rows


def _pick_col(header: list[str], hints: tuple[str, ...]) -> int | None:
    low = [h.lower() for h in header]
    for h in hints:
        if h in low:
            return low.index(h)
    return None


def load_graph(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """SPEC.md 92.5.1 (v0.78): a graph directory →
    (node_features (n, d), raw node labels, [(u, v), ...] edges).

    `node_features` is the **raw** (unstandardized) per-node feature — the
    1-hop mean-neighborhood aggregation (92.5.1). `raw node labels` are the
    label-column values in row order (resolved to targets by the caller,
    24.2). Edges are 0-based node pairs (de-duplicated, both endpoints in
    range). Deterministic given the files (G2)."""
    d = Path(path)
    nodes_p = d / "nodes.csv"
    edges_p = d / "edges.csv"
    if not nodes_p.is_file() or not edges_p.is_file():
        raise ValueError(
            f"a graph task needs a directory with both nodes.csv and "
            f"edges.csv (SPEC.md 92.5); found under {path!r}: "
            f"nodes.csv={nodes_p.is_file()}, edges.csv={edges_p.is_file()}")
    n_header, n_rows = _read_table(nodes_p)
    if not n_rows:
        raise ValueError(f"{nodes_p}: no node rows")
    low = [h.lower() for h in n_header]
    label_i = _pick_col(n_header, _LABEL_HINTS)
    if label_i is None:
        label_i = len(n_header) - 1
    id_i = _pick_col(n_header, _ID_HINTS)
    # feature columns: every column that is numeric and not label/id
    def _col_is_numeric(ci: int) -> bool:
        return all(_is_num(r[ci]) for r in n_rows if ci < len(r))
    feats = [i for i in range(len(n_header))
             if i != label_i and (id_i is None or i != id_i)
             and _col_is_numeric(i)]
    if not feats:
        raise ValueError(
            f"{nodes_p}: no numeric feature columns (columns: {n_header}) "
            f"(SPEC.md 92.5)")
    n = len(n_rows)
    X = np.zeros((n, len(feats)), dtype=np.float64)
    for i, r in enumerate(n_rows):
        for j, ci in enumerate(feats):
            X[i, j] = float(r[ci]) if ci < len(r) else 0.0
    labels = [r[label_i] for r in n_rows if label_i < len(r)]
    if len(labels) != n:
        raise ValueError(f"{nodes_p}: some rows are missing the label "
                         f"column (SPEC.md 92.5)")

    # id -> row map (when nodes.csv carries an id column)
    id_map: dict[str, int] | None = None
    if id_i is not None:
        id_map = {}
        for i, r in enumerate(n_rows):
            if id_i < len(r):
                id_map[r[id_i].lower()] = i

    e_header, e_rows = _read_table(edges_p)
    if len(e_header) < 2:
        raise ValueError(f"{edges_p}: needs at least two columns "
                         f"(source, target), got {e_header} (SPEC.md 92.5)")
    u_i = _pick_col(e_header, _EDGE_HINTS)
    v_i = _pick_col(e_header, _EDGE_HINTS2)
    if u_i is None or v_i is None:
        u_i, v_i = 0, 1

    def _resolve(cell: str) -> int | None:
        if id_map is not None:
            return id_map.get(cell.lower())
        try:
            return int(float(cell))
        except (TypeError, ValueError):
            return None

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in e_rows:
        if u_i >= len(r) or v_i >= len(r):
            continue
        u, v = _resolve(r[u_i]), _resolve(r[v_i])
        if u is None or v is None or not (0 <= u < n and 0 <= v < n):
            continue
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v))
    return X, labels, edges


def aggregate_neighbors(X: np.ndarray, edges: list[tuple[int, int]],
                        n: int | None = None) -> np.ndarray:
    """SPEC.md 92.5.1 (v0.78): 1-hop mean neighborhood aggregation —
    each node's feature becomes the mean of its own feature and its
    neighbours' (itself always included). Pure NumPy, deterministic
    (G2); an isolated node keeps its own feature."""
    X = np.asarray(X, dtype=np.float64)
    nn = int(X.shape[0]) if n is None else int(n)
    adj: list[list[int]] = [[] for _ in range(nn)]
    for u, v in edges:
        if 0 <= u < nn and 0 <= v < nn:
            adj[u].append(v)
            adj[v].append(u)
    agg = np.empty_like(X)
    for i in range(nn):
        idxs = [i] + adj[i]
        agg[i] = X[idxs].mean(axis=0)
    return agg


class GraphTask(Task):
    name = "graph"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train nodes)
    metric = "accuracy"
    capabilities = frozenset()
    feature_mean = None  # instance values (92.5)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2) -> None:
        if path is None:
            raise ValueError(
                "GraphTask needs a graph directory (nodes.csv + edges.csv): "
                "use `autorefine fit --data DIR` or "
                "AutoRefineEnv(task='graph', task_config={'path': DIR}) "
                "(SPEC.md 92.5)")
        d = Path(path)
        if not (d / "nodes.csv").is_file() or not (d / "edges.csv").is_file():
            raise ValueError(
                f"no graph dataset under {path!r} — a directory with both "
                f"nodes.csv and edges.csv is required (SPEC.md 92.5)")
        self.seed = int(seed)
        self.path = str(d)
        self.split_frac = float(split_frac)
        self.label_name = "class"

        X, raw_labels, edges = load_graph(d)
        n = X.shape[0]
        self._edges = edges
        self._n_nodes = n
        x = aggregate_neighbors(X, edges, n)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels(raw_labels)
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(x.shape[1])
        self.feature_names = [f"agg node feature {i}"
                              for i in range(x.shape[1])]
        tr, ho, ge = split_indices(n, self.seed, self.split_frac,
                                   b"graph-split")
        self._ho = ho
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self.feature_mean = mean
        self.feature_std = std
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))

    # --- protocol (SPEC.md 15, 92.5 — mirrors TextTask) ----------------------
    def _rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        s = (split or "").lower()
        if s.startswith("gen"):
            return self._x_ge, self._y_ge
        if s.startswith("train"):
            return self._x_tr, self._y_tr
        return self._x_ho, self._y_ho

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        x, _ = self._rows_for(split)
        return x[: max(1, int(n))] if len(x) else x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states

    def make_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._x_tr) if n_points is None else min(int(n_points), len(self._x_tr))
        return self._x_tr[:n], self._y_tr[:n]

    def score(self, model, split: str, n: int) -> float:
        x, y = self._rows_for(split)
        if len(x) == 0:
            return 0.0
        n = min(int(n), len(x))
        pred = np.asarray(model.forward(x[:n]), dtype=np.float64)
        if self.head == "softmax":
            return float(100.0 * (pred.argmax(axis=1) == y[:n]).mean())
        pred = pred.reshape(-1)
        ss_res = float(((pred - y[:n]) ** 2).sum())
        ss_tot = float(((y[:n] - y[:n].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]

    # no `holdout_errors`: a node is not a displayable media file (92.5.2)
