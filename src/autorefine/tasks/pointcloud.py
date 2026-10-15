"""PointCloudTask: a directory of labelled 3-D point-cloud files as a
fitting task (SPEC.md 92.4, v0.78, A82).

The 3-D-geometry modality: a pure-stdlib ASCII parser (92.4.1 — PLY
`element vertex` blocks and OBJ `v` lines, zero new dependencies) yields a
set of (x, y, z) points per file; the feature is a fixed 83-d vector
(92.4.2) of 19 shape scalars (centroid statistics, bounding-box measures,
and the three principal-axis variances from a symmetric 3×3 covariance
eigendecomposition) plus a 4×4×4 voxel-occupancy histogram (64 log1p
counts over a unit-normalized grid). The labelled-directory protocol
(24.2) is the shared one — head inference, seed split, train-only
standardization, the §22.1 score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task
from .media import (_PC_EXTS, class_label_str, collect_items,
                    resolve_labels, split_indices)

_N_SCALARS = 19
_VOXEL = 4
_N_VOXELS = _VOXEL ** 3  # 64
POINTCLOUD_FEATURE_DIM = _N_SCALARS + _N_VOXELS  # 83 (92.4.2)


def _parse_ply(p: Path) -> np.ndarray:
    """One ASCII PLY file → (n, 3) points (92.4.1).

    Reads the header up to `end_header`, finds the `element vertex` block
    and its `property` list, locates the `x`/`y`/`z` columns, then reads
    the element's vertex lines and returns their (x, y, z). Binary PLY and
    a missing x/y/z column are a fail-loud `ValueError` (the ASCII
    contract, 92.4.1)."""
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip().lower() != "ply":
        raise ValueError(f"{p}: not a PLY file (missing 'ply' magic)")
    try:
        end_i = next(i for i, l in enumerate(lines)
                     if l.strip().lower() == "end_header")
    except StopIteration:
        raise ValueError(f"{p}: PLY header has no end_header") from None
    header = lines[:end_i]
    if any("binary" in l.lower() for l in header):
        raise ValueError(f"{p}: binary PLY is not supported — write ASCII "
                         f"(format ascii 1.0) (SPEC.md 92.4.1)")
    vertex_count = 0
    props: list[str] = []
    in_vertex = False
    for l in header[1:]:
        toks = l.split()
        if not toks:
            continue
        if toks[0].lower() == "element":
            name = toks[1].lower()
            in_vertex = (name == "vertex")
            if in_vertex:
                vertex_count = int(toks[2])
                props = []
            else:
                in_vertex = False
        elif toks[0].lower() == "property" and in_vertex:
            # skip list properties (property list <t> <name>)
            if toks[1].lower() != "list":
                props.append(toks[-1].lower())
    cols = {}
    for want in ("x", "y", "z"):
        if want in props:
            cols[want] = props.index(want)
    if len(cols) != 3:
        raise ValueError(f"{p}: PLY vertex element has no x/y/z properties "
                         f"(properties: {props}) (SPEC.md 92.4.1)")
    pts: list[list[float]] = []
    for l in lines[end_i + 1:end_i + 1 + vertex_count]:
        vals = l.replace(",", " ").split()
        if len(vals) < max(cols.values()) + 1:
            continue
        pts.append([float(vals[cols["x"]]), float(vals[cols["y"]]),
                    float(vals[cols["z"]])])
    if not pts:
        raise ValueError(f"{p}: PLY vertex element has no points")
    return np.array(pts, dtype=np.float64)


def _parse_obj(p: Path) -> np.ndarray:
    """One OBJ file → (n, 3) points (92.4.1). `v` vertex lines only
    (normal `vn` / texture `vt` lines are ignored)."""
    pts: list[list[float]] = []
    for l in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        toks = l.split()
        if not toks or toks[0].lower() != "v":
            continue
        if len(toks) < 4:
            continue
        pts.append([float(toks[1]), float(toks[2]), float(toks[3])])
    if not pts:
        raise ValueError(f"{p}: OBJ file has no vertex (v) lines")
    return np.array(pts, dtype=np.float64)


def load_pointcloud(p: Path) -> np.ndarray:
    """SPEC.md 92.4.1 (v0.78): one `.ply`/`.obj` file → (n, 3) points."""
    ext = p.suffix.lower()
    if ext == ".ply":
        return _parse_ply(p)
    if ext == ".obj":
        return _parse_obj(p)
    raise ValueError(
        f"{p}: unsupported point-cloud extension {ext!r} — accepted: "
        f"{', '.join(_PC_EXTS)} (SPEC.md 92.4.1)")


def _principal_vars(Xc: np.ndarray) -> np.ndarray:
    """The three principal-axis variances (descending) of a centered
    (n, 3) cloud — `eigvalsh` of the symmetric covariance (deterministic,
    G2; only the eigenvalues are used, so eigenvector sign is irrelevant)."""
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(np.asarray(eig, dtype=np.float64), 0.0, None)
    return np.sort(eig)[::-1]  # λ1 ≥ λ2 ≥ λ3


def pointcloud_features(pts: np.ndarray, voxel: int = _VOXEL) -> np.ndarray:
    """SPEC.md 92.4.2 (v0.78): the raw (83,) feature of one point cloud.

    19 shape scalars (log1p point count, rms/max radius, bounding-box
    volume / diagonal / extents, fill ratio, the three principal-axis
    variances λ1≥λ2≥λ3, their stds, and the aspect / flatness / thinness /
    sphericity ratios) plus a `voxel`³ (4³ = 64) log1p occupancy histogram
    over the unit-normalized bounding box. A single-point (degenerate box)
    cloud is well-defined (zero extents, all counts in one voxel).
    Deterministic given the points (G2).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"point cloud must be (n, 3), got {pts.shape}")
    n = pts.shape[0]
    c = pts.mean(axis=0)
    Xc = pts - c
    lam = _principal_vars(Xc)
    lam1, lam2, lam3 = (float(lam[0]), float(lam[1]), float(lam[2]))
    extents = pts.max(axis=0) - pts.min(axis=0)
    bbox_vol = float(extents.prod())
    diag = float(np.sqrt((extents ** 2).sum()))
    rms = float(np.sqrt(((Xc ** 2).sum(axis=1)).mean()))
    maxr = float((Xc ** 2).sum(axis=1).max())
    maxr = float(np.sqrt(maxr))
    lam1_safe = max(lam1, 1e-12)
    scalars = np.array([
        float(np.log1p(n)),                 # 0
        rms,                                # 1
        maxr,                               # 2
        float(np.linalg.norm(pts, axis=1).mean()),  # 3
        bbox_vol,                           # 4
        diag,                               # 5
        float(extents.max()),               # 6
        float(extents.min()),               # 7
        float(n / (bbox_vol + 1e-9)),       # 8
        lam1, lam2, lam3,                   # 9,10,11
        float(np.sqrt(lam1)), float(np.sqrt(lam2)),
        float(np.sqrt(lam3)),               # 12,13,14
        float(lam1 / lam1_safe),            # 15 (principal variance ratio = 1)
        float(1.0 - lam3 / lam1_safe),      # 16 (flatness)
        float(1.0 - lam2 / lam1_safe),      # 17 (thinness)
        float((lam1 + lam2 + lam3) / (3.0 * lam1_safe)),  # 18 (sphericity)
    ], dtype=np.float64)

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = hi - lo
    span = np.where(span > 0.0, span, 1.0)  # degenerate axis -> unit span
    t = (pts - lo) / span  # (n, 3) in [0, 1]
    idx = np.minimum((t * voxel).astype(np.int64), voxel - 1)
    vox = np.zeros(_N_VOXELS, dtype=np.float64)
    for i, j, k in idx:
        vox[i * voxel * voxel + j * voxel + k] += 1.0
    return np.concatenate([scalars, np.log1p(vox)])


class PointCloudTask(Task):
    name = "pointcloud"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)
    metric = "accuracy"
    capabilities = frozenset({"media"})
    feature_mean = None  # instance values (92.4)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2,
                 voxel: int = _VOXEL) -> None:
        if path is None:
            raise ValueError(
                "PointCloudTask needs a directory of labelled point-cloud "
                "files: use `autorefine fit --data DIR` or "
                "AutoRefineEnv(task='pointcloud', task_config={'path': "
                "DIR}) (SPEC.md 92.4)")
        self.seed = int(seed)
        self.path = str(path)
        self.voxel = int(voxel)
        self.split_frac = float(split_frac)
        self.label_name = "class"

        items = collect_items(self.path, _PC_EXTS, "pointcloud")
        x = np.array([pointcloud_features(load_pointcloud(p))
                      for p, _ in items], dtype=np.float64)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(x.shape[1])
        self.feature_names = (
            [f"shape scalar {i}" for i in range(_N_SCALARS)]
            + [f"voxel {i}" for i in range(_N_VOXELS)])
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"pointcloud-split")
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

    # --- decoding -------------------------------------------------------------
    def item_features(self, path: str | Path) -> np.ndarray:
        """SPEC.md 92.4 (v0.78): decode one point-cloud file to the raw
        (unstandardized) feature — the public wrapper `predict --item`
        calls."""
        return pointcloud_features(load_pointcloud(Path(path)))

    # --- protocol (SPEC.md 15, 24.2 — mirrors MidiTask) ----------------------
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

    def holdout_errors(self, model, n: int = 200, n_max: int = 8) -> list[dict]:
        """SPEC.md 92.4 (v0.78): the misclassified holdout items (file
        entries; the point cloud has no inline-renderable preview)."""
        if self.head != "softmax" or int(n_max) <= 0:
            return []
        x, y = self.holdout_rows(n)
        if len(x) == 0:
            return []
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        out: list[dict] = []
        for i in range(len(x)):
            if pred[i] == y[i]:
                continue
            p, _ = self._items[int(self._ho[i])]
            out.append({
                "file": p.name,
                "path": str(p),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            })
            if len(out) >= int(n_max):
                break
        return out
