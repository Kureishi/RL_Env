"""Sample image dataset as a flat directory + an ``index.csv`` (SPEC.md 24.2).

Whereas ``make_sample_media.py`` writes one subfolder per class, this writes
a *flat* directory of PNG files plus an ``index.csv`` of ``file,label`` rows —
the other label layout ``ImageTask`` accepts (SPEC.md 24.2): column 1 = file
path (relative to the directory), column 2 = an integer class id.

    python examples/make_sample_images_csv.py          # writes examples/data/image_flat/
    python -m autorefine fit --data examples/data/image_flat --target 90

Three classes, each a distinct geometric texture (deterministic given
``--seed``, G2):

  0  horizontal stripes
  1  vertical stripes
  2  checkerboard

Per-image seeded jitter + light Gaussian noise keeps the classes cleanly
separable (a ConvNet/MLP clears ``--target 90`` comfortably) while still
looking like real, imperfect scans. Pillow is required —
``pip install autorefine[image]`` (SPEC.md 24.1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def _texture(kind: str, size: int) -> np.ndarray:
    """A unit-valued (size, size) 0/1 pattern for one class."""
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "h":
        return (np.sin(yy * 2.0 * np.pi / 8.0) >= 0.0).astype(np.float64)
    if kind == "v":
        return (np.sin(xx * 2.0 * np.pi / 8.0) >= 0.0).astype(np.float64)
    if kind == "check":
        return (((yy // 8).astype(int) + (xx // 8).astype(int)) % 2
                ).astype(np.float64)
    raise ValueError(f"unknown texture {kind!r}")


def make_image_dir(out: Path, n: int = 12, size: int = 64,
                   seed: int = 7) -> Path:
    """Write ``n`` images per class into a flat dir + an ``index.csv``."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "image samples need Pillow: pip install autorefine[image] "
            "(SPEC.md 24.1)") from exc

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int]] = []
    idx = 0
    for label, kind in enumerate(("h", "v", "check")):
        base = _texture(kind, size)
        for _ in range(n):
            # small seeded shift so each image differs but stays on-class
            off = np.round(size * 0.04 * rng.standard_normal(2)).astype(int)
            pat = np.roll(base, tuple(off), axis=(0, 1))
            noise = rng.normal(0.0, 0.04, size=size)          # light scan noise
            gray = np.clip(40.0 + 175.0 * pat + 255.0 * noise, 0.0, 255.0)
            name = f"img{idx:03d}.png"
            Image.fromarray(gray.astype(np.uint8), mode="L").save(out / name)
            rows.append((name, label))
            idx += 1

    # index.csv: header `file,label`; relative paths; integer class ids
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("file,label\n")
        for name, label in rows:
            fh.write(f"{name},{label}\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write a flat image dir + index.csv for ImageTask testing.")
    ap.add_argument("--out-dir", default=str(DATA / "image_flat"))
    ap.add_argument("--n", type=int, default=12, help="images per class")
    ap.add_argument("--size", type=int, default=64, help="square pixel size")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = make_image_dir(Path(args.out_dir), n=args.n,
                         size=args.size, seed=args.seed)
    n_files = len(list(out.glob("*.png")))
    print(f"wrote {n_files} images + index.csv under {out}")
    print(f"now: python -m autorefine fit --data {out} --target 90")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
