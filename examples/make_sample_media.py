"""v0.10 sample media data (SPEC.md 24.6): synthesize labelled image + audio
directories and fit them with `autorefine fit` — the modality-as-task flow.

    python examples/make_sample_media.py            # writes examples/data/
    python -m autorefine fit --data examples/data/tone_clips    --target 90
    python -m autorefine fit --data examples/data/image_shapes  --target 90

- `tone_clips/`  — 220 Hz vs 440 Hz sine clips, 16-bit PCM WAV, stdlib
  `wave` (zero dependencies, SPEC.md 24.4);
- `image_shapes/` — ring vs cross patterns (Pillow when available —
  `pip install autorefine[image]`, SPEC.md 24.1).

Deterministic given --seed (G2); re-running overwrites the samples.
"""
from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def make_tone_clips(out: Path, n: int = 20, sr: int = 16000, seed: int = 7) -> Path:
    """Two tone classes as 16-bit PCM WAV subfolders (SPEC.md 24.4/24.6)."""
    rng = np.random.default_rng(seed)
    dur = 1.0
    t = np.arange(int(sr * dur)) / sr
    for name, freq in (("low_tone", 220.0), ("high_tone", 440.0)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            amp = 0.5 + 0.3 * rng.random()          # seeded per-clip variation
            phase = rng.random() * 2 * math.pi
            sig = amp * np.sin(2 * math.pi * freq * t + phase)
            pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            with wave.open(str(d / f"clip{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
    return out


def make_image_shapes(out: Path, n: int = 20, size: int = 64, seed: int = 7) -> Path:
    """Two geometric pattern classes as PNG subfolders (SPEC.md 24.3/24.6)."""
    try:
        from PIL import Image
    except ImportError:
        print("image samples need Pillow: pip install autorefine[image] "
              "(SPEC.md 24.1) — tone_clips/ is still usable without it")
        return out
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy)
    ring = (np.abs(r - size * 0.30) < size * 0.07).astype(np.float64)
    cross = (np.minimum(np.abs(xx - cx), np.abs(yy - cy)) < size * 0.07
             ).astype(np.float64)
    for name, pattern in (("ring", ring), ("cross", cross)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            off = size * 0.05 * rng.standard_normal(2)   # seeded jitter
            img = np.roll(pattern, tuple(np.round(off).astype(int)),
                          axis=(0, 1))
            gray = (img * 255.0).astype(np.uint8)
            Image.fromarray(gray, mode="L").save(d / f"shape{i:02d}.png")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DATA))
    ap.add_argument("--n", type=int, default=20, help="clips/images per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out_dir)
    tones = make_tone_clips(out / "tone_clips", n=args.n, seed=args.seed)
    shapes = make_image_shapes(out / "image_shapes", n=args.n, seed=args.seed)
    print(f"wrote {tones} and {shapes}")
    print("try:")
    print(f"  python -m autorefine fit --data {tones} "
          f"--target 90 --experiments 20")
    print(f"  python -m autorefine fit --data {shapes} "
          f"--target 90 --experiments 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
