"""Train the AutoRefine meta-RL improver on the 8-bit parity dataset.

Uses the same public API the dashboard's RL tab drives (`RLRunner`,
`autorefine.rl_dashboard`): epsilon-greedy exploration so the REINFORCE
policy does not collapse onto a single (possibly task-invalid) action,
keep-best so the persisted policy is the best-episode checkpoint.

Run:  python examples/parity8/train_parity8_rl.py
Outputs:
  * the per-episode returns (the self-improvement curve)
  * the final gate verdict + best spec
  * the trained policy file  (policy_parity8.npz, next to the data)
  * the full run artifacts   (runs/<task>-seed7-<timestamp>/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from autorefine.rl_dashboard import RLRunner  # noqa: E402
from autorefine import save_policy  # noqa: E402

DATA = HERE / "parity8_train.csv"
POLICY_OUT = HERE / "policy_parity8.npz"


def on_update(u: dict) -> None:
    if u.get("episode_done"):
        print(f"  episode {u['episode']}: return {u['episode_return']:+.4f} "
              f"(eps={u['epsilon']:.3f})")


def main() -> int:
    print(f"data     : {DATA}  ({DATA.stat().st_size} bytes)")
    print("task     : 8-bit parity (XOR of 8 binary features)")
    print("policy   : meta-RL (REINFORCE), epsilon-greedy 0.25 -> 0.05")

    runner = RLRunner(
        str(DATA), label="parity", target=95.0, seed=7,
        experiments=8, max_seconds=900.0, max_train_seconds=30.0,
        runs_dir=str(HERE.parents[1] / "runs"),
        episodes=10, epsilon=0.25, epsilon_decay=0.85, keep_best=True)
    info = runner.start()
    print(f"baseline : {info['baseline_score']:.2f} (accuracy, 204-pt train)")
    result = runner.run_all(on_update=on_update)

    print()
    print("=== RL training complete ===")
    print(f"  episode returns : {['%.2f' % r for r in result['rl']['episode_returns']]}")
    print(f"  best episode    : {result['rl']['best']}")
    print(f"  policy updates  : {result['rl']['n_updates']}")
    sp = save_policy(runner.policy, str(POLICY_OUT))
    print(f"  policy saved to : {sp}")
    print()
    print("=== summary ===")
    print(json.dumps({k: result[k] for k in (
        "verdict", "target", "baseline_score", "final_best_score",
        "improvement_factor", "experiments_run", "wall_seconds",
        "finished_reason")}, indent=2))
    print("best spec:", json.dumps(result["best_spec"]))
    print("artifacts:", result["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
