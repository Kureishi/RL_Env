"""Task-level tests (SPEC.md T1 + split/dataset invariants);
acceptance A1 (SPEC.md 12: the core task invariants behind the A1-A4 core
acceptance on cartpole-v1)."""
import numpy as np
import pytest

from autorefine.tasks.cartpole import CartPoleV1


def test_reset_is_deterministic_same_seed_bit_exact():
    """T1: same seed => identical first 1000 transitions (bit-exact)."""
    def rollout(seed):
        task = CartPoleV1(seed=seed)
        st = task.initial_conditions("holdout", 1).copy()  # keep (1, 4) batch shape
        out = []
        for _ in range(1000):
            a = int(task.reference_action(st)[0])
            st, term = task.step_vec(st, np.array([a]))
            out.append(st[0])
            if term[0]:
                st = task.initial_conditions("holdout", 1).copy()
        return np.stack(out)

    a = rollout(7)
    b = rollout(7)
    assert np.array_equal(a, b)  # bit-exact
    c = rollout(8)
    assert not np.array_equal(a, c)


def test_splitted_initial_conditions_deterministic_and_distinct():
    task = CartPoleV1(seed=7)
    for split in ("train", "val", "holdout", "gen"):
        a = task.initial_conditions(split, 50)
        b = task.initial_conditions(split, 50)
        assert np.array_equal(a, b), f"split {split} not deterministic"
    holdout = task.initial_conditions("holdout", 50)
    train = task.initial_conditions("train", 50)
    assert not np.allclose(holdout, train)


def test_episode_survival_within_bounds():
    task = CartPoleV1(seed=7)
    ics = task.initial_conditions("holdout", 30)
    for ic in ics:
        st = ic[None].copy()
        n = 0
        for _ in range(task.max_steps):
            a = task.reference_action(st)[0]
            st, term = task.step_vec(st, np.array([a]))
            n += 1
            if term[0]:
                break
        assert 0 < n <= task.max_steps


def test_dataset_invariants():
    task = CartPoleV1(seed=7)
    X, y = task.make_dataset()
    assert X.shape[0] > 1000
    assert X.shape[1] == task.state_dim
    assert set(np.unique(y)).issubset({0, 1})
    # margin property: every labeled state has a confident reference decision
    u = task.control_signal(X)
    assert np.all(np.abs(u) >= task.LABEL_MARGIN)
    # label consistency: label 1 iff u >= 0
    assert np.array_equal(y, (u >= 0).astype(np.int64))
    # balance is sane (not degenerate)
    frac = y.mean()
    assert 0.35 < frac < 0.65


def test_dataset_deterministic():
    a = CartPoleV1(seed=7).make_dataset()
    b = CartPoleV1(seed=7).make_dataset()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
