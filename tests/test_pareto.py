"""Pareto frontier tests (SPEC.md 15, score-vs-train-time memory);
acceptance A6 (SPEC.md 12: the v0.2 extension coverage)."""
import numpy as np

from autorefine.pareto import ParetoFrontier


def test_dominance_logic():
    f = ParetoFrontier()
    f.add(10.0, 1.0, "a")   # dominated by b (better score, same time)
    f.add(12.0, 1.0, "b")   # dominates a
    f.add(8.0, 0.5, "c")    # undominated: cheaper
    f.add(11.0, 2.0, "d")   # dominated by b (worse score AND slower)
    front = f.frontier()
    hashes = [p["spec_hash"] for p in front]
    assert "b" in hashes and "c" in hashes
    assert "a" not in hashes and "d" not in hashes
    # sorted cheapest-first
    secs = [p["seconds"] for p in front]
    assert secs == sorted(secs)


def test_equal_points_are_not_dominated():
    f = ParetoFrontier()
    f.add(5.0, 1.0, "x")
    f.add(5.0, 1.0, "y")  # identical point: neither dominates the other
    assert len(f.frontier()) == 2


def test_best_score_at_budget():
    f = ParetoFrontier()
    f.add(10.0, 2.0, "slow")
    f.add(7.0, 0.5, "fast")
    assert f.best_score_at(0.4) is None
    assert f.best_score_at(0.5) == 7.0
    assert f.best_score_at(1.0) == 7.0
    assert f.best_score_at(2.5) == 10.0


def test_summary_shape_and_efficiency():
    f = ParetoFrontier()
    f.add(10.0, 2.0, "slow")
    f.add(8.0, 0.5, "fast")
    s = f.summary(time_budget=1.0)
    assert set(s) == {"pareto_frontier", "best_score_at_1s", "efficiency_at_1s"}
    assert s["best_score_at_1s"] == 8.0
    assert s["efficiency_at_1s"] == 0.8
    for p in s["pareto_frontier"]:
        assert set(p) == {"score", "train_seconds", "spec_hash"}


def test_empty_frontier_summary():
    s = ParetoFrontier().summary()
    assert s["pareto_frontier"] == []
    assert s["best_score_at_1s"] is None
    assert s["efficiency_at_1s"] is None
