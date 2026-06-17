"""
Unit tests for Fase 5: benchmark metrics, statistical tests, and report generation.
All tests run without a C++ server.
"""
from __future__ import annotations
import json
import math
import os
import tempfile

import numpy as np
import pytest

from rl.benchmark.metrics import (
    EpisodeMetrics, AlgorithmResult, _bootstrap_ci,
)
from rl.benchmark.statistical import (
    wilcoxon_signed_rank,
    mann_whitney_u,
    cohens_d,
    interpret_d,
    compare_all_pairs,
    PairwiseComparison,
    _rank_with_ties,
)
from rl.benchmark.report import (
    print_summary_table, print_ci_table, print_pairwise_table,
    save_csv, save_json,
)


# ---- Fixtures ----

def _make_episode(ep_id: int, wait: float, tp: float, cong: float) -> EpisodeMetrics:
    return EpisodeMetrics(
        total_vehicles_served=tp * 1000,
        throughput_per_step=tp,
        avg_wait_time_s=wait,
        max_wait_time_s=wait * 2,
        congestion_spread=cong,
        avg_speed_ms=10.0,
        steps_completed=1000,
        episode_id=ep_id,
    )


def _make_result(name: str, waits, tps=None, congs=None) -> AlgorithmResult:
    waits = list(waits)
    tps   = list(tps)   if tps   else [0.01] * len(waits)
    congs = list(congs) if congs else [0.3]  * len(waits)
    r = AlgorithmResult(name=name, config_label="test")
    for i, (w, t, c) in enumerate(zip(waits, tps, congs)):
        r.episodes.append(_make_episode(i, w, t, c))
    r.compute_summary()
    return r


# ---- EpisodeMetrics / AlgorithmResult ----

class TestAlgorithmResult:
    def test_compute_summary_mean(self):
        r = _make_result("a", [10.0, 20.0, 30.0])
        assert r.mean_wait == pytest.approx(20.0)

    def test_compute_summary_std(self):
        r = _make_result("a", [10.0, 20.0, 30.0])
        expected_std = float(np.std([10.0, 20.0, 30.0], ddof=1))
        assert r.std_wait == pytest.approx(expected_std, rel=1e-5)

    def test_ci_order(self):
        r = _make_result("a", np.random.default_rng(0).normal(50, 10, 30))
        lo, hi = r.ci95_wait
        assert lo <= r.mean_wait <= hi

    def test_ci_narrows_with_more_episodes(self):
        rng = np.random.default_rng(1)
        r10 = _make_result("a", rng.normal(50, 10, 10))
        r50 = _make_result("a", rng.normal(50, 10, 50))
        width10 = r10.ci95_wait[1] - r10.ci95_wait[0]
        width50 = r50.ci95_wait[1] - r50.ci95_wait[0]
        assert width50 < width10

    def test_empty_result_no_crash(self):
        r = AlgorithmResult(name="empty", config_label="test")
        r.compute_summary()   # should not raise
        assert r.mean_wait == 0.0

    def test_single_episode(self):
        r = _make_result("a", [42.0])
        assert r.mean_wait == pytest.approx(42.0)
        assert r.ci95_wait == (pytest.approx(42.0), pytest.approx(42.0))


class TestBootstrapCI:
    def test_symmetric_data(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lo, hi = _bootstrap_ci(data, n_samples=5000, seed=0)
        assert lo < np.mean(data) < hi

    def test_single_element(self):
        lo, hi = _bootstrap_ci(np.array([7.0]))
        assert lo == hi == 7.0


# ---- Statistical tests ----

class TestWilcoxon:
    def test_identical_samples_p1(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = wilcoxon_signed_rank(x, x)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_clearly_different_p_small(self):
        rng = np.random.default_rng(0)
        x   = rng.normal(10,  1, 30)
        y   = rng.normal(100, 1, 30)
        _, p = wilcoxon_signed_rank(x, y)
        assert p < 0.001

    def test_p_in_range(self):
        rng = np.random.default_rng(42)
        x   = rng.normal(0, 1, 20)
        y   = rng.normal(0.5, 1, 20)
        _, p = wilcoxon_signed_rank(x, y)
        assert 0.0 <= p <= 1.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank(np.array([1, 2, 3]), np.array([1, 2]))

    def test_all_zeros_diffs(self):
        x = np.ones(10)
        y = np.ones(10)
        W, p = wilcoxon_signed_rank(x, y)
        assert p == pytest.approx(1.0, abs=0.01)


class TestMannWhitneyU:
    def test_clearly_different(self):
        x = np.arange(1, 11, dtype=float)
        y = np.arange(20, 30, dtype=float)
        _, p = mann_whitney_u(x, y)
        assert p < 0.001

    def test_p_in_range(self):
        rng = np.random.default_rng(0)
        _, p = mann_whitney_u(rng.normal(0, 1, 20), rng.normal(0, 1, 20))
        assert 0.0 <= p <= 1.0


class TestCohensD:
    def test_no_effect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(x, x)
        assert abs(d) < 1e-6

    def test_large_effect(self):
        x = np.zeros(50)
        y = np.ones(50) * 10
        d = cohens_d(x, y)
        assert abs(d) > 2.0

    def test_interpret_d(self):
        assert interpret_d(0.1) == "negligible"
        assert interpret_d(0.3) == "small"
        assert interpret_d(0.6) == "medium"
        assert interpret_d(1.2) == "large"
        assert interpret_d(-1.2) == "large"

    def test_sign(self):
        x = np.array([10.0, 10.0, 10.0])
        y = np.array([5.0,  5.0,  5.0])
        d = cohens_d(x, y)
        assert d > 0   # x > y


class TestRankWithTies:
    def test_no_ties(self):
        r = _rank_with_ties(np.array([3.0, 1.0, 2.0]))
        assert list(r) == [3.0, 1.0, 2.0]

    def test_all_ties(self):
        r = _rank_with_ties(np.array([5.0, 5.0, 5.0]))
        # Average of ranks 1,2,3 = 2
        assert all(v == 2.0 for v in r)

    def test_partial_ties(self):
        r = _rank_with_ties(np.array([1.0, 2.0, 2.0, 3.0]))
        assert r[0] == 1.0
        assert r[1] == 2.5
        assert r[2] == 2.5
        assert r[3] == 4.0


class TestCompareAllPairs:
    def test_returns_n_choose_2_comparisons(self):
        results = {
            "a": _make_result("a", np.random.default_rng(0).normal(50, 5, 20)),
            "b": _make_result("b", np.random.default_rng(1).normal(40, 5, 20)),
            "c": _make_result("c", np.random.default_rng(2).normal(30, 5, 20)),
        }
        comps = compare_all_pairs(results, metric="wait")
        assert len(comps) == 3   # C(3,2) = 3

    def test_significant_comparison(self):
        rng = np.random.default_rng(0)
        results = {
            "good": _make_result("good", rng.normal(20, 2, 30)),
            "bad":  _make_result("bad",  rng.normal(80, 2, 30)),
        }
        comps = compare_all_pairs(results, metric="wait")
        assert comps[0].significant

    def test_comparison_fields(self):
        results = {
            "a": _make_result("a", np.ones(10) * 50),
            "b": _make_result("b", np.ones(10) * 30),
        }
        c = compare_all_pairs(results, metric="wait")[0]
        assert c.name_a in ("a", "b")
        assert c.name_b in ("a", "b")
        assert c.metric == "wait"
        assert isinstance(c.p_value, float)
        assert isinstance(c.d, float)
        assert isinstance(c.effect_size, str)


# ---- Report generation ----

class TestReportGeneration:
    def _make_results(self):
        rng = np.random.default_rng(99)
        return {
            "fixed_random": _make_result("fixed_random", rng.normal(70, 8, 15)),
            "ppo":          _make_result("ppo",          rng.normal(45, 5, 15)),
            "ippo_gnn":     _make_result("ippo_gnn",     rng.normal(38, 4, 15)),
        }

    def test_print_summary_no_crash(self, capsys):
        results = self._make_results()
        print_summary_table(results, reference="fixed_random")
        out = capsys.readouterr().out
        assert "fixed_random" in out
        assert "ppo" in out

    def test_print_ci_no_crash(self, capsys):
        results = self._make_results()
        print_ci_table(results)
        out = capsys.readouterr().out
        assert "Algorithm" in out

    def test_print_pairwise_no_crash(self, capsys):
        results = self._make_results()
        print_pairwise_table(results)
        out = capsys.readouterr().out
        assert "Wilcoxon" in out

    def test_save_csv(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            save_csv(results, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "fixed_random" in content
            assert "mean_wait_s" in content

    def test_save_json_structure(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            save_json(results, path)
            with open(path) as f:
                data = json.load(f)
        assert "fixed_random" in data
        assert "summary" in data["fixed_random"]
        assert "episodes" in data["fixed_random"]
        assert "_pairwise_stats" in data

    def test_json_pairwise_count(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            save_json(results, path)
            with open(path) as f:
                data = json.load(f)
        # 3 algos, 2 metrics → C(3,2)*2 = 6 comparisons
        assert len(data["_pairwise_stats"]) == 6

    def test_json_episode_count(self):
        n = 12
        rng = np.random.default_rng(0)
        results = {"a": _make_result("a", rng.normal(50, 5, n))}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            save_json(results, path)
            with open(path) as f:
                data = json.load(f)
        assert len(data["a"]["episodes"]) == n


# ---- Mock benchmark end-to-end ----

class TestMockBenchmark:
    def test_mock_run_produces_results(self):
        import sys, types

        # Build fake args namespace
        class Args:
            config     = "config/city_small.yaml"
            episodes   = 10
            mock       = True
            reference  = "fixed_random"
            output_dir = "/tmp/trafficrl_test_bench"

        from rl.training.benchmark import _run_mock_benchmark
        results = _run_mock_benchmark(Args())
        assert len(results) == 4
        assert "fixed_random" in results
        assert "hrl" in results

    def test_mock_hrl_better_than_fixed(self):
        class Args:
            config     = "config/city_small.yaml"
            episodes   = 30
            mock       = True
            reference  = "fixed_random"
            output_dir = "/tmp/trafficrl_test_bench"

        from rl.training.benchmark import _run_mock_benchmark
        results = _run_mock_benchmark(Args())
        assert results["hrl"].mean_wait < results["fixed_random"].mean_wait

    def test_mock_full_pipeline_no_crash(self, capsys):
        """End-to-end: mock data → reports."""
        class Args:
            config     = "config/city_small.yaml"
            episodes   = 20
            mock       = True
            reference  = "fixed_random"
            output_dir = "/tmp/trafficrl_test_bench_full"

        from rl.training.benchmark import _run_mock_benchmark
        import os
        os.makedirs(Args.output_dir, exist_ok=True)

        results = _run_mock_benchmark(Args())
        for r in results.values():
            assert len(r.episodes) == Args.episodes

        print_summary_table(results, reference=Args.reference)
        print_ci_table(results)
        print_pairwise_table(results)

        csv_path  = os.path.join(Args.output_dir, "test.csv")
        json_path = os.path.join(Args.output_dir, "test.json")
        save_csv(results, csv_path)
        save_json(results, json_path)
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)
