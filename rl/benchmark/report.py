"""
Report generation for benchmark results.

Produces:
- ASCII table (stdout)
- CSV (one row per algorithm)
- JSON (full structured output including per-episode data and stats)
"""
from __future__ import annotations
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

from rl.benchmark.metrics import AlgorithmResult
from rl.benchmark.statistical import compare_all_pairs, PairwiseComparison


def _json_safe(obj):
    """Recursively replace NaN/Infinity with None so the output is STANDARD JSON.
    Python's json writes literal NaN/Infinity by default, which browsers (and any
    strict parser) reject — that's what broke the comparison page. Degenerate
    stats (e.g. zero-variance metrics from barely-trained models) produce these."""
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


_COL_W = 18   # column width for ASCII table


def _col(s: str, w: int = _COL_W) -> str:
    return str(s)[:w].ljust(w)


def _fcol(v: float, fmt: str = ".2f", w: int = _COL_W) -> str:
    return _col(f"{v:{fmt}}", w)


def print_summary_table(
    results: Dict[str, AlgorithmResult],
    reference: Optional[str] = None,
) -> None:
    """
    Print a comparison table to stdout.
    If `reference` is given (e.g. "fixed_random"), show % improvement column.
    """
    ref  = results.get(reference) if reference else None
    sep  = "-" * (_COL_W * 6 + 5)

    header = (
        _col("Algorithm")
        + _col("Avg Wait (s)")
        + _col("±σ")
        + _col("Throughput/step")
        + _col("Congestion")
        + _col("Avg Speed m/s")
    )
    if ref:
        header += _col("ΔWait vs ref (%)")

    print("\n" + sep)
    print(header)
    print(sep)

    for name, r in results.items():
        row = (
            _col(name)
            + _fcol(r.mean_wait)
            + _fcol(r.std_wait)
            + _fcol(r.mean_throughput, ".3f")
            + _fcol(r.mean_congestion, ".3f")
            + _fcol(r.mean_speed,      ".2f")
        )
        if ref and ref is not r:
            delta_pct = (r.mean_wait - ref.mean_wait) / (ref.mean_wait + 1e-9) * 100
            row += _fcol(delta_pct, "+.1f")
        elif ref:
            row += _col("(reference)")
        print(row)

    print(sep + "\n")


def print_ci_table(results: Dict[str, AlgorithmResult]) -> None:
    """Print 95% bootstrap CI for wait time and throughput."""
    sep = "-" * (_COL_W * 5 + 4)
    print(sep)
    print(_col("Algorithm") + _col("Wait CI low") + _col("Wait CI high")
          + _col("TP CI low") + _col("TP CI high"))
    print(sep)
    for name, r in results.items():
        print(
            _col(name)
            + _fcol(r.ci95_wait[0])
            + _fcol(r.ci95_wait[1])
            + _fcol(r.ci95_throughput[0], ".3f")
            + _fcol(r.ci95_throughput[1], ".3f")
        )
    print(sep + "\n")


def print_pairwise_table(results: Dict[str, AlgorithmResult]) -> None:
    """Print pairwise statistical test results for wait time."""
    comps_wait = compare_all_pairs(results, metric="wait")
    comps_tp   = compare_all_pairs(results, metric="throughput")

    print("=== Pairwise comparisons (Wilcoxon signed-rank, α=0.05) ===\n")
    print("Metric: avg_wait_time_s")
    for c in comps_wait:
        print(f"  {c}")
    print("\nMetric: throughput_per_step")
    for c in comps_tp:
        print(f"  {c}")
    print()


def save_csv(results: Dict[str, AlgorithmResult], path: str) -> None:
    """Save summary CSV — one row per algorithm."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "algorithm", "config_label", "n_episodes",
        "mean_wait_s", "std_wait_s", "ci95_wait_lo", "ci95_wait_hi",
        "mean_throughput", "std_throughput", "ci95_tp_lo", "ci95_tp_hi",
        "mean_congestion", "mean_speed_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, r in results.items():
            writer.writerow({
                "algorithm":      name,
                "config_label":   r.config_label,
                "n_episodes":     len(r.episodes),
                "mean_wait_s":    f"{r.mean_wait:.4f}",
                "std_wait_s":     f"{r.std_wait:.4f}",
                "ci95_wait_lo":   f"{r.ci95_wait[0]:.4f}",
                "ci95_wait_hi":   f"{r.ci95_wait[1]:.4f}",
                "mean_throughput":f"{r.mean_throughput:.6f}",
                "std_throughput": f"{r.std_throughput:.6f}",
                "ci95_tp_lo":     f"{r.ci95_throughput[0]:.6f}",
                "ci95_tp_hi":     f"{r.ci95_throughput[1]:.6f}",
                "mean_congestion":f"{r.mean_congestion:.4f}",
                "mean_speed_ms":  f"{r.mean_speed:.4f}",
            })
    print(f"[report] CSV saved → {path}")


def save_json(results: Dict[str, AlgorithmResult], path: str) -> None:
    """Save full structured JSON including per-episode data."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {}
    for name, r in results.items():
        out[name] = {
            "config_label":   r.config_label,
            "summary": {
                "mean_wait_s":      r.mean_wait,
                "std_wait_s":       r.std_wait,
                "ci95_wait":        list(r.ci95_wait),
                "mean_throughput":  r.mean_throughput,
                "std_throughput":   r.std_throughput,
                "ci95_throughput":  list(r.ci95_throughput),
                "mean_congestion":  r.mean_congestion,
                "mean_speed_ms":    r.mean_speed,
            },
            "episodes": [
                {
                    "id":                    e.episode_id,
                    "avg_wait_time_s":       e.avg_wait_time_s,
                    "max_wait_time_s":       e.max_wait_time_s,
                    "throughput_per_step":   e.throughput_per_step,
                    "total_vehicles_served": e.total_vehicles_served,
                    "congestion_spread":     e.congestion_spread,
                    "avg_speed_ms":          e.avg_speed_ms,
                    "steps_completed":       e.steps_completed,
                }
                for e in r.episodes
            ],
        }

    # Append pairwise stats
    if len(results) >= 2:
        comps = compare_all_pairs(results, metric="wait") + \
                compare_all_pairs(results, metric="throughput")
        out["_pairwise_stats"] = [
            {
                "a": c.name_a, "b": c.name_b, "metric": c.metric,
                "W": c.W, "p_value": c.p_value,
                "cohens_d": c.d, "effect_size": c.effect_size,
                "significant": c.significant,
            }
            for c in comps
        ]

    with open(path, "w") as f:
        # allow_nan=False guarantees standard JSON; _json_safe already turned any
        # non-finite value into null so this won't raise.
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)
    print(f"[report] JSON saved → {path}")
