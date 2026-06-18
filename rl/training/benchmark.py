"""
Benchmark CLI — Fase 5: comparación de algoritmos.

Runs N evaluation episodes per algorithm and produces:
  - ASCII comparison table
  - 95% bootstrap CIs
  - Wilcoxon pairwise significance tests
  - CSV + JSON reports in rl/results/

Usage examples:

  # Baselines only (no trained models needed)
  python3 -m rl.training.benchmark --config config/city_small.yaml --episodes 20

  # Include trained models
  python3 -m rl.training.benchmark \\
      --config config/city_small.yaml \\
      --episodes 30 \\
      --ppo-model rl/models/ppo_centralized \\
      --ippo-model rl/models/ippo_gnn.pt \\
      --hrl-worker rl/models/hrl/worker.pt \\
      --hrl-manager rl/models/hrl/manager.pt

  # Dry run with mock data (no server required)
  python3 -m rl.training.benchmark --mock --episodes 50
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np

from rl.env.data_types import EnvConfig
from rl.benchmark.metrics import AlgorithmResult, EpisodeMetrics
from rl.benchmark.report import (
    print_summary_table, print_ci_table, print_pairwise_table,
    save_csv, save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TrafficRL Benchmark — Fase 5")
    p.add_argument("--config",       default="config/city_small.yaml")
    p.add_argument("--episodes",     type=int, default=20,
                   help="Evaluation episodes per algorithm")
    p.add_argument("--seed-offset",  type=int, default=1000,
                   help="Base seed for eval episodes (avoids train seeds)")
    p.add_argument("--ppo-model",    default=None,   help="Path to PPO centralized model")
    p.add_argument("--ippo-model",   default=None,   help="Path to IPPO+GNN .pt model")
    p.add_argument("--hrl-worker",   default=None,   help="Path to HRL worker .pt")
    p.add_argument("--hrl-manager",  default=None,   help="Path to HRL manager .pt")
    p.add_argument("--k-hops",       type=int, default=1)
    p.add_argument("--output-dir",   default="rl/results")
    p.add_argument("--mock",         action="store_true",
                   help="Use synthetic data — no C++ server required")
    p.add_argument("--reference",    default="fixed_random",
                   help="Reference algorithm for %% improvement column")
    return p.parse_args()


# ---- Mock runner for offline testing ----

def _make_mock_result(
    name: str,
    config_label: str,
    n: int,
    rng_seed: int,
    # tunable difficulty: worse wait → higher wait_base
    wait_base: float = 60.0,
    wait_noise: float = 15.0,
    tp_base: float = 0.008,
) -> AlgorithmResult:
    rng = np.random.default_rng(rng_seed)
    r   = AlgorithmResult(name=name, config_label=config_label)
    for i in range(n):
        r.episodes.append(EpisodeMetrics(
            total_vehicles_served = float(rng.uniform(50, 200)),
            throughput_per_step   = float(rng.normal(tp_base, tp_base * 0.2)),
            avg_wait_time_s       = float(max(1.0, rng.normal(wait_base, wait_noise))),
            max_wait_time_s       = float(rng.uniform(wait_base, wait_base * 3)),
            congestion_spread     = float(np.clip(rng.beta(2, 5), 0, 1)),
            avg_speed_ms          = float(rng.uniform(5, 15)),
            steps_completed       = 1000,
            episode_id            = i,
        ))
    r.compute_summary()
    return r


def _run_mock_benchmark(args) -> dict:
    label = os.path.splitext(os.path.basename(args.config))[0]
    n     = args.episodes
    # Mock performance: HRL > IPPO > PPO > fixed_random
    # fixed_random: period 500-700 steps, phase-1 green = 20-25% (≥ min_green).
    # N-S vehicles wait up to 560 steps per cycle → high avg_wait, low throughput.
    results = {
        "fixed_random":   _make_mock_result("fixed_random",   label, n, 0,  wait_base=180, tp_base=0.002),
        "ppo_centralized":_make_mock_result("ppo_centralized",label, n, 2,  wait_base=52,  tp_base=0.010),
        "ippo_gnn":       _make_mock_result("ippo_gnn",       label, n, 3,  wait_base=43,  tp_base=0.012),
        "hrl":            _make_mock_result("hrl",            label, n, 4,  wait_base=36,  tp_base=0.014),
    }
    return results


def _run_live_benchmark(args) -> dict:
    from rl.benchmark.runners import FixedRandomRunner

    env_cfg = EnvConfig.from_yaml(args.config)
    label   = os.path.splitext(os.path.basename(args.config))[0]
    n       = args.episodes
    so      = args.seed_offset
    results = {}
    runners = []

    print("[benchmark] Initializing runners…")

    # Realistic "badly configured city" baseline that RL must beat.
    fr = FixedRandomRunner(env_cfg)
    runners.append(fr)
    print("[benchmark] Running fixed_random…")
    t0 = time.perf_counter()
    results["fixed_random"] = fr.run_episodes(n, so, label)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    if args.ppo_model:
        from rl.benchmark.runners import PPOCentralizedRunner
        pr = PPOCentralizedRunner(env_cfg, args.ppo_model)
        runners.append(pr)
        print("[benchmark] Running ppo_centralized…")
        t0 = time.perf_counter()
        results["ppo_centralized"] = pr.run_episodes(n, so, label)
        print(f"  done in {time.perf_counter()-t0:.1f}s")

    if args.ippo_model:
        from rl.benchmark.runners import IPPOGNNRunner
        ir = IPPOGNNRunner(env_cfg, args.ippo_model, k_hops=args.k_hops)
        runners.append(ir)
        print("[benchmark] Running ippo_gnn…")
        t0 = time.perf_counter()
        results["ippo_gnn"] = ir.run_episodes(n, so, label)
        print(f"  done in {time.perf_counter()-t0:.1f}s")

    if args.hrl_worker and args.hrl_manager:
        from rl.benchmark.runners import HRLRunner
        hr = HRLRunner(env_cfg, args.hrl_worker, args.hrl_manager, k_hops=args.k_hops)
        runners.append(hr)
        print("[benchmark] Running hrl…")
        t0 = time.perf_counter()
        results["hrl"] = hr.run_episodes(n, so, label)
        print(f"  done in {time.perf_counter()-t0:.1f}s")

    for r in runners:
        r.close()

    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label = os.path.splitext(os.path.basename(args.config))[0]

    print(f"\n{'='*60}")
    print(f"TrafficRL Benchmark — Fase 5")
    print(f"Config: {args.config}   Episodes: {args.episodes}")
    print(f"Mode: {'MOCK (synthetic data)' if args.mock else 'LIVE (C++ server)'}")
    print(f"{'='*60}")

    if args.mock:
        results = _run_mock_benchmark(args)
    else:
        results = _run_live_benchmark(args)

    # ---- Report ----
    print_summary_table(results, reference=args.reference)
    print_ci_table(results)
    if len(results) >= 2:
        print_pairwise_table(results)

    csv_path  = os.path.join(args.output_dir, f"benchmark_{label}.csv")
    json_path = os.path.join(args.output_dir, f"benchmark_{label}.json")
    save_csv(results, csv_path)
    save_json(results, json_path)

    # ---- Final verdict ----
    if args.reference in results:
        ref_wait = results[args.reference].mean_wait
        print("=== Summary vs reference ===")
        for name, r in results.items():
            if name == args.reference:
                continue
            delta = (r.mean_wait - ref_wait) / (ref_wait + 1e-9) * 100
            marker = "✓" if delta < 0 else "✗"
            print(f"  {marker} {name}: {delta:+.1f}% wait  "
                  f"({r.mean_wait:.1f}s vs {ref_wait:.1f}s)")
        print()


if __name__ == "__main__":
    main()
