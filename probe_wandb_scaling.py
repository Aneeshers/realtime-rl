#!/usr/bin/env python3
"""
probe_wandb_scaling.py

Probes wandb projects to find data for the co-scaling figure (fig:scaling):
  - Planning quality (return / win rate) vs simulation count
  - Inference latency vs simulation count

Projects probed:
  1. tetris_rt_kt_cross_eval       (k=1 model, k=1 eval env)
  2. pacman_kt_cross_eval_mature   (k=1 model, k=1 eval env)
  3. sokoban_gumbel_az_eval
  4. hex_inference_tournament

Run:
    python probe_wandb_scaling.py

Outputs a human-readable summary of what metrics/configs are available in each
project, and whether the co-scaling data is present.
"""

import wandb
import pandas as pd
from collections import defaultdict

ENTITY = "aneeshmuppidi19"

PROJECTS = {
    "tetris": {
        "project": "tetris_rt_kt_cross_eval",
        "description": "Tetris RT_KT — k=1 model evaluated at k=1",
        # Keys to try for sim count in config
        "sim_config_keys": [
            "num_simulations", "sim_count", "mcts_sims", "simulations",
            "n_simulations", "num_sims", "sims", "eval_sims",
            "simulation_count", "num_mcts_sims", "k", "K",
            "eval_k", "train_k", "delay", "planning_depth",
        ],
        # Keys to try for performance
        "perf_metric_keys": [
            "episode_return", "mean_episode_return", "avg_return",
            "score", "mean_score", "eval_return", "eval/episode_return",
            "return", "episode_score", "lines_cleared", "mean_lines_cleared",
        ],
        # Keys to try for inference time / latency
        "latency_metric_keys": [
            "inference_time", "step_time", "latency", "fps",
            "mean_step_time", "mean_inference_time", "wall_time_per_step",
            "time_per_step", "planning_time", "mcts_time",
            "inference_latency", "step_latency", "time_per_action",
        ],
        # For cross-eval projects: filter to k=1 train, k=1 eval
        "filter_hint": "k=1 train, k=1 eval — vary sim count",
    },
    "pacman": {
        "project": "pacman_kt_cross_eval_mature",
        "description": "Pac-Man — k=1 model evaluated at k=1",
        "sim_config_keys": [
            "num_simulations", "sim_count", "mcts_sims", "simulations",
            "n_simulations", "num_sims", "sims", "eval_sims",
            "simulation_count", "num_mcts_sims", "k", "K",
            "eval_k", "train_k", "delay", "planning_depth",
        ],
        "perf_metric_keys": [
            "episode_return", "mean_episode_return", "avg_return",
            "score", "mean_score", "eval_return", "eval/episode_return",
            "return", "episode_score",
        ],
        "latency_metric_keys": [
            "inference_time", "step_time", "latency", "fps",
            "mean_step_time", "mean_inference_time", "wall_time_per_step",
            "time_per_step", "planning_time", "mcts_time",
            "inference_latency", "step_latency", "time_per_action",
        ],
        "filter_hint": "k=1 train, k=1 eval — vary sim count",
    },
    "sokoban": {
        "project": "sokoban_gumbel_az_eval",
        "description": "Sokoban Gumbel AlphaZero",
        "sim_config_keys": [
            "num_simulations", "sim_count", "mcts_sims", "simulations",
            "n_simulations", "num_sims", "sims",
            "simulation_count", "num_mcts_sims",
            "gumbel_sims", "n_gumbel_sims",
        ],
        "perf_metric_keys": [
            "episode_return", "mean_episode_return", "avg_return",
            "solved", "solve_rate", "solved_rate", "success_rate",
            "prop_correct_boxes", "prop_box_correct_boxes",
            "boxes_solved", "score", "mean_score",
            "eval_return", "return",
        ],
        "latency_metric_keys": [
            "inference_time", "step_time", "latency", "fps",
            "mean_step_time", "mean_inference_time", "wall_time_per_step",
            "time_per_step", "planning_time", "mcts_time",
            "inference_latency", "step_latency", "time_per_action",
        ],
        "filter_hint": "vary sim count",
    },
    "hex": {
        "project": "hex_inference_tournament",
        "description": "Speed Hex — inference tournament",
        "sim_config_keys": [
            "num_simulations", "sim_count", "mcts_sims", "simulations",
            "n_simulations", "num_sims", "sims",
            "simulation_count", "num_mcts_sims",
        ],
        "perf_metric_keys": [
            "win_rate", "global_win_rate", "winrate",
            "win_ratio", "wins", "elo", "score",
            "episode_return", "mean_episode_return",
            "p1_win_rate", "p2_win_rate",
        ],
        "latency_metric_keys": [
            "inference_time", "step_time", "latency", "fps",
            "mean_step_time", "mean_inference_time", "wall_time_per_step",
            "time_per_step", "planning_time", "mcts_time",
            "inference_latency", "step_latency", "time_per_action",
            "move_time", "mean_move_time",
        ],
        "filter_hint": "vary sim count, find win rate + inference time",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_keys(d, candidates):
    """Return (key, value) pairs from d whose key matches any candidate (case-insensitive)."""
    found = {}
    lower_candidates = {c.lower(): c for c in candidates}
    for k, v in d.items():
        if k.lower() in lower_candidates:
            found[k] = v
    return found


def _all_numeric_keys(d):
    """Return all keys whose value is numeric (int or float)."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def _summarize_run(run, cfg):
    summary = {}

    # --- config
    config = dict(run.config)
    summary["config_sim"] = _find_keys(config, cfg["sim_config_keys"])
    summary["config_all_numeric"] = _all_numeric_keys(config)

    # --- summary metrics
    metrics = dict(run.summary)
    summary["metric_perf"] = _find_keys(metrics, cfg["perf_metric_keys"])
    summary["metric_latency"] = _find_keys(metrics, cfg["latency_metric_keys"])
    summary["metric_all_numeric"] = _all_numeric_keys(metrics)

    summary["run_name"] = run.name
    summary["run_id"] = run.id
    summary["run_state"] = run.state

    return summary


def _print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_dict(d, indent=4, max_items=30):
    items = list(d.items())[:max_items]
    for k, v in items:
        if isinstance(v, float):
            print(f"{' ' * indent}{k}: {v:.5g}")
        else:
            print(f"{' ' * indent}{k}: {v}")
    if len(d) > max_items:
        print(f"{' ' * indent}... ({len(d) - max_items} more keys)")


# ─────────────────────────────────────────────────────────────────────────────
# Per-project probe
# ─────────────────────────────────────────────────────────────────────────────

def probe_project(env_key, cfg):
    _print_section(f"{env_key.upper()} — {cfg['project']} ({cfg['description']})")
    print(f"  Filter hint: {cfg['filter_hint']}")

    api = wandb.Api()
    try:
        runs = list(api.runs(f"{ENTITY}/{cfg['project']}"))
    except Exception as e:
        print(f"  ERROR fetching runs: {e}")
        return

    print(f"  Total runs found: {len(runs)}")
    if not runs:
        return

    # Summarize all runs
    all_summaries = [_run_brief(run, cfg) for run in runs]

    # Print first few runs in full detail
    print(f"\n  ── First 3 runs (full detail) ──")
    for i, run in enumerate(runs[:3]):
        s = _summarize_run(run, cfg)
        print(f"\n  Run [{i}]: {s['run_name']} ({s['run_id']})  state={s['run_state']}")

        print(f"    Sim-count config keys found:")
        if s["config_sim"]:
            _print_dict(s["config_sim"], indent=6)
        else:
            print("      (none matched — showing all numeric config keys)")
            _print_dict(s["config_all_numeric"], indent=6, max_items=20)

        print(f"    Performance metric keys found:")
        if s["metric_perf"]:
            _print_dict(s["metric_perf"], indent=6)
        else:
            print("      (none matched)")

        print(f"    Latency metric keys found:")
        if s["metric_latency"]:
            _print_dict(s["metric_latency"], indent=6)
        else:
            print("      (none matched — showing all numeric summary keys)")
            _print_dict(s["metric_all_numeric"], indent=6, max_items=20)

    # Aggregate: which sim-related config keys appear across all runs?
    print(f"\n  ── Aggregate across all {len(runs)} runs ──")

    all_config_keys = defaultdict(set)
    all_summary_keys = defaultdict(set)
    for run in runs:
        for k in run.config:
            all_config_keys[k].add(type(run.config[k]).__name__)
        for k in run.summary.keys():
            all_summary_keys[k].add(type(run.summary[k]).__name__)

    # Print config keys that look sim-related
    print(f"\n  Config keys containing 'sim', 'mcts', 'sims', 'k', 'delay', 'plan':")
    keywords = ["sim", "mcts", "sims", "delay", "plan", "budget", "depth"]
    found_any = False
    for k, types in sorted(all_config_keys.items()):
        if any(kw in k.lower() for kw in keywords):
            print(f"    {k}  (types seen: {', '.join(types)})")
            found_any = True
    if not found_any:
        print("    (none — printing all config keys)")
        for k, types in sorted(all_config_keys.items()):
            print(f"    {k}  (types: {', '.join(types)})")

    # Print summary keys that look performance/latency related
    print(f"\n  Summary metric keys containing 'return', 'score', 'win', 'solve', 'time', 'fps', 'latency', 'boxes':")
    perf_kws = ["return", "score", "win", "solve", "reward", "time", "fps",
                "latency", "boxes", "correct", "lines", "cleared", "elo"]
    found_any = False
    for k, types in sorted(all_summary_keys.items()):
        if any(kw in k.lower() for kw in perf_kws):
            print(f"    {k}  (types: {', '.join(types)})")
            found_any = True
    if not found_any:
        print("    (none matched — printing all summary keys)")
        for k, types in sorted(all_summary_keys.items()):
            print(f"    {k}  (types: {', '.join(types)})")

    # Try to group by sim count and show performance range
    print(f"\n  ── Attempting to pivot: sim_count → (perf, latency) ──")
    _try_pivot(runs, cfg)


def _run_brief(run, cfg):
    config = dict(run.config)
    metrics = dict(run.summary)
    sim = _find_keys(config, cfg["sim_config_keys"])
    perf = _find_keys(metrics, cfg["perf_metric_keys"])
    lat = _find_keys(metrics, cfg["latency_metric_keys"])
    return {"sim": sim, "perf": perf, "lat": lat, "name": run.name}


def _try_pivot(runs, cfg):
    """Try to build a sim_count vs (perf, latency) table."""
    rows = []
    for run in runs:
        config = dict(run.config)
        metrics = dict(run.summary)

        sim_hits = _find_keys(config, cfg["sim_config_keys"])
        perf_hits = _find_keys(metrics, cfg["perf_metric_keys"])
        lat_hits = _find_keys(metrics, cfg["latency_metric_keys"])

        if not sim_hits and not perf_hits:
            continue

        row = {"run_name": run.name}
        row.update({f"cfg_{k}": v for k, v in sim_hits.items()})
        row.update({f"perf_{k}": v for k, v in perf_hits.items()})
        row.update({f"lat_{k}": v for k, v in lat_hits.items()})
        rows.append(row)

    if not rows:
        print("  Could not build pivot — no runs had matching sim + perf keys.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(max_rows=30, max_cols=10))


# ─────────────────────────────────────────────────────────────────────────────
# History probe: check if metrics are logged per-step (history) vs summary
# ─────────────────────────────────────────────────────────────────────────────

def probe_history_sample(env_key, cfg):
    """Check the history keys of the first run to see what's logged per step."""
    api = wandb.Api()
    try:
        runs = list(api.runs(f"{ENTITY}/{cfg['project']}"))
    except Exception:
        return
    if not runs:
        return

    run = runs[0]
    print(f"\n  ── History keys (first run: {run.name}) ──")
    try:
        history = run.history(samples=5, pandas=False)
        if history:
            sample = history[0]
            print(f"  Sample history row keys: {sorted(sample.keys())}")
        else:
            print("  No history rows found.")
    except Exception as e:
        print(f"  Could not fetch history: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("wandb co-scaling probe")
    print(f"Entity: {ENTITY}")
    print(f"Goal: find (sim_count, perf_metric, latency_metric) for fig:scaling")

    for env_key, cfg in PROJECTS.items():
        probe_project(env_key, cfg)
        probe_history_sample(env_key, cfg)

    _print_section("SUMMARY")
    print("""
  After running this script, look for:
    1. A config key that represents simulation count (e.g. num_simulations).
    2. A summary metric for planning quality (return / win rate / solve rate).
    3. A summary metric for inference latency (step_time / inference_time / fps).

  If latency is missing from summary, check history keys — it may be logged
  per step and needs to be aggregated (mean).

  The figure needs:
    x-axis : simulation count (one point per run or per sim-count group)
    y-axis₁: planning quality (normalized or raw)
    y-axis₂: inference latency (ms or s per step, or 1/fps)
    one panel per environment (4 panels total)
""")


if __name__ == "__main__":
    main()
