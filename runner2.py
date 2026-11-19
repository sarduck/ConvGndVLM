#!/usr/bin/env python3
"""
Experiment 2 (E2): History Ablation — Memory vs. Structural Failure Test
=========================================================================
CS 6120 Research Question RQ2:
  "Does giving the model an explicit summary of what was established in the
  previous round recover cross-round compression? This distinguishes a memory
  failure (fixable with scaffolding) from a structural failure (not fixable)."

Experimental Design
-------------------
  λ              : Fixed at 1.0 (full instruct model — consistent with E1 baseline)
  Summary modes  : no_summary (control, reuses E1 λ=1.0 data),
                   oracle_summary (deterministic ground-truth context injection)
  Games          : 50 games × 3 rounds per condition
  Max turns/round: 6 (same as E1 for direct comparability)

The Core Hypothesis:
  If the oracle summary recovers compression (R3/R1 drops toward ~0.5),
  the failure is a MEMORY failure — the model has latent pact-forming ability
  but lacks the working memory to exploit shared context across rounds.

  If oracle summary has no effect on compression, the failure is STRUCTURAL —
  the model cannot adapt its communication strategy even when explicitly told
  what has already been established.

Metrics Computed (per condition, per game)
------------------------------------------
  1. Task Score           : correct guesses / max (18 per game)
  2. Per-Round Words      : mean words in R1, R2, R3
  3. Compression Ratio    : words_R3 / words_R1  — < 1.0 means compression
  4. Compression Rate     : % of games where words_R3 < words_R1
  5. Per-Round Turns      : mean turns in R1, R2, R3
  6. Word Novelty Rate    : new words in Rk not in R1 / total words in Rk
     (Hua & Artzi 2024 — lower WNR in later rounds = pact formation)
  7. Score Trajectory     : (score_R3 - score_R1) — positive = accuracy improves
  8. Efficiency Index     : task_score / avg_words
  9. Summary Adherence    : fraction of R2+R3 turns containing acknowledgement
     phrases ("as established", "already confirmed", "from last round", etc.)
     This reveals whether the model is *using* the injected summary.

Output: Experiments/output_E_2/
  results_no_summary.json
  results_oracle_summary.json
  metrics_E2.json          (all metrics in one structured file)
  summary_E2.txt           (human-readable comparison table)

Usage
-----
  .venv/bin/python runner2.py
  .venv/bin/python runner2.py --conditions no_summary,oracle_summary
  .venv/bin/python runner2.py --num_games 10 --quiet          # quick test
  .venv/bin/python runner2.py --resume
"""

import os
import gc
import json
import time
import argparse
from collections import Counter
from tqdm import tqdm

import torch
from config import get_default_device
from data_loader import load_photobook_games
from model_utils import get_vlm
from game_engine import play_game


# ---------------------------------------------------------------------------
# Metric helpers (mirrored from runner3.py for self-containment)
# ---------------------------------------------------------------------------

# Phrases that signal the model is explicitly referencing established context.
# Used for the "summary adherence" metric (E2-specific).
ACKNOWLEDGEMENT_PHRASES = [
    "as established",
    "as we established",
    "already confirmed",
    "already identified",
    "from last round",
    "from the previous round",
    "we agreed",
    "we determined",
    "as confirmed",
    "as noted",
    "as before",
    "already know",
    "previously",
]


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def compute_wnr(msgs_R1: list[str], msgs_Rk: list[str]) -> float:
    """
    Word Novelty Rate: fraction of Rk's vocabulary not seen in R1.
    Lower WNR in later rounds = more lexical reuse = pact formation.
    Returns NaN if Rk is empty.
    """
    vocab_R1 = set(w.lower() for m in msgs_R1 for w in m.split())
    vocab_Rk = set(w.lower() for m in msgs_Rk for w in m.split())
    if not vocab_Rk:
        return float("nan")
    return len(vocab_Rk - vocab_R1) / len(vocab_Rk)


def compute_summary_adherence(rounds: list[dict]) -> float:
    """
    Fraction of turns in rounds 2+ whose message contains at least one
    acknowledgement phrase from ACKNOWLEDGEMENT_PHRASES.

    A high adherence score means the model is explicitly referencing the
    injected summary (or its own prior-round memory).  This is most
    meaningful for the oracle condition — if adherence is high there,
    the model is reading and using the summary.
    """
    total_turns = 0
    ack_turns = 0
    for r in rounds:
        if r["round_nr"] < 2:
            continue
        for turn in r["transcript"]:
            msg = turn["output"].get("message", "").lower()
            total_turns += 1
            if any(phrase in msg for phrase in ACKNOWLEDGEMENT_PHRASES):
                ack_turns += 1
    return ack_turns / total_turns if total_turns > 0 else 0.0


def compute_game_metrics(game_result: dict) -> dict:
    """Compute all E2 metrics for a single game."""
    rounds = game_result["rounds"]
    total_score = game_result["total_score"]
    max_score = game_result["max_score"]

    per_round = {}
    messages_by_round = {}

    for r in rounds:
        rn = r["round_nr"]
        transcript = r["transcript"]
        msgs = [t["output"].get("message", "") for t in transcript]
        word_counts = [count_words(m) for m in msgs]
        turn_count = len(transcript)

        per_round[rn] = {
            "score": r["round_score"],
            "turn_count": turn_count,
            "total_words": sum(word_counts),
            "avg_words_per_turn": sum(word_counts) / turn_count if turn_count > 0 else 0,
        }
        messages_by_round[rn] = msgs

    r1_words = per_round.get(1, {}).get("total_words", 1) or 1
    r3_words = per_round.get(3, {}).get("total_words", 0)
    compression_ratio = r3_words / r1_words

    wnr_r2 = compute_wnr(messages_by_round.get(1, []), messages_by_round.get(2, []))
    wnr_r3 = compute_wnr(messages_by_round.get(1, []), messages_by_round.get(3, []))

    score_r1 = per_round.get(1, {}).get("score", 0)
    score_r3 = per_round.get(3, {}).get("score", 0)
    score_trajectory = score_r3 - score_r1

    total_words = sum(pr["total_words"] for pr in per_round.values())
    efficiency_index = total_score / total_words if total_words > 0 else 0.0
    summary_adherence = compute_summary_adherence(rounds)

    return {
        "game_id": game_result["game_id"],
        "total_score": total_score,
        "max_score": max_score,
        "total_words": total_words,
        "total_turns": sum(pr["turn_count"] for pr in per_round.values()),
        "compression_ratio": compression_ratio,
        "compression_emerged": int(compression_ratio < 1.0),
        "wnr_r2": wnr_r2,
        "wnr_r3": wnr_r3,
        "score_trajectory": score_trajectory,
        "efficiency_index": efficiency_index,
        "summary_adherence": summary_adherence,
        "per_round": per_round,
    }


def aggregate_metrics(game_metrics_list: list[dict], condition: str) -> dict:
    """Aggregate per-game metrics into a summary dict for one condition."""
    n = len(game_metrics_list)
    if n == 0:
        return {"condition": condition, "n_games": 0}

    def safe_avg(key):
        vals = [m[key] for m in game_metrics_list if m[key] == m[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    def round_avg(rn, key):
        vals = [m["per_round"][rn][key] for m in game_metrics_list if rn in m["per_round"]]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "condition": condition,
        "n_games": n,
        "avg_score": safe_avg("total_score"),
        "avg_words": safe_avg("total_words"),
        "avg_turns": safe_avg("total_turns"),
        "avg_compression_ratio": safe_avg("compression_ratio"),
        "compression_emergence_rate": safe_avg("compression_emerged"),
        "avg_wnr_r2": safe_avg("wnr_r2"),
        "avg_wnr_r3": safe_avg("wnr_r3"),
        "avg_score_trajectory": safe_avg("score_trajectory"),
        "avg_efficiency_index": safe_avg("efficiency_index"),
        "avg_summary_adherence": safe_avg("summary_adherence"),
        "per_round": {
            rn: {
                "avg_score": round_avg(rn, "score"),
                "avg_words": round_avg(rn, "total_words"),
                "avg_turns": round_avg(rn, "turn_count"),
                "avg_words_per_turn": round_avg(rn, "avg_words_per_turn"),
            }
            for rn in [1, 2, 3]
        },
    }


def print_summary_table(all_aggregates: list[dict]) -> str:
    """Format a human-readable comparison table for E2."""
    sep = "=" * 110
    lines = [
        sep,
        " E2 HISTORY ABLATION EXPERIMENT — RESULTS SUMMARY",
        " Memory vs. Structural Failure: Does oracle context injection recover compression?",
        sep,
        f"{'Condition':<20} {'Score/18':<10} {'Compress':<10} {'C-Rate%':<10} "
        f"{'WNR-R3':<10} {'ScoreTraj':<12} {'Adherence%':<12} {'Efficiency':<10}",
        f"{'':20} {'(avg)':<10} {'(R3/R1)':<10} {'(R3<R1?)':<10} "
        f"{'(novel wds)':<10} {'(R3-R1 sc)':<12} {'(ack phrase)':<12} {'(sc/wds)':<10}",
        "-" * 110,
    ]

    for agg in all_aggregates:
        c = agg["condition"]
        lines.append(
            f"{c:<20} "
            f"{agg['avg_score']:<10.2f} "
            f"{agg['avg_compression_ratio']:<10.3f} "
            f"{agg['compression_emergence_rate']*100:<10.1f} "
            f"{agg['avg_wnr_r3']:<10.3f} "
            f"{agg['avg_score_trajectory']:+<12.2f} "
            f"{agg['avg_summary_adherence']*100:<12.1f} "
            f"{agg['avg_efficiency_index']:<10.4f}"
        )

    lines += [
        sep,
        "",
        "KEY:",
        "  Compress (R3/R1) < 1.0  → fewer words used in R3 than R1 (compression signal)",
        "  Human baseline R3/R1    ≈ 0.47  (53% compression)",
        "  E1 λ=1.0 baseline       ≈ 0.95  (5% compression — structural failure signal)",
        "  Adherence%              = % of R2+R3 turns containing acknowledgement phrases",
        "  High adherence + no compression → model reads summary but cannot adapt strategy",
        "  High adherence + compression    → memory scaffolding recovers pact behaviour",
        "",
        "Per-Round Breakdown:",
        f"{'Condition':<20} {'R1 Score':<10} {'R2 Score':<10} {'R3 Score':<10} "
        f"{'R1 Words':<10} {'R2 Words':<10} {'R3 Words':<10} "
        f"{'R1 Turns':<10} {'R2 Turns':<10} {'R3 Turns':<10}",
        "-" * 110,
    ]

    for agg in all_aggregates:
        pr = agg["per_round"]
        lines.append(
            f"{agg['condition']:<20} "
            f"{pr[1]['avg_score']:<10.2f} "
            f"{pr[2]['avg_score']:<10.2f} "
            f"{pr[3]['avg_score']:<10.2f} "
            f"{pr[1]['avg_words']:<10.1f} "
            f"{pr[2]['avg_words']:<10.1f} "
            f"{pr[3]['avg_words']:<10.1f} "
            f"{pr[1]['avg_turns']:<10.1f} "
            f"{pr[2]['avg_turns']:<10.1f} "
            f"{pr[3]['avg_turns']:<10.1f}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="E2: History Ablation Experiment Runner")

    parser.add_argument("--backend", type=str, default="huggingface",
                        choices=["huggingface", "ollama"])
    parser.add_argument("--model_family", type=str, default="OpenGVLab/InternVL3_5-4B",
                        help="Instruct model (λ=1.0 fixed for E2)")
    parser.add_argument("--base_model", type=str, default="OpenGVLab/InternVL3_5-4B-Pretrained",
                        help="Base model (not used in E2, λ=1.0 only)")

    default_dev = get_default_device()
    parser.add_argument("--device", type=str, default=default_dev)
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument(
        "--conditions", type=str, default="no_summary,oracle_summary",
        help=(
            "Comma-separated summary conditions to run. "
            "Options: no_summary, oracle_summary  (default: both)"
        ),
    )
    parser.add_argument("--num_games", type=int, default=50)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_turns_per_round", type=int, default=6,
                        help="Turns per round (keep at 6 for comparability with E1).")

    parser.add_argument("--output_dir", type=str, default="Experiments/output_E_2")
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-turn dialogue output.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip conditions that already have complete output files.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",")]
    valid = {"no_summary", "oracle_summary"}
    for c in conditions:
        if c not in valid:
            raise ValueError(f"Unknown condition '{c}'. Valid options: {sorted(valid)}")

    verbose = not args.quiet
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(" PHOTOBOOK E2 — HISTORY ABLATION EXPERIMENT")
    print("=" * 60)
    print(f"Model (λ=1.0):  {args.model_family}")
    print(f"Device:         {args.device}")
    print(f"Conditions:     {conditions}")
    print(f"Games/Cond:     {args.num_games}")
    print(f"Rounds/Game:    {args.max_rounds} (Max score: {args.max_rounds * 6})")
    print(f"Turns/Round:    {args.max_turns_per_round}  (same as E1)")
    print(f"Output Dir:     {args.output_dir}")
    print("=" * 60)

    # Load dataset once — same 50 games for every condition
    print(f"\nLoading {args.num_games} PhotoBook games from '{args.data_dir}'...")
    games = load_photobook_games(args.data_dir, num_games=args.num_games,
                                 max_rounds=args.max_rounds)
    print(f"✓ Loaded {len(games)} valid games.\n")

    image_base_dir = os.path.join(args.data_dir, "images")

    # Minimal args object for get_vlm (always λ=1.0)
    class ModelArgs:
        backend = args.backend
        model_family = args.model_family
        base_model = args.base_model
        device = args.device
        load_in_4bit = args.load_in_4bit

    model_args = ModelArgs()
    all_aggregates = []

    cond_pbar = tqdm(conditions, desc="Conditions", position=0, unit="cond")

    for condition in cond_pbar:
        # Map condition name to summary_mode accepted by play_game
        summary_mode = "oracle" if condition == "oracle_summary" else "none"
        output_file = os.path.join(args.output_dir, f"results_{condition}.json")

        # Resume logic
        existing_results = []
        start_idx = 0
        if args.resume and os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    existing_results = json.load(f)
                start_idx = len(existing_results)
                if start_idx >= len(games):
                    tqdm.write(
                        f"\n✓ [{condition}] Already complete "
                        f"({len(existing_results)} games). Skipping."
                    )
                    gm = [compute_game_metrics(r) for r in existing_results]
                    all_aggregates.append(aggregate_metrics(gm, condition))
                    continue
                tqdm.write(f"\n[{condition}] Resuming from game {start_idx + 1}...")
            except Exception as e:
                tqdm.write(f"Warning: Could not read {output_file}: {e}. Starting fresh.")
                existing_results = []
                start_idx = 0

        tqdm.write(f"\n{'='*60}\n>>> Condition: {condition}\n{'='*60}")

        # Load model (λ=1.0 for all conditions)
        t0 = time.time()
        vlm = get_vlm(model_args, lambda_val=1.0)
        tqdm.write(f"Model loaded in {time.time() - t0:.1f}s.")

        results = list(existing_results)
        game_metrics = [compute_game_metrics(r) for r in results]
        running_scores = [r["total_score"] for r in results]

        games_to_play = games[start_idx:]
        game_pbar = tqdm(
            enumerate(games_to_play, start=start_idx + 1),
            total=len(games),
            initial=start_idx,
            desc=condition,
            position=1,
            leave=False,
            unit="game",
        )

        for game_num, game_config in game_pbar:
            gid = game_config["game_id"]
            if verbose:
                tqdm.write(f"\n--- [{condition}] Game {game_num}/{len(games)} (ID: {gid}) ---")

            try:
                game_result = play_game(
                    game_config=game_config,
                    vlm_A=vlm,
                    vlm_B=vlm,
                    image_base_dir=image_base_dir,
                    max_rounds=args.max_rounds,
                    max_turns_per_round=args.max_turns_per_round,
                    summary_mode=summary_mode,
                    verbose=verbose,
                )
                results.append(game_result)
                gm = compute_game_metrics(game_result)
                game_metrics.append(gm)
                running_scores.append(game_result["total_score"])

                # Autosave after every game
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                tqdm.write(f"Error during game {gid}: {e}")

            cur_avg = sum(running_scores) / len(running_scores) if running_scores else 0
            game_pbar.set_postfix({
                "score": f"{running_scores[-1] if running_scores else 0}/18",
                "avg": f"{cur_avg:.2f}",
            })

            # MPS memory management
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        game_pbar.close()

        agg = aggregate_metrics(game_metrics, condition)
        all_aggregates.append(agg)
        tqdm.write(
            f"\n✓ {condition} done: {len(results)} games | "
            f"Score={agg['avg_score']:.2f}/18 | Words={agg['avg_words']:.0f} | "
            f"CompressRatio={agg['avg_compression_ratio']:.3f} | "
            f"CompressRate={agg['compression_emergence_rate']*100:.1f}% | "
            f"SummaryAdherence={agg['avg_summary_adherence']*100:.1f}%"
        )

        # Free model before next condition
        del vlm
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    cond_pbar.close()

    # Save structured metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics_E2.json")
    with open(metrics_path, "w") as f:
        json.dump(all_aggregates, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Print and save summary table
    summary = print_summary_table(all_aggregates)
    print("\n" + summary)
    summary_path = os.path.join(args.output_dir, "summary_E2.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary table saved to: {summary_path}")

    print("\n" + "=" * 60)
    print(" INTERPRETATION GUIDE")
    print("=" * 60)
    print("  oracle R3/R1 << no_summary R3/R1  →  MEMORY FAILURE")
    print("    The model can form pacts when given external memory.")
    print("    Fix: retrieval-augmented prompting, explicit memory modules.")
    print()
    print("  oracle R3/R1 ≈  no_summary R3/R1  →  STRUCTURAL FAILURE")
    print("    Even a perfect prior-round summary doesn't change strategy.")
    print("    Fix: needs training on collaborative dialogue corpora,")
    print("    architecturally grounded common-ground tracking.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
