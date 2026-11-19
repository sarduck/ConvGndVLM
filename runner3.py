#!/usr/bin/env python3
"""
Experiment 3 (E3): Token Budget — Costless-Token Hypothesis Test
=================================================================
CS 6120 Research Question RQ3:
  "If we impose a hard per-turn word budget, does human-like compression
  emerge? Tests the costless-token hypothesis. Distinguishes a structural
  failure from a representational one."

Experimental Design
-------------------
  λ         : Fixed at 1.0 (full instruct model — only it can follow instructions)
  Word budgets tested : unlimited (control = E1 baseline), 75, 50, 25 words / turn
  Turn cap  : Raised to 15 per round (floats freely under tight budgets)
  Games     : 50 games × 3 rounds = 150 rounds per budget condition
  Total     : 4 budget conditions × 150 rounds = 600 rounds

The Core Hypothesis (from Imai et al. 2025, §Limitations):
  Human verbosity drops 50%+ across rounds because cognitive/effort cost
  rises with repeated description. Models face ZERO cost per token, so they
  re-describe verbosely every round. If imposing an artificial budget
  forces efficiency, compression should emerge — proving the failure is
  *representational* (the model CAN compress, it just doesn't have to).
  If compression STILL doesn't emerge under budget constraints, it is a
  *structural* failure (the model lacks the mechanism for conceptual pacts).

Metrics Computed (per budget condition, per game)
-------------------------------------------------
  1. Task Score          : correct guesses / max (18 per game)
  2. Per-Round Words     : mean words in R1, R2, R3
  3. Compression Ratio   : words_R3 / words_R1 — < 1.0 means compression
  4. Compression Rate    : % of games where words_R3 < words_R1
  5. Per-Round Turns     : mean turns in R1, R2, R3 (floats with budget)
  6. Budget Adherence    : % of turns where len(message.split()) ≤ budget
  7. Word Novelty Rate   : new words in Rk not seen in R1 / total words in Rk
     (from Hua & Artzi 2024 — lower WNR in later rounds = pact formation)
  8. Score Trajectory    : (score_R3 - score_R1) — positive = accuracy improves
  9. Efficiency Index    : task_score / avg_words — higher is better

Output: Experiments/output_E_3/
  results_budget_unlimited.json
  results_budget_75.json
  results_budget_50.json
  results_budget_25.json
  metrics_E3.json         (all metrics in one structured file)
  summary_E3.txt          (human-readable summary table)

Usage
-----
  .venv/bin/python runner3.py
  .venv/bin/python runner3.py --budgets unlimited,75,50,25 --num_games 50
  .venv/bin/python runner3.py --quiet --resume
"""

import os
import gc
import sys
import json
import time
import argparse
from collections import Counter
from tqdm import tqdm

import torch
from config import get_default_device
from data_loader import load_photobook_games
from model_utils import get_vlm
from game_engine import PROMPT_ENGINEERED, play_game


# ---------------------------------------------------------------------------
# Budget-augmented Prompt
# ---------------------------------------------------------------------------

BUDGET_PROMPT_TEMPLATE = """\
Goal: find which images are COMMON between you and your partner's list. \
You are Player {{player}}. Refer to images only as "Image 1", "Image 2", or "Image 3" \
– never mention file names or IDs. Important: the number (Image 1/2/3) is local to you. \
Your Image 1 could be your partner's Image 3. Decide only from what you SEE, not from the number.

Your images:
• Image 1
• Image 2
• Image 3

WORD LIMIT — STRICTLY ENFORCED: Your "message" field must contain AT MOST {budget} words. \
Count carefully before you send. Use only the single most distinguishing visual detail per turn. \
Shorter is better. If a previous description established a point, do not repeat it — build on it.

Strategy:
– Name only the key distinguishing feature of one image (colour, action, object, composition).
– Ask one sharp yes/no question about your partner's image.
– Do not re-describe what you already said in a prior turn.

When – and only when – you are ready to identify all three images, include "guesses".

Output format – every turn must be a single valid JSON object with:
• "message" – your utterance as plain text (≤ {budget} words)
• "reference" – "Image 1", "Image 2", "Image 3", or null
• "guesses" – null while discussing, or a list like ["C", "D", "C"]
"""

UNLIMITED_PROMPT = PROMPT_ENGINEERED  # unchanged from E1


def build_prompt(budget):
    """Return system prompt template for a given numeric budget (or 'unlimited')."""
    if budget == "unlimited":
        return UNLIMITED_PROMPT
    return BUDGET_PROMPT_TEMPLATE.replace("{budget}", str(budget))


# ---------------------------------------------------------------------------
# E3-specific Metrics
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def compute_wnr(round_messages_R1: list[str], round_messages_Rk: list[str]) -> float:
    """
    Word Novelty Rate for round k relative to round 1.
    Counts token types (unique words) new in Rk vs vocabulary seen in R1.
    WNR = |vocab(Rk) - vocab(R1)| / |vocab(Rk)|
    Lower WNR in later rounds = more lexical reuse = pact formation.
    Returns NaN if Rk is empty.
    """
    vocab_R1 = set(w.lower() for msg in round_messages_R1 for w in msg.split())
    vocab_Rk = set(w.lower() for msg in round_messages_Rk for w in msg.split())
    if not vocab_Rk:
        return float("nan")
    novel = vocab_Rk - vocab_R1
    return len(novel) / len(vocab_Rk)


def compute_game_metrics(game_result: dict, budget) -> dict:
    """
    Compute all E3 metrics for a single game result.
    Returns a flat metrics dict suitable for aggregation.
    """
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

        over_budget = 0
        avg_words_per_turn = sum(word_counts) / turn_count if turn_count > 0 else 0

        # Budget utilization rate: how much of the allowed budget is actually used?
        # This is the KEY signal for distinguishing structural vs representational failure.
        # If utilization drops R1 → R3, the model is forming pacts (representational success).
        # If utilization stays flat at ~1.0 every round, the model maxes the budget every time
        # regardless of shared context — structural failure.
        if budget == "unlimited":
            # For unlimited, use the max turn's word count as the implicit "budget"
            utilization = float("nan")
        else:
            utilization = avg_words_per_turn / budget if budget > 0 else float("nan")

        if budget != "unlimited":
            over_budget = sum(1 for wc in word_counts if wc > budget)

        per_round[rn] = {
            "score": r["round_score"],
            "turn_count": turn_count,
            "total_words": sum(word_counts),
            "avg_words_per_turn": avg_words_per_turn,
            "budget_utilization": utilization,  # avg_words_per_turn / budget; drops if pacts form
            "over_budget_turns": over_budget,
        }
        messages_by_round[rn] = msgs

    # Compression ratio: R3 words / R1 words  (< 1.0 = compression emerged)
    r1_words = per_round.get(1, {}).get("total_words", 1) or 1
    r3_words = per_round.get(3, {}).get("total_words", 0)
    compression_ratio = r3_words / r1_words

    # Budget utilization drop: R3_utilization - R1_utilization
    # Negative = pact formed (using less of the budget in later rounds)
    # Near zero or positive = no pact (maxing out budget every round)
    r1_util = per_round.get(1, {}).get("budget_utilization", float("nan"))
    r3_util = per_round.get(3, {}).get("budget_utilization", float("nan"))
    utilization_drop = (r3_util - r1_util) if (r1_util == r1_util and r3_util == r3_util) else float("nan")
    # utilization_drop < 0 → model uses proportionally LESS of its budget in R3 → pact formed

    # Word Novelty Rate: R2 vs R1, R3 vs R1
    wnr_r2 = compute_wnr(messages_by_round.get(1, []), messages_by_round.get(2, []))
    wnr_r3 = compute_wnr(messages_by_round.get(1, []), messages_by_round.get(3, []))

    # Score trajectory
    score_r1 = per_round.get(1, {}).get("score", 0)
    score_r3 = per_round.get(3, {}).get("score", 0)
    score_trajectory = score_r3 - score_r1

    # Total budget adherence
    total_turns = sum(pr["turn_count"] for pr in per_round.values())
    total_over = sum(pr["over_budget_turns"] for pr in per_round.values())
    adherence = 1.0 - (total_over / total_turns) if total_turns > 0 else 1.0

    # Efficiency index: score / total_words (higher = doing more with less)
    total_words = sum(pr["total_words"] for pr in per_round.values())
    efficiency_index = total_score / total_words if total_words > 0 else 0.0

    return {
        "game_id": game_result["game_id"],
        "total_score": total_score,
        "max_score": max_score,
        "total_words": total_words,
        "total_turns": total_turns,
        "compression_ratio": compression_ratio,
        "compression_emerged": int(compression_ratio < 1.0),
        "utilization_drop": utilization_drop,         # KEY: R3_util - R1_util; negative = pact formed
        "wnr_r2": wnr_r2,
        "wnr_r3": wnr_r3,
        "score_trajectory": score_trajectory,
        "budget_adherence": adherence,
        "efficiency_index": efficiency_index,
        "per_round": per_round,
    }


def aggregate_metrics(game_metrics_list: list[dict], budget) -> dict:
    """Aggregate per-game metrics into a single summary dict for a budget condition."""
    n = len(game_metrics_list)
    if n == 0:
        return {}

    def safe_avg(key):
        vals = [m[key] for m in game_metrics_list if m[key] == m[key]]  # exclude NaN
        return sum(vals) / len(vals) if vals else float("nan")

    def round_avg(rn, key):
        vals = [m["per_round"][rn][key] for m in game_metrics_list if rn in m["per_round"]]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "budget": budget,
        "n_games": n,
        "avg_score": safe_avg("total_score"),
        "avg_words": safe_avg("total_words"),
        "avg_turns": safe_avg("total_turns"),
        "avg_compression_ratio": safe_avg("compression_ratio"),
        "compression_emergence_rate": safe_avg("compression_emerged"),
        "avg_utilization_drop": safe_avg("utilization_drop"),   # negative = pact formation signal
        "avg_wnr_r2": safe_avg("wnr_r2"),
        "avg_wnr_r3": safe_avg("wnr_r3"),
        "avg_score_trajectory": safe_avg("score_trajectory"),
        "avg_budget_adherence": safe_avg("budget_adherence"),
        "avg_efficiency_index": safe_avg("efficiency_index"),
        "per_round": {
            rn: {
                "avg_score": round_avg(rn, "score"),
                "avg_words": round_avg(rn, "total_words"),
                "avg_turns": round_avg(rn, "turn_count"),
                "avg_words_per_turn": round_avg(rn, "avg_words_per_turn"),
                "avg_budget_utilization": round_avg(rn, "budget_utilization"),
            }
            for rn in [1, 2, 3]
        },
    }


def print_summary_table(all_aggregates: list[dict]) -> str:
    """Format a human-readable summary table, returned as a string."""
    lines = []
    sep = "=" * 100
    lines.append(sep)
    lines.append(" E3 TOKEN BUDGET EXPERIMENT — RESULTS SUMMARY")
    lines.append(" Costless-Token Hypothesis Test: Does a per-turn word cap induce human-like compression?")
    lines.append(sep)
    lines.append(
        f"{'Budget':<12} {'Score/18':<10} {'Compress':<10} {'C-Rate%':<10} "
        f"{'Util-Drop':<12} {'WNR-R3':<10} {'Adherence%':<12} {'ScoreTraj':<10} {'Efficiency':<10}"
    )
    lines.append(
        f"{'(words/turn)':<12} {'(avg)':<10} {'(R3/R1)':<10} {'(R3<R1?)':<10} "
        f"{'(R3-R1 util)':<12} {'(novel wds)':<10} {'(≤budget)':<12} {'(R3-R1 sc)':<10} {'(sc/wds)':<10}"
    )
    lines.append("-" * 100)
    for agg in all_aggregates:
        b = "unlimited" if agg["budget"] == "unlimited" else str(agg["budget"])
        ud = agg.get("avg_utilization_drop", float("nan"))
        ud_str = f"{ud:+.3f}" if ud == ud else "  N/A "
        lines.append(
            f"{b:<12} "
            f"{agg['avg_score']:<10.2f} "
            f"{agg['avg_compression_ratio']:<10.3f} "
            f"{agg['compression_emergence_rate']*100:<10.1f} "
            f"{ud_str:<12} "
            f"{agg['avg_wnr_r3']:<10.3f} "
            f"{agg['avg_budget_adherence']*100:<12.1f} "
            f"{agg['avg_score_trajectory']:+<10.2f} "
            f"{agg['avg_efficiency_index']:<10.4f}"
        )
    lines.append(sep)
    lines.append("")
    lines.append("KEY: Util-Drop = avg_words_per_turn(R3)/budget - avg_words_per_turn(R1)/budget")
    lines.append("     Negative Util-Drop → model uses proportionally LESS of budget in R3 → pact formed")
    lines.append("     Near-zero Util-Drop → model maxes budget every round → structural failure")
    lines.append("")
    lines.append("Per-Round Breakdown:")
    lines.append(
        f"{'Budget':<12} {'R1 Score':<10} {'R2 Score':<10} {'R3 Score':<10} "
        f"{'R1 Words':<10} {'R2 Words':<10} {'R3 Words':<10} "
        f"{'R1 Turns':<10} {'R2 Turns':<10} {'R3 Turns':<10}"
    )
    lines.append(
        f"{'(wds/turn)':<12} {'(util%)':<10} {'(util%)':<10} {'(util%)':<10} "
        f"{'(total)':<10} {'(total)':<10} {'(total)':<10} "
        f"{'(count)':<10} {'(count)':<10} {'(count)':<10}"
    )
    lines.append("-" * 100)
    for agg in all_aggregates:
        b = "unlimited" if agg["budget"] == "unlimited" else str(agg["budget"])
        pr = agg["per_round"]
        budget_num = agg["budget"]

        def util_str(rn):
            u = pr[rn].get("avg_budget_utilization", float("nan"))
            if u != u or budget_num == "unlimited":
                return "N/A"
            return f"{u*100:.0f}%"

        lines.append(
            f"{b:<12} "
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
# CLI Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="E3: Token Budget Experiment Runner")

    parser.add_argument("--backend", type=str, default="huggingface",
                        choices=["huggingface", "ollama"])
    parser.add_argument("--model_family", type=str, default="OpenGVLab/InternVL3_5-4B",
                        help="Instruct model (λ=1.0 fixed for E3)")
    parser.add_argument("--base_model", type=str, default="OpenGVLab/InternVL3_5-4B-Pretrained",
                        help="Base model (not used in E3, λ=1.0 only)")

    default_dev = get_default_device()
    parser.add_argument("--device", type=str, default=default_dev)
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument("--budgets", type=str, default="unlimited,75,50,25",
                        help="Comma-separated word budgets to test. 'unlimited' = E1 baseline.")
    parser.add_argument("--num_games", type=int, default=50)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_turns_per_round", type=int, default=15,
                        help="Safety cap on turns per round. Floats freely under budget pressure.")

    parser.add_argument("--output_dir", type=str, default="Experiments/output_E_3")
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-turn dialogue output")
    parser.add_argument("--resume", action="store_true",
                        help="Skip budget conditions that already have complete output files")

    return parser.parse_args()


def parse_budgets(raw: str):
    """Parse budget list from CLI string into list of int | 'unlimited'."""
    result = []
    for item in raw.split(","):
        item = item.strip()
        result.append("unlimited" if item == "unlimited" else int(item))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    verbose = not args.quiet

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(" PHOTOBOOK E3 — TOKEN BUDGET EXPERIMENT")
    print("=" * 60)
    print(f"Model (λ=1.0):  {args.model_family}")
    print(f"Device:         {args.device}")
    print(f"Budget levels:  {budgets}")
    print(f"Games/Budget:   {args.num_games}")
    print(f"Rounds/Game:    {args.max_rounds} (Max score: {args.max_rounds * 6})")
    print(f"Max Turns/Round:{args.max_turns_per_round}  (floats freely)")
    print(f"Output Dir:     {args.output_dir}")
    print("=" * 60)

    # Load dataset once (same 50 games for every budget condition)
    print(f"\nLoading {args.num_games} PhotoBook games from '{args.data_dir}'...")
    games = load_photobook_games(args.data_dir, num_games=args.num_games, max_rounds=args.max_rounds)
    print(f"✓ Loaded {len(games)} valid games.\n")

    image_base_dir = os.path.join(args.data_dir, "images")

    # Build a fake args object for get_vlm (always λ=1.0 in E3)
    class ModelArgs:
        backend = args.backend
        model_family = args.model_family
        base_model = args.base_model
        device = args.device
        load_in_4bit = args.load_in_4bit

    model_args = ModelArgs()

    all_aggregates = []

    # Outer progress bar: budget conditions
    budget_pbar = tqdm(budgets, desc="Budget Conditions", position=0, unit="budget")

    for budget in budget_pbar:
        label = "unlimited" if budget == "unlimited" else str(budget)
        output_file = os.path.join(args.output_dir, f"results_budget_{label}.json")

        # Resume check
        existing_results = []
        start_idx = 0
        if args.resume and os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    existing_results = json.load(f)
                start_idx = len(existing_results)
                if start_idx >= len(games):
                    tqdm.write(f"\n✓ [Budget={label}] Already complete ({len(existing_results)} games). Skipping.")
                    game_metrics = [compute_game_metrics(r, budget) for r in existing_results]
                    all_aggregates.append(aggregate_metrics(game_metrics, budget))
                    continue
                tqdm.write(f"\n[Budget={label}] Resuming from game {start_idx + 1}...")
            except Exception as e:
                tqdm.write(f"Warning: Could not read {output_file}: {e}. Starting fresh.")
                existing_results = []
                start_idx = 0

        tqdm.write(f"\n{'='*60}\n>>> Budget = {label} words/turn\n{'='*60}")

        # Load model (λ=1.0 only, same model reused across budget conditions)
        t0 = time.time()
        vlm = get_vlm(model_args, lambda_val=1.0)
        tqdm.write(f"Model loaded in {time.time() - t0:.1f}s.")

        # Build prompt for this budget
        prompt_template = build_prompt(budget)

        results = list(existing_results)
        game_metrics = [compute_game_metrics(r, budget) for r in results]
        running_scores = [r["total_score"] for r in results]

        games_to_play = games[start_idx:]
        game_pbar = tqdm(
            enumerate(games_to_play, start=start_idx + 1),
            total=len(games),
            initial=start_idx,
            desc=f"Budget={label}",
            position=1,
            leave=False,
            unit="game"
        )

        for game_num, game_config in game_pbar:
            gid = game_config["game_id"]
            if verbose:
                tqdm.write(f"\n--- [Budget={label}] Game {game_num}/{len(games)} (ID: {gid}) ---")

            try:
                game_result = play_game(
                    game_config=game_config,
                    vlm_A=vlm,
                    vlm_B=vlm,
                    image_base_dir=image_base_dir,
                    max_rounds=args.max_rounds,
                    prompt_override=prompt_template,
                    max_turns_per_round=args.max_turns_per_round,
                    verbose=verbose,
                )
                results.append(game_result)
                gm = compute_game_metrics(game_result, budget)
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

        # Aggregate metrics for this budget condition
        agg = aggregate_metrics(game_metrics, budget)
        all_aggregates.append(agg)
        tqdm.write(
            f"\n✓ Budget={label} done: {len(results)} games | "
            f"Score={agg['avg_score']:.2f}/18 | Words={agg['avg_words']:.0f} | "
            f"CompressRatio={agg['avg_compression_ratio']:.3f} | "
            f"CompressRate={agg['compression_emergence_rate']*100:.1f}%"
        )

        # Free model before next budget condition
        del vlm
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    budget_pbar.close()

    # Save structured metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics_E3.json")
    with open(metrics_path, "w") as f:
        json.dump(all_aggregates, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Print and save summary table
    summary = print_summary_table(all_aggregates)
    print("\n" + summary)
    summary_path = os.path.join(args.output_dir, "summary_E3.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()

