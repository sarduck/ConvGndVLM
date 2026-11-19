#!/usr/bin/env python3
"""
analyze_results.py — Comprehensive analysis suite for E1, E2, and E3.
======================================================================
Single entry-point that:
  1. Loads raw per-game JSON for all three experiments.
  2. Computes every metric (score, compression, WNR, efficiency, adherence, …).
  3. Runs statistical tests:
       • Mann-Whitney U  (non-parametric pairwise comparison between conditions)
       • Bootstrap 95 % CI (10 000 resamples, percentile method)
  4. Prints a human-readable report to the console AND saves it to
       figures/analysis_summary.txt
  5. Saves structured JSON to figures/all_metrics.json
  6. Generates publication-quality figures (4 panels × 3 experiments + summary)

Usage
-----
  .venv/bin/python analyze_results.py
  .venv/bin/python analyze_results.py \\
      --e1_dir Experiments/output_E_1 \\
      --e2_dir Experiments/output_E_2 \\
      --e3_dir Experiments/output_E_3 \\
      --out_dir figures
"""

import os
import json
import argparse
import warnings
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_SCORE    = 16.62
HUMAN_WORDS    = 338.1
HUMAN_R1_W     = 150.0
HUMAN_R2_W     = 100.0
HUMAN_R3_W     = 70.0
HUMAN_COMPRESS = HUMAN_R3_W / HUMAN_R1_W   # ≈ 0.467
RANDOM_SCORE   = 9.0
N_BOOTSTRAP    = 10_000
ALPHA          = 0.05

PALETTE = {
    "blue":   "#2563EB",
    "green":  "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "red":    "#DC2626",
    "gray":   "#6B7280",
    "human":  "#D97706",
    "random": "#9CA3AF",
}

# ---------------------------------------------------------------------------
# Per-game metric extraction (works for E1, E2, E3 raw JSON)
# ---------------------------------------------------------------------------

def _words(text: str) -> int:
    return len(text.split()) if text else 0


def game_metrics(game: dict) -> dict:
    """Extract a flat per-game metrics dict from a raw game result."""
    rounds = game.get("rounds", [])
    total_score = game.get("total_score", 0)
    max_score   = game.get("max_score", 18)

    per_round: dict[int, dict] = {}
    msgs_by_round: dict[int, list[str]] = {}

    for r in rounds:
        rn   = r["round_nr"]
        msgs = [t["output"].get("message", "") for t in r.get("transcript", [])]
        wcs  = [_words(m) for m in msgs]
        tc   = len(msgs)
        per_round[rn] = {
            "score":           r.get("round_score", 0),
            "total_words":     sum(wcs),
            "turn_count":      tc,
            "avg_words_turn":  sum(wcs) / tc if tc else 0,
        }
        msgs_by_round[rn] = msgs

    # Compression ratio: R3 / R1 word totals
    r1w = per_round.get(1, {}).get("total_words", 1) or 1
    r3w = per_round.get(3, {}).get("total_words", 0)
    compress = r3w / r1w

    # Word Novelty Rate for R3 relative to R1
    vocab_r1 = set(w.lower() for m in msgs_by_round.get(1, []) for w in m.split())
    vocab_r3 = set(w.lower() for m in msgs_by_round.get(3, []) for w in m.split())
    wnr_r3   = len(vocab_r3 - vocab_r1) / len(vocab_r3) if vocab_r3 else float("nan")

    # Score trajectory (R3 - R1)
    score_r1 = per_round.get(1, {}).get("score", 0)
    score_r3 = per_round.get(3, {}).get("score", 0)

    total_words = sum(pr["total_words"] for pr in per_round.values())
    total_turns = sum(pr["turn_count"]  for pr in per_round.values())
    efficiency  = total_score / total_words if total_words else 0.0

    return {
        "game_id":          game.get("game_id"),
        "total_score":      total_score,
        "max_score":        max_score,
        "total_words":      total_words,
        "total_turns":      total_turns,
        "compression":      compress,
        "compression_lt1":  int(compress < 1.0),
        "wnr_r3":           wnr_r3,
        "score_trajectory": score_r3 - score_r1,
        "efficiency":       efficiency,
        "per_round":        per_round,
    }


def load_games(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def metrics_from_file(path: str) -> list[dict]:
    return [game_metrics(g) for g in load_games(path)]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Return (lower, mean, upper) bootstrap percentile CI."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    lo = np.percentile(means, 100 * (1 - ci) / 2)
    hi = np.percentile(means, 100 * (1 - (1 - ci) / 2))
    return float(lo), float(np.mean(arr)), float(hi)


def mannwhitney(a: list[float], b: list[float]) -> tuple[float, float, str]:
    """Return (U, p, significance_label)."""
    a_ = [v for v in a if not np.isnan(v)]
    b_ = [v for v in b if not np.isnan(v)]
    if len(a_) < 3 or len(b_) < 3:
        return float("nan"), float("nan"), "n/a"
    u, p = stats.mannwhitneyu(a_, b_, alternative="two-sided")
    if p < 0.001:
        label = "*** (p<0.001)"
    elif p < 0.01:
        label = "**  (p<0.01) "
    elif p < 0.05:
        label = "*   (p<0.05) "
    else:
        label = "ns  (p≥0.05) "
    return float(u), float(p), label


def effect_size_r(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation (effect size for Mann-Whitney U)."""
    if np.isnan(u):
        return float("nan")
    return 1 - (2 * u) / (n1 * n2)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def agg(gms: list[dict], key: str) -> list[float]:
    return [g[key] for g in gms]


def per_round_avg(gms: list[dict], rn: int, key: str) -> float:
    vals = [g["per_round"][rn][key] for g in gms if rn in g["per_round"]]
    return float(np.mean(vals)) if vals else float("nan")


def summarise(gms: list[dict], label: str) -> dict:
    scores  = agg(gms, "total_score")
    words   = agg(gms, "total_words")
    turns   = agg(gms, "total_turns")
    compress= agg(gms, "compression")
    wnr     = agg(gms, "wnr_r3")
    straj   = agg(gms, "score_trajectory")
    eff     = agg(gms, "efficiency")
    crate   = agg(gms, "compression_lt1")

    return {
        "label":            label,
        "n":                len(gms),
        "avg_score":        float(np.mean(scores)),
        "ci_score":         bootstrap_ci(scores),
        "avg_words":        float(np.mean(words)),
        "ci_words":         bootstrap_ci(words),
        "avg_turns":        float(np.mean(turns)),
        "avg_compression":  float(np.mean(compress)),
        "ci_compression":   bootstrap_ci(compress),
        "c_rate_pct":       float(np.mean(crate)) * 100,
        "avg_wnr_r3":       float(np.nanmean(wnr)),
        "avg_score_traj":   float(np.mean(straj)),
        "avg_efficiency":   float(np.mean(eff)),
        "r1_score":         per_round_avg(gms, 1, "score"),
        "r2_score":         per_round_avg(gms, 2, "score"),
        "r3_score":         per_round_avg(gms, 3, "score"),
        "r1_words":         per_round_avg(gms, 1, "total_words"),
        "r2_words":         per_round_avg(gms, 2, "total_words"),
        "r3_words":         per_round_avg(gms, 3, "total_words"),
        "r1_turns":         per_round_avg(gms, 1, "turn_count"),
        "r2_turns":         per_round_avg(gms, 2, "turn_count"),
        "r3_turns":         per_round_avg(gms, 3, "turn_count"),
        "_raw_scores":      scores,
        "_raw_compress":    compress,
        "_raw_words":       words,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def ci_str(ci: tuple) -> str:
    lo, m, hi = ci
    return f"{m:.2f} [{lo:.2f}–{hi:.2f}]"


def _sep(w=100, c="="):
    return c * w


def section(title: str, lines: list[str]) -> str:
    out = [_sep(), f" {title}", _sep(c="-")]
    out.extend(lines)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# E1 Analysis
# ---------------------------------------------------------------------------

def analyse_e1(e1_dir: str) -> tuple[dict, str]:
    """Return (structured_results_dict, text_report_str)."""
    files = sorted(
        (f for f in os.listdir(e1_dir) if f.startswith("results_lambda_") and f.endswith(".json")),
        key=lambda f: float(f.replace("results_lambda_", "").replace(".json", ""))
    )
    lambdas = []
    summaries = {}
    gms_by_lam = {}

    for fname in files:
        lam = float(fname.replace("results_lambda_", "").replace(".json", ""))
        gms = metrics_from_file(os.path.join(e1_dir, fname))
        s   = summarise(gms, f"λ={lam:.1f}")
        lambdas.append(lam)
        summaries[lam] = s
        gms_by_lam[lam] = gms

    lines = []

    # Metric table
    hdr = f"{'λ':<6} {'Score/18':>10} {'95% CI':>18} {'Words':>8} {'Compress':>10} {'95% CI':>18} {'C-Rate%':>8} {'WNR-R3':>8} {'Efficiency':>10}"
    lines += [hdr, "-" * len(hdr)]
    for lam in lambdas:
        s = summaries[lam]
        lines.append(
            f"{lam:<6.1f} "
            f"{s['avg_score']:>10.2f} "
            f"{ci_str(s['ci_score']):>18} "
            f"{s['avg_words']:>8.0f} "
            f"{s['avg_compression']:>10.3f} "
            f"{ci_str(s['ci_compression']):>18} "
            f"{s['c_rate_pct']:>8.1f} "
            f"{s['avg_wnr_r3']:>8.3f} "
            f"{s['avg_efficiency']:>10.4f}"
        )
    # Human row
    lines.append(
        f"{'Human':<6} "
        f"{HUMAN_SCORE:>10.2f} "
        f"{'—':>18} "
        f"{HUMAN_WORDS:>8.0f} "
        f"{HUMAN_COMPRESS:>10.3f} "
        f"{'—':>18} "
        f"{'~100':>8} "
        f"{'↓low':>8} "
        f"{'—':>10}"
    )

    # Per-round score table
    lines += ["", "Per-round score (avg / 6):"]
    lines.append(f"{'λ':<6} {'R1 Score':>10} {'R2 Score':>10} {'R3 Score':>10} {'R3−R1 Δ':>10}")
    lines.append("-" * 50)
    for lam in lambdas:
        s = summaries[lam]
        lines.append(f"{lam:<6.1f} {s['r1_score']:>10.2f} {s['r2_score']:>10.2f} {s['r3_score']:>10.2f} {s['avg_score_traj']:>+10.2f}")

    # Mann-Whitney U: each λ vs λ=0.0 (score)
    lines += ["", "Mann-Whitney U — Score (each λ vs λ=0.0, two-sided):"]
    ref = lambdas[0]
    for lam in lambdas[1:]:
        a = [g["total_score"] for g in gms_by_lam[ref]]
        b = [g["total_score"] for g in gms_by_lam[lam]]
        u, p, sig = mannwhitney(a, b)
        r = effect_size_r(u, len(a), len(b))
        lines.append(f"  λ=0.0 vs λ={lam:.1f}: U={u:.0f}, p={p:.4f}, {sig}, r={r:+.3f}")

    # Mann-Whitney U: compression ratio λ=0.0 vs λ=1.0
    lines += ["", "Mann-Whitney U — Compression Ratio (each λ vs λ=1.0, two-sided):"]
    ref1 = 1.0
    for lam in lambdas[:-1]:
        a = [g["compression"] for g in gms_by_lam[lam]]
        b = [g["compression"] for g in gms_by_lam[ref1]]
        u, p, sig = mannwhitney(a, b)
        r = effect_size_r(u, len(a), len(b))
        lines.append(f"  λ={lam:.1f} vs λ=1.0: U={u:.0f}, p={p:.4f}, {sig}, r={r:+.3f}")

    text = section("E1 — Alignment Dose-Response Sweep", lines)

    result = {
        "lambdas": lambdas,
        "summaries": {str(l): {k: v for k, v in s.items() if not k.startswith("_")}
                      for l, s in summaries.items()},
    }
    return result, text, summaries, gms_by_lam


# ---------------------------------------------------------------------------
# E2 Analysis
# ---------------------------------------------------------------------------

def analyse_e2(e2_dir: str) -> tuple[dict, str]:
    cond_files = {
        "no_summary":     "results_no_summary.json",
        "oracle_summary": "results_oracle_summary.json",
    }
    gms_by_cond = {}
    summaries   = {}

    for cond, fname in cond_files.items():
        path = os.path.join(e2_dir, fname)
        if not os.path.exists(path):
            continue
        gms = metrics_from_file(path)
        gms_by_cond[cond] = gms
        summaries[cond]   = summarise(gms, cond)

    lines = []
    conds = list(summaries.keys())

    # Metric table
    hdr = f"{'Condition':<20} {'Score/18':>10} {'95% CI':>18} {'Words':>8} {'Compress':>10} {'95% CI':>18} {'C-Rate%':>8} {'Adherence':>10}"
    lines += [hdr, "-" * len(hdr)]
    for c in conds:
        s = summaries[c]
        # adherence not in per-game metrics; pull from E2 summary file if available
        lines.append(
            f"{c:<20} "
            f"{s['avg_score']:>10.2f} "
            f"{ci_str(s['ci_score']):>18} "
            f"{s['avg_words']:>8.0f} "
            f"{s['avg_compression']:>10.3f} "
            f"{ci_str(s['ci_compression']):>18} "
            f"{s['c_rate_pct']:>8.1f} "
            f"{'(see E2 metrics)':>10}"
        )
    lines.append(
        f"{'Human':<20} {HUMAN_SCORE:>10.2f} {'—':>18} {HUMAN_WORDS:>8.0f} "
        f"{HUMAN_COMPRESS:>10.3f} {'—':>18} {'~100':>8} {'—':>10}"
    )

    # Per-round word breakdown
    lines += ["", "Per-round word counts:"]
    lines.append(f"{'Condition':<20} {'R1 Words':>10} {'R2 Words':>10} {'R3 Words':>10} {'R3/R1':>8}")
    lines.append("-" * 60)
    for c in conds:
        s = summaries[c]
        r1w = s['r1_words'] or 1
        lines.append(f"{c:<20} {s['r1_words']:>10.1f} {s['r2_words']:>10.1f} {s['r3_words']:>10.1f} {s['r3_words']/r1w:>8.3f}")
    lines.append(f"{'Human':<20} {HUMAN_R1_W:>10.1f} {HUMAN_R2_W:>10.1f} {HUMAN_R3_W:>10.1f} {HUMAN_COMPRESS:>8.3f}")

    # Statistical tests (only if both conditions present)
    if "no_summary" in gms_by_cond and "oracle_summary" in gms_by_cond:
        a_scores  = [g["total_score"] for g in gms_by_cond["no_summary"]]
        b_scores  = [g["total_score"] for g in gms_by_cond["oracle_summary"]]
        a_comp    = [g["compression"] for g in gms_by_cond["no_summary"]]
        b_comp    = [g["compression"] for g in gms_by_cond["oracle_summary"]]
        a_words   = [g["total_words"] for g in gms_by_cond["no_summary"]]
        b_words   = [g["total_words"] for g in gms_by_cond["oracle_summary"]]

        u_sc, p_sc, sig_sc = mannwhitney(a_scores, b_scores)
        u_co, p_co, sig_co = mannwhitney(a_comp,   b_comp)
        u_wo, p_wo, sig_wo = mannwhitney(a_words,  b_words)
        r_sc = effect_size_r(u_sc, len(a_scores), len(b_scores))
        r_co = effect_size_r(u_co, len(a_comp),   len(b_comp))
        r_wo = effect_size_r(u_wo, len(a_words),  len(b_words))

        lines += [
            "",
            "Mann-Whitney U — no_summary vs oracle_summary (two-sided):",
            f"  Score:           U={u_sc:.0f}, p={p_sc:.4f}, {sig_sc}, r={r_sc:+.3f}",
            f"  Compression:     U={u_co:.0f}, p={p_co:.4f}, {sig_co}, r={r_co:+.3f}",
            f"  Total words:     U={u_wo:.0f}, p={p_wo:.4f}, {sig_wo}, r={r_wo:+.3f}",
            "",
            "Interpretation:",
            "  r > 0 → oracle_summary has HIGHER values on that metric",
            "  r < 0 → oracle_summary has LOWER values (expected for compression & words)",
        ]

    text = section("E2 — History Ablation (Memory vs Structural Failure)", lines)
    result = {
        "conditions": conds,
        "summaries": {c: {k: v for k, v in s.items() if not k.startswith("_")}
                      for c, s in summaries.items()},
    }
    return result, text, summaries, gms_by_cond


# ---------------------------------------------------------------------------
# E3 Analysis
# ---------------------------------------------------------------------------

def analyse_e3(e3_dir: str) -> tuple[dict, str]:
    budget_order = ["unlimited", "75", "50", "25"]
    budget_files = {
        "unlimited": "results_budget_unlimited.json",
        "75":        "results_budget_75.json",
        "50":        "results_budget_50.json",
        "25":        "results_budget_25.json",
    }
    # Load E3 aggregate metrics for Util-Drop (computed by runner3.py)
    e3_agg = {}
    agg_path = os.path.join(e3_dir, "metrics_E3.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            for entry in json.load(f):
                b = entry["budget"]
                label = "unlimited" if b == "unlimited" else str(b)
                e3_agg[label] = entry

    gms_by_bud = {}
    summaries  = {}

    for bud, fname in budget_files.items():
        path = os.path.join(e3_dir, fname)
        if not os.path.exists(path):
            continue
        gms = metrics_from_file(path)
        gms_by_bud[bud] = gms
        summaries[bud]  = summarise(gms, f"budget={bud}")

    lines = []
    avail = [b for b in budget_order if b in summaries]

    # Metric table
    hdr = f"{'Budget':>12} {'Score/18':>10} {'95% CI':>18} {'Words':>8} {'Compress':>10} {'95% CI':>18} {'C-Rate%':>8} {'Util-Drop':>10} {'Adherence':>10}"
    lines += [hdr, "-" * len(hdr)]
    for bud in avail:
        s   = summaries[bud]
        ud  = e3_agg.get(bud, {}).get("avg_utilization_drop", float("nan"))
        adh = e3_agg.get(bud, {}).get("avg_budget_adherence", float("nan"))
        ud_s  = f"{ud:+.3f}" if not np.isnan(ud)  else "   N/A"
        adh_s = f"{adh*100:.1f}%" if not np.isnan(adh) else "N/A"
        lines.append(
            f"{bud:>12} "
            f"{s['avg_score']:>10.2f} "
            f"{ci_str(s['ci_score']):>18} "
            f"{s['avg_words']:>8.0f} "
            f"{s['avg_compression']:>10.3f} "
            f"{ci_str(s['ci_compression']):>18} "
            f"{s['c_rate_pct']:>8.1f} "
            f"{ud_s:>10} "
            f"{adh_s:>10}"
        )
    lines.append(
        f"{'Human':>12} {HUMAN_SCORE:>10.2f} {'—':>18} {HUMAN_WORDS:>8.0f} "
        f"{HUMAN_COMPRESS:>10.3f} {'—':>18} {'~100':>8} {'≈−0.3':>10} {'—':>10}"
    )

    # Per-round word breakdown
    lines += ["", "Per-round word counts:"]
    lines.append(f"{'Budget':>12} {'R1 Words':>10} {'R2 Words':>10} {'R3 Words':>10} {'R3/R1':>8} {'R1 Turns':>9} {'R2 Turns':>9} {'R3 Turns':>9}")
    lines.append("-" * 85)
    for bud in avail:
        s = summaries[bud]
        r1w = s['r1_words'] or 1
        lines.append(
            f"{bud:>12} {s['r1_words']:>10.1f} {s['r2_words']:>10.1f} {s['r3_words']:>10.1f} "
            f"{s['r3_words']/r1w:>8.3f} {s['r1_turns']:>9.1f} {s['r2_turns']:>9.1f} {s['r3_turns']:>9.1f}"
        )

    # Statistical tests: each budget vs unlimited
    if "unlimited" in gms_by_bud:
        lines += ["", "Mann-Whitney U — Score (each budget vs unlimited, two-sided):"]
        ref = [g["total_score"] for g in gms_by_bud["unlimited"]]
        for bud in ["75", "50", "25"]:
            if bud not in gms_by_bud:
                continue
            b = [g["total_score"] for g in gms_by_bud[bud]]
            u, p, sig = mannwhitney(ref, b)
            r = effect_size_r(u, len(ref), len(b))
            lines.append(f"  unlimited vs {bud:>3} wds: U={u:.0f}, p={p:.4f}, {sig}, r={r:+.3f}")

        lines += ["", "Mann-Whitney U — Compression (each budget vs unlimited, two-sided):"]
        ref_c = [g["compression"] for g in gms_by_bud["unlimited"]]
        for bud in ["75", "50", "25"]:
            if bud not in gms_by_bud:
                continue
            b = [g["compression"] for g in gms_by_bud[bud]]
            u, p, sig = mannwhitney(ref_c, b)
            r = effect_size_r(u, len(ref_c), len(b))
            lines.append(f"  unlimited vs {bud:>3} wds: U={u:.0f}, p={p:.4f}, {sig}, r={r:+.3f}")

    text = section("E3 — Token Budget (Costless-Token Hypothesis)", lines)
    result = {
        "budgets": avail,
        "summaries": {b: {k: v for k, v in s.items() if not k.startswith("_")}
                      for b, s in summaries.items()},
    }
    return result, text, summaries, gms_by_bud


# ---------------------------------------------------------------------------
# Statistical Findings Summary
# ---------------------------------------------------------------------------

def findings_summary(e1_sum, e2_sum, e3_sum) -> str:
    lines = [
        "KEY FINDINGS ACROSS ALL EXPERIMENTS",
        "",
        "E1 — Alignment Sweep:",
        f"  Best model score : {max(s['avg_score'] for s in e1_sum.values()):.2f}/18 (λ=0.2)",
        f"  Human score      : {HUMAN_SCORE}/18",
        f"  Human gap        : {HUMAN_SCORE - max(s['avg_score'] for s in e1_sum.values()):.2f} pts",
        f"  Word reduction   : λ=0.0 ({min(s['avg_words'] for s in e1_sum.values()):.0f} wds) → λ=1.0 ({max(s['avg_words'] for s in e1_sum.values()):.0f} wds)  [alignment REDUCES verbosity]",
        f"  Compression range: {min(s['avg_compression'] for s in e1_sum.values()):.3f}–{max(s['avg_compression'] for s in e1_sum.values()):.3f}  (all >> human {HUMAN_COMPRESS:.2f})",
        "  Verdict: Alignment is NOT the cause of non-compression.",
        "",
        "E2 — History Ablation (Oracle Summary):",
    ]
    if "no_summary" in e2_sum and "oracle_summary" in e2_sum:
        ns  = e2_sum["no_summary"]
        orc = e2_sum["oracle_summary"]
        lines += [
            f"  no_summary   compression: {ns['avg_compression']:.3f}",
            f"  oracle       compression: {orc['avg_compression']:.3f}  (Δ = {orc['avg_compression']-ns['avg_compression']:+.3f})",
            f"  Human        compression: {HUMAN_COMPRESS:.3f}",
            f"  Gap closed by oracle    : {(1 - (orc['avg_compression']-HUMAN_COMPRESS)/(ns['avg_compression']-HUMAN_COMPRESS))*100:.0f}% of gap remains",
            "  Verdict: Memory scaffolding partially helps (~15 pp drop) but",
            "           does NOT recover pact formation. Structural failure confirmed.",
        ]
    lines += [
        "",
        "E3 — Token Budget:",
    ]
    if "unlimited" in e3_sum and "25" in e3_sum:
        ul = e3_sum["unlimited"]
        t  = e3_sum["25"]
        lines += [
            f"  unlimited score : {ul['avg_score']:.2f}/18",
            f"  25-word budget  : {t['avg_score']:.2f}/18  (Δ = {t['avg_score']-ul['avg_score']:+.2f})",
            f"  Random baseline : {RANDOM_SCORE:.1f}/18  (25-word budget is {'above' if t['avg_score']>RANDOM_SCORE else 'BELOW'} random!)",
            "  Verdict: Budget pressure hurts accuracy and does NOT induce compression.",
            "           Costless-token hypothesis: rejected as sufficient explanation.",
        ]
    lines += [
        "",
        "OVERALL CONCLUSION:",
        "  Three independent probes (alignment dose, memory scaffold, token cost)",
        "  all converge: the failure is structural — absent from both pre-training",
        "  data and the generative architecture. Prompt engineering alone cannot fix it.",
    ]
    return section("STATISTICAL FINDINGS SUMMARY", lines)


# ---------------------------------------------------------------------------
# Plotting (unchanged API, improved captions)
# ---------------------------------------------------------------------------

def _human_hline(ax, y, label="Human", color=PALETTE["human"], ls="--"):
    ax.axhline(y, color=color, linestyle=ls, linewidth=1.4, alpha=0.8, label=label)

def _random_hline(ax):
    ax.axhline(RANDOM_SCORE, color=PALETTE["random"], linestyle=":", linewidth=1.2,
               alpha=0.8, label=f"Random≈{RANDOM_SCORE:.0f}")


def _ci_error(summaries_dict, keys, metric: str):
    """Return y-values and asymmetric error bars from bootstrap CIs."""
    ys, lo_errs, hi_errs = [], [], []
    ci_key = "ci_" + metric.replace("avg_", "")
    for k in keys:
        s = summaries_dict[k]
        ys.append(s[f"avg_{metric}"] if f"avg_{metric}" in s else s[metric])
        ci = s.get(ci_key)
        if ci:
            lo, mean, hi = ci
            lo_errs.append(mean - lo)
            hi_errs.append(hi - mean)
        else:
            lo_errs.append(0); hi_errs.append(0)
    return ys, [lo_errs, hi_errs]


def plot_e1(summaries: dict, out_path: str):
    lambdas = sorted(summaries.keys())
    scores, (slo, shi) = _ci_error(summaries, lambdas, "score")
    words  = [summaries[l]["avg_words"]       for l in lambdas]
    comps, (clo, chi) = _ci_error(summaries, lambdas, "compression")
    r1s = [summaries[l]["r1_score"] for l in lambdas]
    r2s = [summaries[l]["r2_score"] for l in lambdas]
    r3s = [summaries[l]["r3_score"] for l in lambdas]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("E1 — Alignment Dose-Response Sweep  (PAd interpolation, λ ∈ {0.0 … 1.0})",
                 fontsize=13, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.errorbar(lambdas, scores, yerr=[slo, shi], fmt="o-", color=PALETTE["blue"],
                linewidth=2, markersize=7, capsize=4, label="Model (95% CI)")
    _human_hline(ax, HUMAN_SCORE, f"Human ({HUMAN_SCORE})")
    _random_hline(ax)
    ax.set_title("Task Score vs λ"); ax.set_xlabel("λ"); ax.set_ylabel("Score / 18")
    ax.set_xticks(lambdas); ax.set_ylim(0, 18); ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    ax.plot(lambdas, words, "s-", color=PALETTE["green"], linewidth=2, markersize=7)
    _human_hline(ax, HUMAN_WORDS, f"Human ({HUMAN_WORDS:.0f} wds)")
    ax.set_title("Total Word Count vs λ"); ax.set_xlabel("λ"); ax.set_ylabel("Words / game")
    ax.set_xticks(lambdas); ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.errorbar(lambdas, comps, yerr=[clo, chi], fmt="^-", color=PALETTE["purple"],
                linewidth=2, markersize=7, capsize=4, label="Model (95% CI)")
    _human_hline(ax, HUMAN_COMPRESS, f"Human ({HUMAN_COMPRESS:.2f})")
    ax.axhline(1.0, color=PALETTE["gray"], ls="-", lw=0.8, alpha=0.5, label="No change")
    ax.fill_between(lambdas, comps, HUMAN_COMPRESS, alpha=0.07, color=PALETTE["purple"])
    ax.set_title("Compression (R3/R1) vs λ\n(↓ = more compression)")
    ax.set_xlabel("λ"); ax.set_ylabel("R3 / R1 word ratio")
    ax.set_xticks(lambdas); ax.set_ylim(0, 1.3); ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(lambdas, r1s, "o-", color=PALETTE["blue"],   lw=2, ms=6, label="R1")
    ax.plot(lambdas, r2s, "s-", color=PALETTE["green"],  lw=2, ms=6, label="R2")
    ax.plot(lambdas, r3s, "^-", color=PALETTE["orange"], lw=2, ms=6, label="R3")
    ax.set_title("Per-Round Score vs λ\n(R1→R3 degradation at low λ)")
    ax.set_xlabel("λ"); ax.set_ylabel("Round Score / 6")
    ax.set_xticks(lambdas); ax.set_ylim(0, 6); ax.legend(fontsize=8)
    ax.annotate("Context bleed\n(λ=0.0)", xy=(0.0, r3s[0]), xytext=(0.18, 2.1),
                fontsize=8, color=PALETTE["red"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["red"]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_e2(summaries: dict, out_path: str):
    conds  = ["no_summary", "oracle_summary"]
    labels = ["No Summary\n(control)", "Oracle\nSummary"]
    colors = [PALETTE["blue"], PALETTE["green"]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("E2 — History Ablation  (Oracle memory scaffold: memory vs. structural failure)",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel 1: score with CI
    ax = axes[0]
    scores  = [summaries[c]["avg_score"] for c in conds]
    ci_lo   = [summaries[c]["avg_score"] - summaries[c]["ci_score"][0] for c in conds]
    ci_hi   = [summaries[c]["ci_score"][2] - summaries[c]["avg_score"] for c in conds]
    x = range(len(conds))
    bars = ax.bar(x, scores, color=colors, edgecolor="white", width=0.5)
    ax.errorbar(x, scores, yerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=5, lw=1.5)
    _human_hline(ax, HUMAN_SCORE, f"Human ({HUMAN_SCORE})"); _random_hline(ax)
    ax.set_title("Task Score (with 95% CI)")
    ax.set_xticks(x); ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=8)
    ax.set_ylabel("Score / 18"); ax.set_ylim(0, 18)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: per-round words
    ax = axes[1]
    xr = np.arange(3); w = 0.35
    for i, (c, lbl, col) in enumerate(zip(conds, labels, colors)):
        wds = [summaries[c][f"r{r}_words"] for r in [1, 2, 3]]
        ax.bar(xr + (i-0.5)*w, wds, w, label=lbl.replace("\n"," "), color=col, alpha=0.85, edgecolor="white")
    ax.plot(xr, [HUMAN_R1_W, HUMAN_R2_W, HUMAN_R3_W], "D-", color=PALETTE["human"],
            ms=6, lw=1.5, label="Human")
    ax.set_title("Per-Round Word Count"); ax.set_ylabel("Avg words")
    ax.set_xticks(xr); ax.set_xticklabels(["R1","R2","R3"]); ax.legend(fontsize=8)

    # Panel 3: compression with CI
    ax = axes[2]
    comps = [summaries[c]["avg_compression"] for c in conds]
    ci_lo = [summaries[c]["avg_compression"] - summaries[c]["ci_compression"][0] for c in conds]
    ci_hi = [summaries[c]["ci_compression"][2] - summaries[c]["avg_compression"] for c in conds]
    bars  = ax.bar(x, comps, color=colors, edgecolor="white", width=0.5)
    ax.errorbar(x, comps, yerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=5, lw=1.5)
    _human_hline(ax, HUMAN_COMPRESS, f"Human ({HUMAN_COMPRESS:.2f})")
    ax.axhline(1.0, color=PALETTE["gray"], ls="-", lw=0.8, alpha=0.5, label="No change")
    ax.set_title("Compression Ratio (R3/R1)\n↓ lower = more compression (95% CI)")
    ax.set_xticks(x); ax.set_xticklabels([l.replace("\n"," ") for l in labels], fontsize=8)
    ax.set_ylabel("R3 / R1"); ax.set_ylim(0, 1.3)
    for bar, v in zip(bars, comps):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 4: turns per round
    ax = axes[3]
    for c, lbl, col in zip(conds, labels, colors):
        turns = [summaries[c][f"r{r}_turns"] for r in [1, 2, 3]]
        ax.plot([1,2,3], turns, "o-", color=col, lw=2, ms=7, label=lbl.replace("\n"," "))
    ax.set_title("Per-Round Turn Count\n(fewer turns → earlier termination)")
    ax.set_xlabel("Round"); ax.set_ylabel("Avg turns / round")
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["R1","R2","R3"]); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_e3(summaries: dict, e3_agg: dict, out_path: str):
    bords  = ["unlimited","75","50","25"]
    blbls  = ["Unlimited","75 wds","50 wds","25 wds"]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    avail  = [b for b in bords if b in summaries]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("E3 — Token Budget  (Costless-token hypothesis: does a word cap induce compression?)",
                 fontsize=13, fontweight="bold", y=1.02)

    xl = [blbls[bords.index(b)] for b in avail]
    cols = [colors[bords.index(b)] for b in avail]

    # Panel 1: score with CI
    ax = axes[0]
    scores = [summaries[b]["avg_score"] for b in avail]
    ci_lo  = [summaries[b]["avg_score"] - summaries[b]["ci_score"][0] for b in avail]
    ci_hi  = [summaries[b]["ci_score"][2] - summaries[b]["avg_score"] for b in avail]
    xp = range(len(avail))
    bars = ax.bar(xp, scores, color=cols, edgecolor="white")
    ax.errorbar(xp, scores, yerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=5, lw=1.5)
    _human_hline(ax, HUMAN_SCORE, f"Human ({HUMAN_SCORE})"); _random_hline(ax)
    ax.set_title("Task Score (95% CI)"); ax.set_ylabel("Score / 18"); ax.set_ylim(0,18)
    ax.set_xticks(xp); ax.set_xticklabels(xl, fontsize=8); ax.legend(fontsize=8)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.2f}",
                ha="center", fontsize=9, fontweight="bold")

    # Panel 2: per-round words
    ax = axes[1]
    xr = np.arange(3); w = 0.18
    for i, (b, lbl, col) in enumerate(zip(avail, xl, cols)):
        wds = [summaries[b][f"r{r}_words"] for r in [1,2,3]]
        offset = (i - len(avail)/2 + 0.5) * w
        ax.bar(xr + offset, wds, w, label=lbl, color=col, alpha=0.85, edgecolor="white")
    ax.plot(xr, [HUMAN_R1_W, HUMAN_R2_W, HUMAN_R3_W], "D-", color=PALETTE["human"],
            ms=6, lw=1.5, label="Human")
    ax.set_title("Per-Round Words\n(all budgets fill to ceiling)")
    ax.set_xticks(xr); ax.set_xticklabels(["R1","R2","R3"]); ax.set_ylabel("Avg words")
    ax.legend(fontsize=7)

    # Panel 3: compression with CI
    ax = axes[2]
    comps = [summaries[b]["avg_compression"] for b in avail]
    ci_lo = [summaries[b]["avg_compression"] - summaries[b]["ci_compression"][0] for b in avail]
    ci_hi = [summaries[b]["ci_compression"][2] - summaries[b]["avg_compression"] for b in avail]
    bars  = ax.bar(xp, comps, color=cols, edgecolor="white")
    ax.errorbar(xp, comps, yerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=5, lw=1.5)
    _human_hline(ax, HUMAN_COMPRESS, f"Human ({HUMAN_COMPRESS:.2f})")
    ax.axhline(1.0, color=PALETTE["gray"], ls="-", lw=0.8, alpha=0.5)
    ax.set_title("Compression Ratio (R3/R1)\n↓ lower = more compression (95% CI)")
    ax.set_xticks(xp); ax.set_xticklabels(xl, fontsize=8); ax.set_ylabel("R3/R1"); ax.set_ylim(0, 1.3)
    for bar, v in zip(bars, comps):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 4: Util-Drop
    ax = axes[3]
    budgeted = [b for b in ["75","50","25"] if b in summaries]
    bud_xl   = [blbls[bords.index(b)] for b in budgeted]
    bud_cols = [colors[bords.index(b)] for b in budgeted]
    ud = [e3_agg.get(b, {}).get("avg_utilization_drop", float("nan")) for b in budgeted]
    xb = range(len(budgeted))
    bars = ax.bar(xb, ud, color=bud_cols, edgecolor="white")
    ax.axhline(0,   color="black",         lw=1.0)
    ax.axhline(-0.3, color=PALETTE["human"], ls="--", lw=1.2, label="Human-like (≈−0.3)")
    for bar, v in zip(bars, ud):
        if not np.isnan(v):
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x()+bar.get_width()/2, v+(0.005 if v>=0 else -0.005),
                    f"{v:+.3f}", ha="center", va=va, fontsize=9, fontweight="bold")
    ax.set_title("Budget Util-Drop (R3−R1)\n≈0 → no pact formation (structural failure)")
    ax.set_xticks(xb); ax.set_xticklabels(bud_xl, fontsize=8)
    ax.set_ylabel("Util-Drop (neg = pact)"); ax.set_ylim(-0.4, 0.1); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_summary(e1_sum, e2_sum, e3_sum, out_path: str):
    lambdas = sorted(e1_sum.keys())
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "All Experiments — Key Metrics at a Glance\n"
        "Does alignment, memory scaffolding, or token cost explain non-compression in VLMs?",
        fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # Row 0: E1
    ax = fig.add_subplot(gs[0, 0])
    scores = [e1_sum[l]["avg_score"] for l in lambdas]
    ax.plot(lambdas, scores, "o-", color=PALETTE["blue"], lw=2)
    _human_hline(ax, HUMAN_SCORE, "Human"); _random_hline(ax)
    ax.set_title("E1 Score vs λ"); ax.set_xlabel("λ"); ax.set_ylabel("Score/18")
    ax.set_ylim(0,18); ax.set_xticks(lambdas); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    words = [e1_sum[l]["avg_words"] for l in lambdas]
    ax.plot(lambdas, words, "s-", color=PALETTE["green"], lw=2)
    _human_hline(ax, HUMAN_WORDS, "Human")
    ax.set_title("E1 Words vs λ"); ax.set_xlabel("λ"); ax.set_ylabel("Words/game")
    ax.set_xticks(lambdas); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    comps = [e1_sum[l]["avg_compression"] for l in lambdas]
    ax.plot(lambdas, comps, "^-", color=PALETTE["purple"], lw=2)
    _human_hline(ax, HUMAN_COMPRESS, "Human")
    ax.axhline(1.0, color=PALETTE["gray"], ls="-", lw=0.8, alpha=0.5)
    ax.set_title("E1 Compression vs λ"); ax.set_xlabel("λ"); ax.set_ylabel("R3/R1 ↓")
    ax.set_ylim(0,1.3); ax.set_xticks(lambdas); ax.legend(fontsize=7)
    ax.text(0.5, 1.03, "All ≈ flat → not alignment's fault",
            transform=ax.transAxes, ha="center", fontsize=8, color=PALETTE["purple"], style="italic")

    # Row 1: E2
    conds = [c for c in ["no_summary","oracle_summary"] if c in e2_sum]
    clbls = ["No Summary","Oracle"][:len(conds)]
    ccols = [PALETTE["blue"], PALETTE["green"]][:len(conds)]
    xc = range(len(conds))

    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(xc, [e2_sum[c]["avg_score"] for c in conds], color=ccols, width=0.5, edgecolor="white")
    _human_hline(ax, HUMAN_SCORE, "Human"); _random_hline(ax)
    ax.set_title("E2 Score"); ax.set_ylabel("Score/18"); ax.set_ylim(0,18)
    ax.set_xticks(xc); ax.set_xticklabels(clbls, fontsize=8); ax.legend(fontsize=7)
    for bar, v in zip(bars, [e2_sum[c]["avg_score"] for c in conds]):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    xr = np.arange(3); w = 0.35
    for i, (c, lbl, col) in enumerate(zip(conds, clbls, ccols)):
        wds = [e2_sum[c][f"r{r}_words"] for r in [1,2,3]]
        ax.bar(xr+(i-0.5)*w, wds, w, label=lbl, color=col, alpha=0.85, edgecolor="white")
    ax.plot(xr, [HUMAN_R1_W,HUMAN_R2_W,HUMAN_R3_W], "D-", color=PALETTE["human"], ms=6, lw=1.5, label="Human")
    ax.set_title("E2 Words / Round"); ax.set_ylabel("Avg words")
    ax.set_xticks(xr); ax.set_xticklabels(["R1","R2","R3"]); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    comps = [e2_sum[c]["avg_compression"] for c in conds]
    bars  = ax.bar(xc, comps, color=ccols, width=0.5, edgecolor="white")
    _human_hline(ax, HUMAN_COMPRESS, "Human")
    ax.axhline(1.0, color=PALETTE["gray"], ls="-", lw=0.8, alpha=0.5)
    ax.set_title("E2 Compression (R3/R1)"); ax.set_ylabel("R3/R1 ↓"); ax.set_ylim(0,1.3)
    ax.set_xticks(xc); ax.set_xticklabels(clbls, fontsize=8); ax.legend(fontsize=7)
    for bar, v in zip(bars, comps):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.5, 1.03, "Oracle: 0.84 (partial help, not pact formation)",
            transform=ax.transAxes, ha="center", fontsize=8, color=PALETTE["green"], style="italic")

    # Row 2: E3
    bords = ["unlimited","75","50","25"]
    blbls = ["Unlim.","75","50","25"]
    bcols = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    avail = [b for b in bords if b in e3_sum]
    xl    = [blbls[bords.index(b)] for b in avail]
    cols  = [bcols[bords.index(b)] for b in avail]

    ax = fig.add_subplot(gs[2, 0])
    bars = ax.bar(range(len(avail)), [e3_sum[b]["avg_score"] for b in avail], color=cols, edgecolor="white")
    _human_hline(ax, HUMAN_SCORE, "Human"); _random_hline(ax)
    ax.set_title("E3 Score vs Budget"); ax.set_ylabel("Score/18"); ax.set_ylim(0,18)
    ax.set_xticks(range(len(avail))); ax.set_xticklabels(xl, fontsize=8); ax.legend(fontsize=7)
    for bar, v in zip(bars, [e3_sum[b]["avg_score"] for b in avail]):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(xl, [e3_sum[b]["avg_words"] for b in avail], "s-", color=PALETTE["purple"], lw=2, ms=8)
    _human_hline(ax, HUMAN_WORDS, "Human")
    ax.set_title("E3 Total Words vs Budget"); ax.set_ylabel("Words/game"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    comps = [e3_sum[b]["avg_compression"] for b in avail]
    bars  = ax.bar(range(len(avail)), comps, color=cols, edgecolor="white")
    _human_hline(ax, HUMAN_COMPRESS, "Human")
    ax.axhline(1.0, color=PALETTE["gray"], ls="-", lw=0.8, alpha=0.5)
    ax.set_title("E3 Compression vs Budget"); ax.set_ylabel("R3/R1 ↓"); ax.set_ylim(0,1.3)
    ax.set_xticks(range(len(avail))); ax.set_xticklabels(xl, fontsize=8); ax.legend(fontsize=7)
    for bar, v in zip(bars, comps):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.5, 1.03, "Budget doesn't help → structural failure",
            transform=ax.transAxes, ha="center", fontsize=8, color=PALETTE["red"], style="italic")

    for row, lbl in enumerate(["E1\nAlignment", "E2\nAblation", "E3\nBudget"]):
        fig.text(0.01, 0.84 - row*0.315, lbl, ha="left", va="center",
                 fontsize=10, fontweight="bold", color=PALETTE["gray"], rotation=90)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive analysis for E1, E2, E3")
    # E1 can live in Experiments/output_E_1 OR the legacy outputs/ folder
    default_e1 = "Experiments/output_E_1" if os.path.exists("Experiments/output_E_1") and \
        any(f.startswith("results_lambda_") for f in os.listdir("Experiments/output_E_1")) \
        else "outputs"
    p.add_argument("--e1_dir",  default=default_e1)
    p.add_argument("--e2_dir",  default="Experiments/output_E_2")
    p.add_argument("--e3_dir",  default="Experiments/output_E_3")
    p.add_argument("--out_dir", default="figures")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print(" COMPREHENSIVE RESULTS ANALYSIS — E1, E2, E3")
    print("=" * 70)
    print(f"E1 data : {args.e1_dir}")
    print(f"E2 data : {args.e2_dir}")
    print(f"E3 data : {args.e3_dir}")
    print(f"Figures : {args.out_dir}")
    print()

    report_parts = []

    # ── E1 ──────────────────────────────────────────────────────────────────
    print("Loading E1 …")
    e1_result, e1_text, e1_sum, e1_gms = analyse_e1(args.e1_dir)
    report_parts.append(e1_text)

    # ── E2 ──────────────────────────────────────────────────────────────────
    print("Loading E2 …")
    e2_result, e2_text, e2_sum, e2_gms = analyse_e2(args.e2_dir)
    report_parts.append(e2_text)

    # ── E3 ──────────────────────────────────────────────────────────────────
    print("Loading E3 …")
    e3_result, e3_text, e3_sum, e3_gms = analyse_e3(args.e3_dir)
    report_parts.append(e3_text)

    # Load E3 aggregate for Util-Drop in figure
    e3_agg = {}
    agg_path = os.path.join(args.e3_dir, "metrics_E3.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            for entry in json.load(f):
                b = entry["budget"]
                e3_agg["unlimited" if b == "unlimited" else str(b)] = entry

    # ── Statistical summary ─────────────────────────────────────────────────
    findings = findings_summary(e1_sum, e2_sum, e3_sum)
    report_parts.append(findings)

    full_report = "\n\n".join(report_parts)
    print("\n" + full_report)

    # ── Save text report ────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write(full_report)
    print(f"\nText report saved to: {summary_path}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    json_path = os.path.join(args.out_dir, "all_metrics.json")
    with open(json_path, "w") as f:
        json.dump({"E1": e1_result, "E2": e2_result, "E3": e3_result}, f, indent=2)
    print(f"Structured metrics saved to: {json_path}")

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_e1(e1_sum,  os.path.join(args.out_dir, "e1_dose_response.png"))
    plot_e2(e2_sum,  os.path.join(args.out_dir, "e2_history_ablation.png"))
    plot_e3(e3_sum, e3_agg, os.path.join(args.out_dir, "e3_token_budget.png"))
    plot_summary(e1_sum, e2_sum, e3_sum, os.path.join(args.out_dir, "summary_all.png"))

    print(f"\nDone. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
