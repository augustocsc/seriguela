"""Generate all data figures for the dissertation from real results in results/.

Every figure produced here uses REAL data only (no fictitious values).
Run from anywhere:  python make_figures.py
Outputs PDF (for LaTeX) + PNG (for preview) into ../figures/.
"""
import json
import glob
import os
import re
import csv
import math
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = r"C:/Users/Rudá/seriguela"
RES = os.path.join(REPO, "results")
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALGOS = ["best_of_n", "pure_ppo", "pure_grpo", "bon_ppo", "bon_grpo"]
LABEL = {
    "best_of_n": "Best-of-N",
    "pure_ppo": "Pure-PPO",
    "pure_grpo": "Pure-GRPO",
    "bon_ppo": "BoN-PPO",
    "bon_grpo": "BoN-GRPO",
}
COLOR = {
    "best_of_n": "#7f7f7f",
    "pure_ppo": "#1f77b4",
    "pure_grpo": "#2ca02c",
    "bon_ppo": "#d62728",
    "bon_grpo": "#9467bd",
}
PROBLEMS_P2 = ["nguyen_3", "nguyen_5", "nguyen_7", "nguyen_9"]
PTITLE = {
    "nguyen_3": "Nguyen-3",
    "nguyen_5": "Nguyen-5",
    "nguyen_7": "Nguyen-7",
    "nguyen_9": "Nguyen-9",
}
CAT_COLOR = {"nguyen": "#d62728", "feynman": "#1f77b4", "strogatz": "#2ca02c"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def load_phase2_aggregate(problem, algo):
    pat = os.path.join(RES, "phase_2", "base_infix", problem, algo, "aggregate_*.json")
    files = sorted(glob.glob(pat))
    # prefer the multi-seed aggregate (longest seed list in filename)
    files.sort(key=lambda p: len(os.path.basename(p)), reverse=True)
    if not files:
        return None
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- fig 1
def fig_zero_shot_selection():
    """Base vs Large zero-shot best-of-10-seeds R^2 on the 122 problems."""
    probs = sorted(os.listdir(os.path.join(RES, "phase_1b")))
    xs, ys, cats = [], [], []
    for p in probs:
        pt = {}
        for phase in ["phase_1b", "phase_1c"]:
            fs = glob.glob(os.path.join(RES, phase, p, "aggregate_*.json"))
            if not fs:
                continue
            d = json.load(open(fs[0], encoding="utf-8"))
            rs = [r.get("best_r2") for r in d.get("individual_results", [])
                  if r.get("best_r2") is not None]
            if rs:
                pt[phase] = max(rs)
        if "phase_1b" in pt and "phase_1c" in pt:
            xs.append(pt["phase_1b"])
            ys.append(pt["phase_1c"])
            cats.append(p.split("_")[0])
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for cat in ["feynman", "strogatz", "nguyen"]:
        sel = [i for i, c in enumerate(cats) if c == cat]
        ax.scatter([xs[i] for i in sel], [ys[i] for i in sel], s=18, alpha=0.75,
                   color=CAT_COLOR[cat], label=f"{cat.capitalize()} (n={len(sel)})",
                   edgecolors="white", linewidths=0.3)
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.8, zorder=0)
    ax.set_xlabel("GPT-2 Base (124M) — best $R^2$ over 10 seeds")
    ax.set_ylabel("GPT-2 Large (774M) — best $R^2$ over 10 seeds")
    nb = sum(1 for a, b in zip(xs, ys) if a > b)
    ax.annotate(f"Base wins: {nb}/{len(xs)} problems\n"
                f"mean $R^2$: Base {st.mean(xs):.3f} vs Large {st.mean(ys):.3f}",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc"))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_zero_shot_selection")


# ---------------------------------------------------------------- fig 2
def fig_phase2_box():
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, prob in zip(axes, PROBLEMS_P2):
        for i, algo in enumerate(ALGOS):
            d = load_phase2_aggregate(prob, algo)
            if d is None:
                continue
            vals = [r["best_r2"] for r in d["individual_results"]]
            x = np.full(len(vals), i) + rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(x, vals, s=22, color=COLOR[algo], alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4)
            m = st.mean(vals)
            ax.hlines(m, i - 0.28, i + 0.28, color=COLOR[algo], lw=2.2, zorder=4)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            ax.vlines(i, m - sd, m + sd, color=COLOR[algo], lw=1.0, zorder=2, alpha=0.7)
        ax.set_title(PTITLE[prob])
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels([LABEL[a] for a in ALGOS], rotation=38, ha="right")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="#bbbbbb", lw=0.6, ls=":")
    axes[0].set_ylabel("Best $R^2$ (per seed)")
    fig.tight_layout()
    save(fig, "fig_phase2_box")


# ---------------------------------------------------------------- fig 3+4
def _histories(prob):
    """dict algo -> list of per-seed history lists"""
    out = {}
    for algo in ALGOS:
        d = load_phase2_aggregate(prob, algo)
        if d is None:
            continue
        hs = []
        for r in d["individual_results"]:
            h = r.get("history")
            if h:
                hs.append(h)
        if hs:
            out[algo] = hs
    return out


def fig_phase2_convergence():
    prob = "nguyen_5"
    H = _histories(prob)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    # Best-of-N has no per-step history (single sampling pass at the same budget):
    dbon = load_phase2_aggregate(prob, "best_of_n")
    if dbon is not None:
        m = st.mean([r["best_r2"] for r in dbon["individual_results"]])
        ax.axhline(m, color=COLOR["best_of_n"], lw=1.8, ls="--",
                   label="Best-of-N (final, same budget)")
    for algo in ALGOS:
        if algo not in H:
            continue
        curves = []
        for h in H[algo]:
            best = [s.get("best_r2", 0.0) for s in h]
            # enforce running max (best-so-far)
            for i in range(1, len(best)):
                best[i] = max(best[i], best[i - 1])
            curves.append(best)
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        steps = np.arange(n)
        mean = arr.mean(axis=0)
        ax.plot(steps, mean, color=COLOR[algo], lw=1.8, label=LABEL[algo])
        ax.fill_between(steps, arr.min(axis=0), arr.max(axis=0),
                        color=COLOR[algo], alpha=0.12, lw=0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Best $R^2$ so far (mean over 5 seeds)")
    ax.set_title("Nguyen-5 — convergence (shaded: min–max across seeds)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, "fig_phase2_convergence")


def fig_phase2_validity():
    prob = "nguyen_5"
    H = _histories(prob)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for algo in ALGOS:
        if algo not in H:
            continue
        curves = []
        for h in H[algo]:
            v = [s.get("valid_rate", np.nan) for s in h]
            curves.append(v)
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        mean = np.nanmean(arr, axis=0)
        # light smoothing (window 5)
        k = 5
        sm = np.convolve(mean, np.ones(k) / k, mode="same")
        sm[:k // 2] = mean[:k // 2]
        sm[-(k // 2):] = mean[-(k // 2):]
        ax.plot(np.arange(n), sm, color=COLOR[algo], lw=1.7, label=LABEL[algo])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Valid-expression rate")
    ax.set_title("Nguyen-5 — the validity valley (mean over 5 seeds)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    save(fig, "fig_phase2_validity")


# ---------------------------------------------------------------- fig 5
def fig_phase2_exploration():
    prob = "nguyen_5"
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for algo in ALGOS:
        d = load_phase2_aggregate(prob, algo)
        if d is None:
            continue
        uniq = [r.get("total_unique_expressions") for r in d["individual_results"]
                if r.get("total_unique_expressions")]
        r2 = [r["best_r2"] for r in d["individual_results"]]
        if not uniq:
            continue
        mu, mr = st.mean(uniq), st.mean(r2)
        marker = "s" if algo.startswith("bon") else "o"
        ax.errorbar(mu, mr,
                    xerr=st.stdev(uniq) if len(uniq) > 1 else 0,
                    yerr=st.stdev(r2) if len(r2) > 1 else 0,
                    fmt=marker, ms=9, color=COLOR[algo], capsize=3, lw=1.1)
        offsets = {"pure_ppo": (8, -16), "bon_ppo": (8, 8), "pure_grpo": (-6, -22),
                   "bon_grpo": (10, 12), "best_of_n": (8, 6)}
        ax.annotate(LABEL[algo], (mu, mr), textcoords="offset points",
                    xytext=offsets.get(algo, (8, 6)), fontsize=8.5, color=COLOR[algo])
    ax.set_xscale("log")
    ax.set_xlabel("Unique expressions discovered per seed (exploration)")
    ax.set_ylabel("Best $R^2$ (mean over 5 seeds)")
    ax.set_title("Nguyen-5 — exploration vs. performance\n(squares: with elite buffer)")
    ax.set_ylim(0, 1.05)
    save(fig, "fig_phase2_exploration")


# ---------------------------------------------------------------- fig 6
def read_mother_table():
    rows = []
    with open(os.path.join(RES, "mother_table.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def fig_hgen():
    rows = [r for r in read_mother_table()
            if r["status"] == "done" and r["best_r2"] not in ("", None)]
    by_v = {}
    for r in rows:
        v = int(r["n_vars"])
        by_v.setdefault(v, []).append((float(r["best_r2"]), r["cat"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = sorted(by_v)
    data = [[a for a, _ in by_v[v]] for v in xs]
    bp = ax.boxplot(data, positions=xs, widths=0.55, showfliers=False,
                    medianprops=dict(color="#d62728", lw=1.6),
                    boxprops=dict(color="#444444"), whiskerprops=dict(color="#444444"),
                    capprops=dict(color="#444444"))
    rng = np.random.default_rng(1)
    seen = set()
    for v in xs:
        for val, cat in by_v[v]:
            lab = cat.capitalize() if cat not in seen else None
            seen.add(cat)
            ax.scatter(v + rng.uniform(-0.16, 0.16), val, s=14, alpha=0.65,
                       color=CAT_COLOR[cat], zorder=3, label=lab,
                       edgecolors="white", linewidths=0.3)
    meds = [st.median(d) for d in data]
    ax.plot(xs, meds, color="#d62728", lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.axvspan(5.5, max(xs) + 0.6, color="#fff3cd", alpha=0.45, zorder=0)
    ax.annotate("beyond SFT training scope\n($n_{vars}>5$)", xy=(7.2, 0.08),
                fontsize=8, ha="center", color="#7a5c00")
    ax.set_xlabel("Number of input variables ($n_{vars}$)")
    ax.set_ylabel("Best $R^2$ (train)")
    ax.set_xticks(xs)
    counts = ", ".join(f"{v}: {len(by_v[v])}" for v in xs)
    ax.set_title(f"Champion RL across all {sum(len(d) for d in data)} problems, by dimensionality")
    ax.set_ylim(-0.05, 1.06)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    save(fig, "fig_hgen")
    print("  hgen counts per n_vars:", {v: len(by_v[v]) for v in xs})


# ---------------------------------------------------------------- fig 7
def fig_srbench():
    feyn = [
        ("AIFeynman", 57.3), ("AFP\\_FE", 27.0), ("GP-GOMEA", 26.8), ("ITEA", 22.4),
        ("AFP", 21.1), ("DSR", 19.7), ("Operon", 16.6), ("gplearn", 16.4),
        ("Seriguela (ours)", 15.6), ("SBP-GP", 12.7), ("EPLEX", 12.4), ("BSR", 2.5),
        ("FEAT", 0.0), ("FFX", 0.0), ("MRGP", 0.0),
    ]
    stro = [
        ("GP-GOMEA", 29.5), ("AIFeynman", 28.2), ("AFP\\_FE", 20.0), ("DSR", 19.6),
        ("AFP", 15.2), ("SBP-GP", 11.6), ("Operon", 11.4), ("EPLEX", 8.9),
        ("gplearn", 8.9), ("Seriguela (ours)", 7.1), ("ITEA", 7.1),
        ("FEAT", 0.9), ("BSR", 0.9), ("FFX", 0.0), ("MRGP", 0.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, data, title in zip(axes, [feyn, stro],
                               ["Feynman (96 problems)", "Strogatz (14 problems)"]):
        names = [n.replace("\\_", "_") for n, _ in data][::-1]
        vals = [v for _, v in data][::-1]
        colors = ["#d62728" if "Seriguela" in n else "#9db8d2" for n in names]
        ax.barh(range(len(names)), vals, color=colors, height=0.68)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.5)
        for i, v in enumerate(vals):
            ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=7)
        ax.set_xlabel("Solution rate (%)")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, max(vals) * 1.18)
    fig.suptitle("Symbolic-solution rate vs. published SRBench results (noise 0)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig_srbench")


# ---------------------------------------------------------------- fig 8
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\*\*|[-+*/]|\d+\.?\d*")


def complexity(expr):
    if not expr:
        return None
    return len(TOKEN_RE.findall(expr))


def fig_pareto():
    rows = [r for r in read_mother_table()
            if r["status"] == "done" and r["best_r2"] not in ("", None) and r["found"]]
    xs, ys, cats = [], [], []
    for r in rows:
        c = complexity(r["found"])
        if c is None:
            continue
        xs.append(c)
        ys.append(float(r["best_r2"]))
        cats.append(r["cat"])
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    seen = set()
    rng = np.random.default_rng(2)
    for x, y, c in zip(xs, ys, cats):
        lab = c.capitalize() if c not in seen else None
        seen.add(c)
        ax.scatter(x + rng.uniform(-0.25, 0.25), y, s=20, alpha=0.7,
                   color=CAT_COLOR[c], label=lab, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Complexity of best expression (token count)")
    ax.set_ylabel("Best $R^2$ (train)")
    ax.set_title("Accuracy vs. complexity of champion-discovered expressions")
    ax.set_ylim(-0.05, 1.06)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_pareto")


# ---------------------------------------------------------------- fig 9
def fig_sric_deadzone():
    """Pilot evidence of the sr_ic dead zone: reward flat/negative -> no learning."""
    cand = glob.glob(os.path.join(RES, "pilot_timing", "solo_pure_ppo_nguyen5",
                                  "**", "results_latest.json"), recursive=True)
    hist_ic, tag = None, ""
    for c in cand:
        d = json.load(open(c, encoding="utf-8"))
        tag = d.get("reward_fn", "")
        hist_ic = d.get("history")
        break
    # phase-2 pure_ppo (r2_clipped): pick the seed that solved the problem
    d2 = load_phase2_aggregate("nguyen_5", "pure_ppo")
    hist_rc = None
    if d2:
        solved = sorted(d2["individual_results"], key=lambda r: -r["best_r2"])
        hist_rc = solved[0].get("history")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1))
    C_IC, C_RC = "#8c564b", COLOR["pure_ppo"]
    if hist_ic:
        mr = [s.get("mean_reward", np.nan) for s in hist_ic]
        ax1.plot(range(len(mr)), mr, color=C_IC, lw=1.2,
                 label="sr_ic (pilot, 500 steps)")
    if hist_rc:
        mr2 = [s.get("mean_reward", np.nan) for s in hist_rc]
        ax1.plot(range(len(mr2)), mr2, color=C_RC, lw=1.2,
                 label="r2_clipped (Phase 2)")
    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean batch reward")
    ax1.set_title("(a) Reward signal received")
    ax1.legend(frameon=False, fontsize=8)
    if hist_ic:
        best = [s.get("best_r2", 0.0) for s in hist_ic]
        for i in range(1, len(best)):
            best[i] = max(best[i], best[i - 1])
        ax2.plot(range(len(best)), best, color=C_IC, lw=1.8,
                 label="sr_ic: stuck (sampling only)")
    if hist_rc:
        best2 = [s.get("best_r2", 0.0) for s in hist_rc]
        for i in range(1, len(best2)):
            best2[i] = max(best2[i], best2[i - 1])
        ax2.plot(range(len(best2)), best2, color=C_RC, lw=1.8,
                 label="r2_clipped: learns, solves")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Best $R^2$ so far")
    ax2.set_title("(b) Consequence on discovery")
    ax2.set_ylim(-0.05, 1.08)
    ax2.legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Pure-PPO on Nguyen-5: the sr_ic reward dead zone under the C=1 protocol",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig_sric_deadzone")
    print("  pilot reward_fn:", tag)


if __name__ == "__main__":
    fig_zero_shot_selection()
    fig_phase2_box()
    fig_phase2_convergence()
    fig_phase2_validity()
    fig_phase2_exploration()
    fig_hgen()
    fig_srbench()
    fig_pareto()
    fig_sric_deadzone()
    print("done ->", OUT)
