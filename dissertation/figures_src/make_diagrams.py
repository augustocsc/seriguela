"""Conceptual diagrams for the dissertation (matplotlib, flat style).

These replace figures referenced in the original text whose source files were
missing from the repository (Overview.pdf, Data Engeneering.pdf, genetic_tree.pdf,
neuron.pdf, GPT-2 Tokenization Result.pdf) and add the RL loop diagram.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight"})

BLUE = "#dbe9f6"
GREEN = "#e2f0d9"
ORANGE = "#fbe5d6"
GRAY = "#f2f2f2"
EDGE = "#5b7fa6"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc=BLUE, fontsize=8.5, bold=False, ec=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color="#444444", style="-|>", lw=1.2, connectionstyle=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=connectionstyle,
                                 shrinkA=2, shrinkB=2))


# ------------------------------------------------------------------ overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    # stage 1
    box(ax, 0.15, 1.1, 2.6, 1.6, "", fc=GREEN)
    ax.text(1.45, 2.45, "1. Data Engineering", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 0.35, 1.75, 2.2, 0.45, "generate 758K expressions", fc="white", fontsize=7.5)
    box(ax, 0.35, 1.25, 2.2, 0.45, "clean + build prompts", fc="white", fontsize=7.5)
    ax.text(1.45, 0.85, "(runs once)", ha="center", fontsize=7.5, style="italic", color="#555555")
    # stage 2
    box(ax, 3.35, 1.1, 2.6, 1.6, "", fc=BLUE)
    ax.text(4.65, 2.45, "2. Supervised Fine-Tuning", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 3.55, 1.75, 2.2, 0.45, "GPT-2 + LoRA adapters", fc="white", fontsize=7.5)
    box(ax, 3.55, 1.25, 2.2, 0.45, "expression generator LM", fc="white", fontsize=7.5)
    ax.text(4.65, 0.85, "(runs once)", ha="center", fontsize=7.5, style="italic", color="#555555")
    # stage 3
    box(ax, 6.55, 1.1, 3.3, 1.6, "", fc=ORANGE)
    ax.text(8.2, 2.45, "3. Expression Discovery (RL)", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 6.75, 1.75, 1.4, 0.45, "generate\nbatch", fc="white", fontsize=7)
    box(ax, 8.3, 1.75, 1.35, 0.45, "evaluate\n($R^2$ reward)", fc="white", fontsize=7)
    box(ax, 6.75, 1.25, 1.4, 0.45, "policy update\n(PPO / GRPO)", fc="white", fontsize=7)
    box(ax, 8.3, 1.25, 1.35, 0.45, "elite buffer\n(top-$K$)", fc="white", fontsize=7)
    ax.text(8.2, 0.85, "(loop per input dataset)", ha="center", fontsize=7.5,
            style="italic", color="#555555")
    # arrows between stages
    arrow(ax, 2.78, 1.9, 3.32, 1.9)
    arrow(ax, 5.98, 1.9, 6.52, 1.9)
    # inner loop arrows
    arrow(ax, 8.15, 1.975, 8.28, 1.975)
    arrow(ax, 8.95, 1.73, 8.95, 1.72)
    arrow(ax, 8.28, 1.475, 8.17, 1.475)
    arrow(ax, 8.97, 1.75, 8.97, 1.72, lw=1.0)
    # inputs / outputs
    box(ax, 3.5, 3.0, 2.3, 0.5, "tabular dataset $D=\\{(\\mathbf{x}_i, y_i)\\}$", fc=GRAY, fontsize=7.5)
    arrow(ax, 5.8, 3.25, 7.6, 2.72, connectionstyle="arc3,rad=-0.25")
    box(ax, 6.9, 0.05, 2.6, 0.5, "best expression  $f^*(\\mathbf{x})$", fc=GRAY, fontsize=8)
    arrow(ax, 8.2, 1.08, 8.2, 0.57)
    save(fig, "fig_overview")


# ------------------------------------------------------------- data pipeline
def fig_data_engineering():
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    box(ax, 0.15, 1.0, 2.7, 1.35, "", fc=GREEN)
    ax.text(1.5, 2.1, "1.1 Generation", ha="center", fontsize=9, fontweight="bold")
    ax.text(1.5, 1.55, "random expression trees\n(Lample & Charton sampler)\nprefix $\\rightarrow$ infix",
            ha="center", fontsize=7.2)
    box(ax, 3.4, 1.0, 2.7, 1.35, "", fc=BLUE)
    ax.text(4.75, 2.1, "1.2 Cleaning", ha="center", fontsize=9, fontweight="bold")
    ax.text(4.75, 1.55, "SymPy parse validation\ndeduplication\ndrop invalid syntax",
            ha="center", fontsize=7.2)
    box(ax, 6.65, 1.0, 3.2, 1.35, "", fc=ORANGE)
    ax.text(8.25, 2.1, "1.3 Prompt Engineering", ha="center", fontsize=9, fontweight="bold")
    ax.text(8.25, 1.55, "vars / oper / cons header\ncontext augmentation\ninfix + prefix variants",
            ha="center", fontsize=7.2)
    arrow(ax, 2.88, 1.67, 3.38, 1.67)
    arrow(ax, 6.13, 1.67, 6.63, 1.67)
    ax.text(1.5, 0.55, "raw trees", ha="center", fontsize=7.5, color="#555555", style="italic")
    ax.text(4.75, 0.55, "valid unique expressions", ha="center", fontsize=7.5,
            color="#555555", style="italic")
    ax.text(8.25, 0.55, "758,255 training samples (90/10 split)", ha="center",
            fontsize=7.5, color="#555555", style="italic")
    save(fig, "fig_data_engineering")


# ------------------------------------------------------------------ GP trees
def draw_tree(ax, x0, y0, nodes, edges, scale=1.0, highlight=None, hcolor="#fbe5d6"):
    """nodes: dict name->(dx,dy,label); edges: list (a,b)."""
    for a, b in edges:
        xa, ya, _ = nodes[a]
        xb, yb, _ = nodes[b]
        ax.plot([x0 + xa * scale, x0 + xb * scale], [y0 + ya * scale, y0 + yb * scale],
                color="#888888", lw=1.0, zorder=1)
    for name, (dx, dy, lab) in nodes.items():
        fc = hcolor if (highlight and name in highlight) else "white"
        c = Circle((x0 + dx * scale, y0 + dy * scale), 0.16 * scale, fc=fc,
                   ec="#5b7fa6", lw=1.1, zorder=2)
        ax.add_patch(c)
        ax.text(x0 + dx * scale, y0 + dy * scale, lab, ha="center", va="center",
                fontsize=8, zorder=3)


def fig_genetic_tree():
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.2, 3.3)
    ax.axis("off")
    ax.set_aspect("equal")
    # parent 1:  (x + sin(y))
    t1 = {"r": (0.5, 2.2, "+"), "a": (0.0, 1.5, "x"), "b": (1.0, 1.5, "sin"),
          "c": (1.0, 0.8, "y")}
    e1 = [("r", "a"), ("r", "b"), ("b", "c")]
    # parent 2: (x * y) - 2
    t2 = {"r": (0.5, 2.2, "$-$"), "a": (0.0, 1.5, "×"), "b": (1.0, 1.5, "2"),
          "c": (-0.35, 0.8, "x"), "d": (0.35, 0.8, "y")}
    e2 = [("r", "a"), ("r", "b"), ("a", "c"), ("a", "d")]
    draw_tree(ax, 0.6, 0.25, t1, e1, highlight={"b", "c"})
    draw_tree(ax, 3.1, 0.25, t2, e2, highlight={"a", "c", "d"})
    ax.text(1.6, 3.05, "1. Parents selected by fitness", fontsize=9, fontweight="bold")
    ax.text(1.1, -0.12, "$x + \\sin(y)$", ha="center", fontsize=8)
    ax.text(3.6, -0.12, "$x \\cdot y - 2$", ha="center", fontsize=8)
    # crossover arrow
    arrow(ax, 4.9, 1.6, 5.9, 1.6, lw=1.6)
    ax.text(5.4, 1.95, "subtree\ncrossover", ha="center", fontsize=8, style="italic")
    # offspring 1: x + (x*y)
    o1 = {"r": (0.5, 2.2, "+"), "a": (0.0, 1.5, "x"), "b": (1.0, 1.5, "×"),
          "c": (0.65, 0.8, "x"), "d": (1.35, 0.8, "y")}
    eo1 = [("r", "a"), ("r", "b"), ("b", "c"), ("b", "d")]
    # offspring 2: sin(y) - 2
    o2 = {"r": (0.5, 2.2, "$-$"), "a": (0.0, 1.5, "sin"), "b": (1.0, 1.5, "2"),
          "c": (0.0, 0.8, "y")}
    eo2 = [("r", "a"), ("r", "b"), ("a", "c")]
    draw_tree(ax, 6.3, 0.25, o1, eo1, highlight={"b", "c", "d"}, hcolor="#e2f0d9")
    draw_tree(ax, 8.9, 0.25, o2, eo2, highlight={"a", "c"}, hcolor="#e2f0d9")
    ax.text(7.5, 3.05, "2. Offspring (next generation)", fontsize=9, fontweight="bold")
    ax.text(6.95, -0.12, "$x + x \\cdot y$", ha="center", fontsize=8)
    ax.text(9.4, -0.12, "$\\sin(y) - 2$", ha="center", fontsize=8)
    save(fig, "fig_genetic_tree")


# -------------------------------------------------------------------- neuron
def fig_neuron():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax in (ax1, ax2):
        ax.axis("off")
        ax.set_aspect("equal")
    # left: single neuron
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 4)
    for i, y in enumerate([3.2, 2.0, 0.8]):
        c = Circle((0.7, y), 0.28, fc="white", ec=EDGE, lw=1.1)
        ax1.add_patch(c)
        ax1.text(0.7, y, f"$x_{i+1}$", ha="center", va="center", fontsize=9)
        arrow(ax1, 1.0, y, 2.6, 2.0 + (y - 2.0) * 0.12)
        ax1.text(1.75, 2.06 + (y - 2.0) * 0.56, f"$w_{i+1}$", fontsize=8, color="#555555")
    c = Circle((2.95, 2.0), 0.42, fc=BLUE, ec=EDGE, lw=1.2)
    ax1.add_patch(c)
    ax1.text(2.95, 2.0, "$\\Sigma$", ha="center", va="center", fontsize=11)
    ax1.text(2.95, 1.2, "$+\\,b$", ha="center", fontsize=8, color="#555555")
    arrow(ax1, 3.4, 2.0, 4.1, 2.0)
    c = Circle((4.5, 2.0), 0.42, fc=ORANGE, ec=EDGE, lw=1.2)
    ax1.add_patch(c)
    ax1.text(4.5, 2.0, "$\\sigma$", ha="center", va="center", fontsize=11)
    arrow(ax1, 4.95, 2.0, 5.7, 2.0)
    ax1.text(5.85, 2.0, "$y$", fontsize=10, va="center")
    ax1.set_title("Artificial neuron", fontsize=9.5)
    # right: feedforward network
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 4)
    L = {0: [3.3, 2.0, 0.7], 1: [3.6, 2.6, 1.4, 0.4], 2: [2.6, 1.4]}
    X = {0: 0.8, 1: 2.9, 2: 5.0}
    for l in range(2):
        for ya in L[l]:
            for yb in L[l + 1]:
                ax2.plot([X[l], X[l + 1]], [ya, yb], color="#bbbbbb", lw=0.7, zorder=1)
    names = {0: "$x$", 1: "$h$", 2: "$y$"}
    cols = {0: "white", 1: BLUE, 2: ORANGE}
    for l, ys in L.items():
        for k, y in enumerate(ys):
            c = Circle((X[l], y), 0.24, fc=cols[l], ec=EDGE, lw=1.0, zorder=2)
            ax2.add_patch(c)
    ax2.text(0.8, 3.75, "input", ha="center", fontsize=8)
    ax2.text(2.9, 3.95, "hidden layer", ha="center", fontsize=8)
    ax2.text(5.0, 3.2, "output", ha="center", fontsize=8)
    ax2.text(1.85, 0.35, "$W$", ha="center", fontsize=9, color="#555555")
    ax2.text(3.95, 0.35, "$U$", ha="center", fontsize=9, color="#555555")
    ax2.set_title("Feedforward network", fontsize=9.5)
    save(fig, "fig_neuron")


# -------------------------------------------------------------- tokenization
def fig_tokenization():
    fig, ax = plt.subplots(figsize=(8.4, 1.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.text(0.1, 2.25, "Input:", fontsize=9, fontweight="bold")
    ax.text(1.25, 2.25, "expr: x_7 + exp(C*x_2**C) + C", fontsize=9, family="monospace")
    toks = ["expr", ":", "Ġx", "_", "7", "Ġ+", "Ġexp", "(", "C", "*",
            "x", "_", "2", "**", "C", ")", "Ġ+", "ĠC"]
    colors = ["#dbe9f6", "#e2f0d9", "#fbe5d6", "#fff2cc", "#e6e0ec"]
    x = 0.1
    for i, t in enumerate(toks):
        disp = t.replace("Ġ", "␣")  # visible space marker
        w = 0.28 + 0.145 * len(disp)
        box(ax, x, 0.9, w, 0.55, disp, fc=colors[i % len(colors)], fontsize=8, ec="#999999")
        x += w + 0.08
    ax.text(0.1, 0.45, "18 sub-word tokens — variables like x_2 split into (x, _, 2), "
                       "so unseen indices (e.g. x_99) remain generable",
            fontsize=8, style="italic", color="#444444")
    save(fig, "fig_tokenization")


# ------------------------------------------------------------------- RL loop
def fig_rl_loop():
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    box(ax, 0.2, 3.3, 2.6, 0.75, "prompt from dataset features\nvars / oper / cons", fc=GRAY, fontsize=8)
    box(ax, 3.7, 3.3, 2.6, 0.75, "policy $\\pi_\\theta$ (GPT-2 + SFT)\nsamples batch of 1024", fc=BLUE, fontsize=8)
    box(ax, 7.2, 3.3, 2.6, 0.75, "candidate expressions\n$e_1, \\ldots, e_{1024}$", fc="white", fontsize=8)
    box(ax, 7.2, 1.9, 2.6, 0.75, "evaluation (SymPy, C=1)\n$R^2 \\rightarrow$ reward + penalties", fc=GREEN, fontsize=8)
    box(ax, 3.7, 1.9, 2.6, 0.75, "policy update\nPPO / GRPO", fc=ORANGE, fontsize=8)
    box(ax, 3.7, 0.35, 2.6, 0.85, "elite buffer (top-$K$ by $R^2$)\nre-injects 20% of batch\n(BoN variants only)", fc="#fff2cc", fontsize=7.5)
    box(ax, 0.2, 1.9, 2.6, 0.75, "best expression so far\n(early stop at $R^2 \\geq 0.999$)", fc=GRAY, fontsize=8)
    arrow(ax, 2.82, 3.67, 3.68, 3.67)
    arrow(ax, 6.32, 3.67, 7.18, 3.67)
    arrow(ax, 8.5, 3.28, 8.5, 2.67)
    arrow(ax, 7.18, 2.27, 6.32, 2.27)
    arrow(ax, 4.9, 2.7, 4.9, 3.28)          # update -> policy
    arrow(ax, 7.6, 1.88, 5.6, 1.22, connectionstyle="arc3,rad=0.18")  # eval -> buffer
    arrow(ax, 4.6, 1.22, 4.6, 1.88)          # buffer -> update
    arrow(ax, 3.68, 2.27, 2.82, 2.27)
    ax.text(5.0, 4.25, "one training step (repeated up to 200 steps)", fontsize=9,
            style="italic", ha="center", color="#333333")
    save(fig, "fig_rl_loop")


if __name__ == "__main__":
    fig_overview()
    fig_data_engineering()
    fig_genetic_tree()
    fig_neuron()
    fig_tokenization()
    fig_rl_loop()
    print("done ->", OUT)
