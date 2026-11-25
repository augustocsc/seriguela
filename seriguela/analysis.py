"""Camada 5 (R2c): análise estatística pré-registrada, teto de expressividade e relatório.

Segue estritamente ``docs/reports/PLANO_V3_2026-11.md`` (§5.5, §6.3, §6.4, §6.5):
- Unidade de inferência = o PROBLEMA (cluster), com seeds PAREADAS entre os braços.
- Contrastes pré-definidos:
  * Família primária (corrigida por Holm-Bonferroni):
    - ``H1_ppo``: ``ppo+elite - ppo`` (replay ajuda no sinal de média?)
    - ``H1_rspg``: ``rspg+elite - rspg`` (replay ajuda na busca de risco/cauda?)
    - ``H2_interaction``: ``(rspg+elite - rspg) - (ppo+elite - ppo)`` (interação sinal × replay)
    - ``H3_vs_bon``: braço campeão (definido no G2) contra ``bon``
    - ``H4_ppo_mechanism``: ``ppo+elite - ppo+random`` (elite vs. replay aleatório)
    - ``H4_rspg_mechanism``: ``rspg+elite - rspg+random`` (quando ``rspg+random`` está presente)
  * Secundário (relatado em separado):
    - ``H5_anchor``: ``grpo - ppo`` (KL contra o prior do SFT vs. bônus de entropia)
- Estimação: diferença média de recuperação por problema, IC 95% por cluster-bootstrap
  sobre problemas (preservando o pareamento das seeds), teste exato/Monte-Carlo de
  permutação de sinais pareados por problema, e razão de chances estratificada de
  Mantel-Haenszel (escala logística condicional por problema, imune a separação completa).
- Regra de interpretação: ``p_holm >= alpha`` NUNCA é lido como equivalência, e sim
  como ``inconclusive`` acompanhado dos limites do IC 95%.
"""
from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import collect
from .data import BENCH_DIR, NGUYEN_PROBLEMS, _NGUYEN, srbench_truth
from .policy import OPERATORS


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    weights: Dict[str, float]      # soma dos pesos = 0
    family: str                    # "primary" (entra em Holm) ou "secondary"
    description: str


@dataclass
class ContrastResult:
    name: str
    family: str
    description: str
    n_problems: int
    n_paired_cells: int
    estimate: float                # diferença na taxa (escala [-1, 1])
    ci_low: float
    ci_high: float
    odds_ratio_mh: Optional[float] # razão de chances estratificada por problema (quando contraste 1x1)
    p_raw: float
    p_holm: Optional[float]
    verdict: str                   # "positive", "negative", "inconclusive"


def default_contrasts(available_arms: Sequence[str], champion_arm: str = "ppo+elite") -> List[ContrastSpec]:
    """Devolve os contrastes pré-registrados cujos braços existem na tabela."""
    arms = set(available_arms)
    candidates = [
        ContrastSpec("H1_ppo", {"ppo+elite": 1.0, "ppo": -1.0}, "primary",
                     "Elite replay vs. sem replay em PPO (z-score)"),
        ContrastSpec("H1_rspg", {"rspg+elite": 1.0, "rspg": -1.0}, "primary",
                     "Elite replay vs. sem replay em RSPG (busca de risco)"),
        ContrastSpec("H2_interaction",
                     {"rspg+elite": 1.0, "rspg": -1.0, "ppo+elite": -1.0, "ppo": 1.0},
                     "primary", "Interação sinal (cauda vs. média) × replay de elite"),
        ContrastSpec("H3_vs_bon", {champion_arm: 1.0, "bon": -1.0}, "primary",
                     f"Campeão pré-especificado ({champion_arm}) vs. Best-of-N"),
        ContrastSpec("H4_ppo_mechanism", {"ppo+elite": 1.0, "ppo+random": -1.0}, "primary",
                     "Replay de elite vs. replay aleatório em PPO"),
        ContrastSpec("H4_rspg_mechanism", {"rspg+elite": 1.0, "rspg+random": -1.0}, "primary",
                     "Replay de elite vs. replay aleatório em RSPG"),
        ContrastSpec("H5_anchor", {"grpo": 1.0, "ppo": -1.0}, "secondary",
                     "Âncora KL ao prior (GRPO) vs. entropia (PPO)"),
    ]
    return [c for c in candidates if set(c.weights).issubset(arms) and len(c.weights) >= 2]


def _problem_contrast_vector(frame: pd.DataFrame, weights: Mapping[str, float],
                             metric: str) -> Tuple[np.ndarray, int]:
    """Vetor ``d_p`` (um valor por problema) pareando estritamente as seeds comuns."""
    needed = list(weights.keys())
    sub = frame[frame["arm"].isin(needed)].copy()
    sub["_val"] = sub[metric].astype(float)
    diffs: List[float] = []
    paired_cells = 0
    for _, grp in sub.groupby("problem", sort=True):
        piv = grp.pivot_table(index="seed", columns="arm", values="_val", aggfunc="first")
        if not set(needed).issubset(piv.columns):
            continue
        piv = piv[needed].dropna()
        if len(piv) == 0:
            continue
        paired_cells += len(piv)
        per_arm_mean = piv.mean(axis=0)
        diffs.append(float(sum(w * per_arm_mean[a] for a, w in weights.items())))
    return np.asarray(diffs, dtype=np.float64), paired_cells


def _mantel_haenszel_or(frame: pd.DataFrame, arm_pos: str, arm_neg: str, metric: str) -> Optional[float]:
    """Razão de chances estratificada por problema (escala logística condicional)."""
    sub = frame[frame["arm"].isin([arm_pos, arm_neg])].copy()
    sub["_y"] = (sub[metric].astype(float) > 0.5).astype(int)
    num = den = 0.0
    discordant_strata = 0
    for _, grp in sub.groupby("problem", sort=True):
        piv = grp.pivot_table(index="seed", columns="arm", values="_y", aggfunc="first")
        if not {arm_pos, arm_neg}.issubset(piv.columns):
            continue
        piv = piv[[arm_pos, arm_neg]].dropna()
        n = len(piv)
        if n == 0:
            continue
        a = float(piv[arm_pos].sum())
        b = float(piv[arm_neg].sum())
        total = 2.0 * n
        if (a == 0 and b == 0) or (a == n and b == n):
            continue
        discordant_strata += 1
        num += a * (n - b) / total
        den += b * (n - a) / total
    if discordant_strata == 0:
        return None
    # Estabilização de Haldane-Anscombe (0.5/total) se um lado nunca venceu em estrato discordante
    if num == 0.0 or den == 0.0:
        num += 0.25
        den += 0.25
    return round(float(num / den), 4)


def _paired_signflip_pvalue(diffs: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Teste bilateral de permutação de sinais pareados sobre os problemas."""
    m = len(diffs)
    if m == 0 or np.all(diffs == 0):
        return 1.0
    obs = abs(float(diffs.mean()))
    if m <= 15:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=m)), dtype=np.float64)
        means = np.abs((signs * diffs[None, :]).mean(axis=1))
        return float(np.mean(means >= obs - 1e-15))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, m))
    means = np.abs((signs * diffs[None, :]).mean(axis=1))
    # Inclui a identidade observada (+1 no numerador e denominador: Phipson & Smyth, 2010)
    return float((np.sum(means >= obs - 1e-15) + 1) / (n_perm + 1))


def _cluster_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                          seed: int = 0) -> Tuple[float, float]:
    """IC bilateral por reamostragem de problemas (clusters) com reposição."""
    m = len(diffs)
    if m == 0:
        return (float("nan"), float("nan"))
    if m == 1 or np.allclose(diffs, diffs[0]):
        v = float(diffs[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(lo), float(hi)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Correção step-down de Holm-Bonferroni preservando a ordem original."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_values[idx]) * (m - rank))
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted.tolist()


def evaluate_contrasts(frame: pd.DataFrame, metric: str = "solved_symbolic",
                       champion_arm: str = "ppo+elite", alpha: float = 0.05,
                       n_boot: int = 10000, n_perm: int = 20000, seed: int = 0,
                       specs: Optional[Sequence[ContrastSpec]] = None) -> List[ContrastResult]:
    """Executa todos os contrastes pré-registrados sobre ``frame``."""
    work = frame.copy()
    if metric == "solved_symbolic":
        work[metric] = work[metric].map(lambda v: 1.0 if v is True else 0.0)
    specs = list(specs) if specs is not None else default_contrasts(sorted(work["arm"].unique()), champion_arm)
    raw_results: List[ContrastResult] = []
    for i, spec in enumerate(specs):
        diffs, n_cells = _problem_contrast_vector(work, spec.weights, metric)
        est = float(diffs.mean()) if len(diffs) else float("nan")
        lo, hi = _cluster_bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed + i + 1)
        p_raw = _paired_signflip_pvalue(diffs, n_perm=n_perm, seed=seed + 100 + i)
        mh_or = None
        if len(spec.weights) == 2 and sorted(spec.weights.values()) == [-1.0, 1.0] \
                and metric in ("solved_symbolic", "solved_accuracy"):
            pos = next(a for a, w in spec.weights.items() if w > 0)
            neg = next(a for a, w in spec.weights.items() if w < 0)
            mh_or = _mantel_haenszel_or(work, pos, neg, metric)
        raw_results.append(ContrastResult(
            name=spec.name, family=spec.family, description=spec.description,
            n_problems=len(diffs), n_paired_cells=n_cells,
            estimate=round(est, 6), ci_low=round(lo, 6), ci_high=round(hi, 6),
            odds_ratio_mh=mh_or, p_raw=round(p_raw, 6), p_holm=None, verdict="inconclusive",
        ))

    prim_idx = [i for i, r in enumerate(raw_results) if r.family == "primary"]
    prim_holm = holm_adjust([raw_results[i].p_raw for i in prim_idx])
    for idx, p_adj in zip(prim_idx, prim_holm):
        raw_results[idx].p_holm = round(p_adj, 6)

    for r in raw_results:
        p_ref = r.p_holm if r.family == "primary" else r.p_raw
        if p_ref is not None and p_ref < alpha and r.estimate > 0:
            r.verdict = "positive"
        elif p_ref is not None and p_ref < alpha and r.estimate < 0:
            r.verdict = "negative"
        else:
            r.verdict = "inconclusive (not equivalence)"
    return raw_results


def required_problems(pilot_diffs: Sequence[float], target_delta: float,
                      power: float = 0.80, alpha: float = 0.05) -> int:
    """Tamanho do painel confirmatório M a partir da variância entre problemas no G2."""
    from scipy.stats import norm
    d = np.asarray(pilot_diffs, dtype=np.float64)
    if target_delta <= 0:
        raise ValueError("target_delta deve ser positivo")
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.25
    sd = max(sd, 1e-6)
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    return int(math.ceil(((z_a + z_b) * sd / target_delta) ** 2))


# ------------------------------------------------------------ teto de expressividade (§6.5)

_ALGEBRAIC_REWRITES = {"cot", "tanh", "sinh", "cosh"}  # redutíveis a sin/cos/exp em OPERATORS
_ALLOWED_OPS = set(OPERATORS)


def classify_expressivity(truth: str) -> str:
    """Classifica se ``truth`` é representável com ``OPERATORS`` sob ``C=1`` ou ``fit``."""
    funcs = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", truth))
    if not funcs.issubset(_ALLOWED_OPS | _ALGEBRAIC_REWRITES):
        return "out_of_vocabulary"
    has_non_int_const = bool(re.search(r"\b\d+\.\d+\b|\bpi\b", truth))
    # Inteiros pequenos podem ser somas repetidas, mas >= 5 ou racionais exigem C ajustável na prática
    big_ints = [int(x) for x in re.findall(r"(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_.])", truth)]
    # Ignora expoentes inteiros pequenos (x_1**2..6)
    exponents = [int(x) for x in re.findall(r"\*\*\s*(\d+)", truth)]
    rem = list(big_ints)
    for e in exponents:
        if e in rem:
            rem.remove(e)
    if has_non_int_const or any(v > 4 for v in rem) or (funcs & _ALGEBRAIC_REWRITES):
        return "exact_with_fit"
    return "exact_c1"


def expressivity_ceiling() -> Dict[str, Dict]:
    """Teto de expressividade por família sobre todos os benchmarks cadastrados."""
    items: List[Tuple[str, str, str]] = []
    for name in NGUYEN_PROBLEMS:
        items.append(("nguyen", name, _NGUYEN[name][0]))
    for fam in ("feynman", "strogatz"):
        for meta in sorted((BENCH_DIR / fam).glob("*.meta.json")):
            d = json.loads(meta.read_text(encoding="utf-8"))
            cols = [f["name"] for f in d.get("pmlb_metadata", {}).get("features", [])] or d["variables"]
            t = srbench_truth(d["name"], fam, cols)
            if t:
                items.append((fam, d["name"], t))
    summary: Dict[str, Dict] = {}
    for fam, name, truth in items:
        cat = classify_expressivity(truth)
        bucket = summary.setdefault(fam, {"total": 0, "exact_c1": 0, "exact_with_fit": 0, "out_of_vocabulary": 0})
        bucket["total"] += 1
        bucket[cat] += 1
    return summary


# ------------------------------------------------------------ relatórios Markdown e HTML

def render_markdown(frame: pd.DataFrame, contrasts: Sequence[ContrastResult],
                    ceiling: Optional[Dict] = None) -> str:
    lines = ["# Relatório Pré-Registrado de Campanha (Seriguela v3)", ""]
    commits = sorted(frame["commit"].unique()) if len(frame) else []
    lines.append(f"- **Runs válidos:** {len(frame)} · **Problemas:** {frame['problem'].nunique() if len(frame) else 0} "
                 f"· **Commits:** `{', '.join(c[:8] for c in commits) or 'nenhum'}`")
    lines.append("")
    if len(frame):
        lines += ["## 1. Resumo por Problema e Braço", "",
                  collect.summary(frame).to_markdown(index=False) if hasattr(pd.DataFrame, "to_markdown") else
                  "```\n" + collect.summary(frame).to_string(index=False) + "\n```", ""]
    lines += ["## 2. Contrastes Pré-Registrados (Unidade = Problema, Seeds Pareadas)", "",
              "| Hipótese | Família | Problemas | Δ Taxa | IC 95% (Bootstrap) | OR (MH) | p bruto | p Holm | Veredito |",
              "|---|---|---:|---:|---|---:|---:|---:|---|"]
    for c in contrasts:
        p_h = f"{c.p_holm:.4f}" if c.p_holm is not None else "—"
        or_s = f"{c.odds_ratio_mh:.2f}" if c.odds_ratio_mh is not None else "—"
        lines.append(
            f"| `{c.name}` | {c.family} | {c.n_problems} | {c.estimate:+.4f} | "
            f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}] | {or_s} | {c.p_raw:.4f} | {p_h} | **{c.verdict}** |"
        )
    lines.append("")
    if ceiling:
        lines += ["## 3. Teto de Expressividade dos Benchmarks", "",
                  "| Família | Total | Exata com C=1 | Exata com Fit (L-BFGS-B) | Fora do Vocabulário |",
                  "|---|---:|---:|---:|---:|"]
        for fam, b in ceiling.items():
            lines.append(f"| {fam} | {b['total']} | {b['exact_c1']} | {b['exact_with_fit']} | {b['out_of_vocabulary']} |")
    return "\n".join(lines) + "\n"


def render_html(md_text: str, title: str = "Relatório Seriguela v3") -> str:
    return (
        f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "pre{background:#f6f8fa;padding:1rem;overflow-x:auto;border-radius:6px}</style></head>"
        f"<body><pre>{html.escape(md_text)}</pre></body></html>\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="diretório com resultados v3 (ex.: results/v3/g3)")
    parser.add_argument("--metric", default="solved_symbolic", choices=["solved_symbolic", "solved_accuracy", "test_r2"])
    parser.add_argument("--champion", default="ppo+elite")
    parser.add_argument("--allow_mixed", action="store_true")
    parser.add_argument("--md", help="caminho para salvar relatório Markdown")
    parser.add_argument("--html", help="caminho para salvar relatório HTML")
    parser.add_argument("--json", help="caminho para salvar JSON estruturado")
    args = parser.parse_args(argv)

    frame = collect.load(Path(args.root), allow_mixed=args.allow_mixed)
    contrasts = evaluate_contrasts(frame, metric=args.metric, champion_arm=args.champion)
    ceiling = expressivity_ceiling()
    md = render_markdown(frame, contrasts, ceiling)
    print(md)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
    if args.html:
        Path(args.html).write_text(render_html(md), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "contrasts": [asdict(c) for c in contrasts],
            "expressivity_ceiling": ceiling,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
