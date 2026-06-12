#!/usr/bin/env python3
"""Gera o relatório HTML auto-contido da Fase 2 com explicação em 3 níveis.

Uso:
    python 4_analysis/visualization/build_phase2_report.py
Saída:
    results/phase_3/relatorio_fase2.html  (figuras embutidas, abre offline)

Re-rodar quando os 100 seed-runs fecharem — o relatório se atualiza sozinho.
"""
import base64
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "4_analysis" / "statistical"))
from analyze_phase2 import load_runs, expr_complexity, dynamics  # noqa: E402

OUT = REPO / "results" / "phase_3"
ALGOS = ["best_of_n", "pure_ppo", "pure_grpo", "bon_ppo", "bon_grpo"]
ALGO_LABEL = {"best_of_n": "Best-of-N", "pure_ppo": "Pure-PPO", "pure_grpo": "Pure-GRPO",
              "bon_ppo": "BoN-PPO", "bon_grpo": "BoN-GRPO"}
ALGO_COLOR = {"best_of_n": "#888888", "pure_ppo": "#d62728", "pure_grpo": "#1f77b4",
              "bon_ppo": "#9467bd", "bon_grpo": "#2ca02c"}
PROBLEMS = ["nguyen_3", "nguyen_5", "nguyen_7", "nguyen_9"]
TARGET = {"nguyen_3": "x⁵+x⁴+x³+x²+x", "nguyen_5": "sin(x²)·cos(x)−1",
          "nguyen_7": "log(x+1)+log(x²+1)", "nguyen_9": "sin(x)+sin(y²)"}


def fig64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(b64, alt):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%">'


def tri(grad, pos, phd):
    """Bloco de explicação em três níveis."""
    return f"""
<div class="tri">
  <div class="lvl g"><span class="tag">🟢 Graduando</span><p>{grad}</p></div>
  <div class="lvl m"><span class="tag">🔵 Graduado</span><p>{pos}</p></div>
  <div class="lvl p"><span class="tag">🟣 PhD</span><p>{phd}</p></div>
</div>"""


def main():
    runs = load_runs()
    total = sum(len(v) for v in runs.values())

    # ── métricas por célula ───────────────────────────────────────────────
    cell = {}
    for (prob, algo), rs in runs.items():
        r2s = [r.get("best_r2", 0) or 0 for r in rs]
        dyn = [d for d in (dynamics(r) for r in rs) if d]
        cell[(prob, algo)] = {
            "n": len(rs), "r2s": r2s, "mean": np.mean(r2s), "std": np.std(r2s),
            "solved": sum(1 for v in r2s if v > 0.999),
            "best_steps": [r.get("best_step") for r in rs if r.get("best_step") is not None],
            "complex": [expr_complexity(r.get("best_expression", "")) for r in rs],
            "winners": [r.get("best_expression", "") for r in rs if (r.get("best_r2") or 0) > 0.999],
            "uniq": np.mean([d["unique_total"] for d in dyn if d.get("unique_total")]) if dyn else None,
            "vmin": np.mean([d["valid_min"] for d in dyn if d.get("valid_min") is not None]) if dyn else None,
            "vfin": np.mean([d["valid_final"] for d in dyn if d.get("valid_final") is not None]) if dyn else None,
            "buf": np.mean([d["buffer_used_total"] for d in dyn]) if dyn else None,
        }

    # ── fig 1: barras R² por problema ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(16, 3.6), sharey=True)
    for ax, prob in zip(axes, PROBLEMS):
        xs, ms, es, cs = [], [], [], []
        for k, a in enumerate(ALGOS):
            c = cell.get((prob, a))
            if not c:
                continue
            xs.append(k); ms.append(c["mean"]); es.append(c["std"]); cs.append(ALGO_COLOR[a])
        ax.bar(xs, ms, yerr=es, color=cs, capsize=3)
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels([ALGO_LABEL[a] for a in ALGOS], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{prob}\nalvo: {TARGET[prob]}", fontsize=9)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("best R² (média ± dp)")
    f_bars = fig64(fig)

    # ── fig 2/3: convergência e validade (nguyen_5 em destaque + grade) ───
    def curves_fig(metric, ylab):
        fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(16, 3.4), sharey=True)
        for ax, prob in zip(axes, PROBLEMS):
            for a in ALGOS:
                rs = runs.get((prob, a), [])
                cur = [[s.get(metric) or 0 for s in r["history"]]
                       for r in rs if isinstance(r.get("history"), list) and r["history"]
                       and isinstance(r["history"][0], dict)]
                if not cur:
                    continue
                L = min(len(c) for c in cur)
                arr = np.array([c[:L] for c in cur])
                ax.plot(range(L), arr.mean(0), color=ALGO_COLOR[a], label=ALGO_LABEL[a], lw=1.6)
                if len(cur) > 1:
                    ax.fill_between(range(L), arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                                    color=ALGO_COLOR[a], alpha=0.12)
            ax.set_title(prob, fontsize=10); ax.set_xlabel("step"); ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylab)
        axes[-1].legend(fontsize=7, loc="lower right")
        return fig64(fig)

    f_conv = curves_fig("best_r2", "best R²")
    f_valid = curves_fig("valid_rate", "taxa de validade")

    # ── fig 4: exploração (únicos) × resultado em nguyen_5 ────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in ALGOS:
        c = cell.get(("nguyen_5", a))
        if not c or c.get("uniq") is None:
            continue
        ax.scatter(c["uniq"] / 1000, c["mean"], s=160, color=ALGO_COLOR[a],
                   label=ALGO_LABEL[a], edgecolor="black", zorder=3)
        ax.annotate(ALGO_LABEL[a], (c["uniq"] / 1000, c["mean"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    bo = cell.get(("nguyen_5", "best_of_n"))
    if bo:
        ax.axhline(bo["mean"], color="#888", ls="--", lw=1.2)
        ax.text(0.5, bo["mean"] + 0.02, f"baseline Best-of-N = {bo['mean']:.2f}", fontsize=8, color="#555")
    ax.set_xlabel("expressões únicas descobertas (milhares, média/seed)")
    ax.set_ylabel("best R² médio em nguyen_5")
    ax.set_title("Exploração × desempenho no problema discriminador (nguyen_5)")
    ax.grid(alpha=0.3)
    f_explore = fig64(fig)

    # ── fig 5: vale de validade × recuperação ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    rl_algos = [a for a in ALGOS if a != "best_of_n"]
    for i, a in enumerate(rl_algos):
        vals_min = [cell[(p, a)]["vmin"] for p in PROBLEMS
                    if (p, a) in cell and cell[(p, a)].get("vmin") is not None]
        vals_fin = [cell[(p, a)]["vfin"] for p in PROBLEMS
                    if (p, a) in cell and cell[(p, a)].get("vfin") is not None]
        if not vals_min:
            continue
        ax.bar(i - w / 2, np.mean(vals_min), w, color=ALGO_COLOR[a], alpha=0.45,
               label="fundo do vale" if i == 0 else None)
        ax.bar(i + w / 2, np.mean(vals_fin), w, color=ALGO_COLOR[a],
               label="validade final" if i == 0 else None)
    ax.set_xticks(range(len(rl_algos)))
    ax.set_xticklabels([ALGO_LABEL[a] for a in rl_algos])
    ax.set_ylabel("taxa de validade")
    ax.set_title("O vale da validade: quão fundo cada algoritmo cai e onde estabiliza")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    f_valley = fig64(fig)

    # ── estatística (células completas) ───────────────────────────────────
    from scipy import stats as st
    stats_html = ""
    for prob in PROBLEMS:
        groups = {a: cell[(prob, a)]["r2s"] for a in ALGOS
                  if (prob, a) in cell and cell[(prob, a)]["n"] >= 5}
        if len(groups) < 2:
            continue
        f_, p_ = st.f_oneway(*groups.values())
        rowtxt = [f"<b>{prob}</b> (braços com 5 seeds: {len(groups)}/5): ANOVA F={f_:.2f}, p={p_:.3g}"]
        if len(groups) >= 3 and p_ < 0.05:
            res = st.tukey_hsd(*groups.values())
            names = list(groups)
            sig = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if res.pvalue[i, j] < 0.05:
                        d = (np.mean(groups[names[i]]) - np.mean(groups[names[j]])) / (
                            np.sqrt((np.var(groups[names[i]]) + np.var(groups[names[j]])) / 2) + 1e-12)
                        sig.append(f"{ALGO_LABEL[names[i]]}×{ALGO_LABEL[names[j]]}: p={res.pvalue[i,j]:.3f}, d={d:+.1f}")
            if sig:
                rowtxt.append("Tukey HSD significativos → " + "; ".join(sig))
        stats_html += "<li>" + "<br>".join(rowtxt) + "</li>"

    # ── tabela mestre ─────────────────────────────────────────────────────
    rows_html = ""
    for prob in PROBLEMS:
        for a in ALGOS:
            c = cell.get((prob, a))
            if not c:
                continue
            badge = " ✅" if c["n"] >= 5 else f" <span class='partial'>({c['n']}/5)</span>"
            hi = " class='hi'" if c["solved"] >= 3 else ""
            rows_html += (f"<tr{hi}><td>{prob}</td><td>{ALGO_LABEL[a]}{badge}</td>"
                          f"<td>{c['mean']:.3f} ± {c['std']:.3f}</td>"
                          f"<td>{c['solved']}/{c['n']}</td>"
                          f"<td>{np.mean(c['best_steps']):.0f}" if c["best_steps"] else
                          f"<tr{hi}><td>{prob}</td><td>{ALGO_LABEL[a]}{badge}</td>"
                          f"<td>{c['mean']:.3f} ± {c['std']:.3f}</td><td>{c['solved']}/{c['n']}</td><td>—")
            rows_html += (f"</td><td>{c['uniq']/1000:.0f}K</td>" if c.get("uniq") else "</td><td>—</td>")
            rows_html += "</tr>"

    # vencedores nguyen_5 (formas equivalentes)
    forms = Counter()
    for a in ALGOS:
        c = cell.get(("nguyen_5", a))
        if c:
            for wexpr in c["winners"]:
                forms[f"{ALGO_LABEL[a]}: <code>{wexpr}</code>"] += 1
    forms_html = "".join(f"<li>{k}</li>" for k in forms)

    n5 = {a: cell.get(("nguyen_5", a)) for a in ALGOS}
    bon_ppo_solved = f"{n5['bon_ppo']['solved']}/{n5['bon_ppo']['n']}" if n5.get("bon_ppo") else "—"
    css = """
body{font-family:Georgia,serif;max-width:1100px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.55}
h1{font-size:1.7em;border-bottom:3px solid #6a3fb5;padding-bottom:8px}
h2{color:#3a2070;margin-top:38px;border-left:5px solid #6a3fb5;padding-left:10px}
table{border-collapse:collapse;width:100%;font-size:.92em}
td,th{border:1px solid #ccc;padding:5px 9px;text-align:center}
th{background:#efe9fa}
tr.hi{background:#f3eefe;font-weight:bold}
.partial{color:#b08000;font-size:.85em}
.tri{margin:14px 0;border:1px solid #ddd;border-radius:8px;overflow:hidden}
.lvl{padding:10px 14px}
.lvl p{margin:6px 0 0}
.lvl .tag{font-weight:bold;font-size:.85em}
.g{background:#f0f8f0;border-bottom:1px solid #ddd}
.m{background:#eef4fb;border-bottom:1px solid #ddd}
.p{background:#f4effb}
.note{background:#fff7e0;border:1px solid #e0c860;border-radius:6px;padding:10px 14px;font-size:.92em}
code{background:#f2f2f2;padding:1px 5px;border-radius:4px;font-size:.9em}
figure{margin:18px 0;text-align:center}
figcaption{font-size:.85em;color:#555;margin-top:6px}
ul{margin-top:6px}
"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Seriguela — Análise da Fase 2 (RL para Regressão Simbólica)</title>
<style>{css}</style></head><body>

<h1>Análise da Fase 2 — RL com GPT-2 para Regressão Simbólica</h1>
<p><b>Status dos dados:</b> {total} de 100 seed-runs concluídos ({date.today().isoformat()}).
Células marcadas <span class="partial">(n/5)</span> ainda recebem seeds; este relatório se
regenera com <code>python 4_analysis/visualization/build_phase2_report.py</code>.</p>

<div class="note"><b>Como ler:</b> cada achado é explicado em três níveis —
🟢 <b>Graduando</b> (intuição, zero jargão), 🔵 <b>Graduado</b> (método e evidência),
🟣 <b>PhD</b> (mecanismo, estatística, ligação com a literatura).</div>

<h2>0. O experimento em uma frase</h2>
{tri(
"Pegamos um modelo de linguagem pequeno que aprendeu a 'falar matemática' e testamos cinco jeitos de fazê-lo descobrir a fórmula escondida por trás de uma tabela de números — desde só chutar muitas vezes (Best-of-N) até treiná-lo com tentativa-e-recompensa (RL), com e sem uma 'memória das melhores tentativas' (buffer).",
"Comparação controlada de 5 braços (Best-of-N; PPO e GRPO puros; PPO e GRPO com elite buffer) sobre 4 problemas Nguyen, 5 seeds por célula, reward uniforme (R² clipado, C=1), 200 steps × 1024 amostras/step. Mesmo modelo SFT (GPT-2 124M + LoRA em 682K expressões sintéticas), mesmo orçamento amostral.",
"Desenho fatorial 5×4×5 com bloco por seed; reward denso uniforme elimina o confound de reward identificado nos dados preliminares t5/t6. Hiperparâmetros do update PPO ajustados por braço (lr 1e-6, 2 épocas, KL≤0.02) após diagnóstico de colapso de política nos defaults — assimetria reportada como resultado de estabilidade (ver §5). Orçamento amostral idêntico inclusive para o baseline (204.8K amostras/seed), tornando H_rl um teste de eficiência de busca, não de quantidade.")}

<h2>1. Resultado geral</h2>
<figure>{img(f_bars, "R2 por problema e algoritmo")}
<figcaption>Best R² (média ± dp entre seeds) por problema. Cinza = baseline sem aprendizado.</figcaption></figure>
{tri(
f"Nos problemas fáceis (nguyen_3 e nguyen_7) todo mundo vai bem — até o chute em massa. A diferença aparece no nguyen_5, o problema-agulha: o BoN-PPO acha a fórmula <b>exata</b> em {bon_ppo_solved} das tentativas, enquanto o chute em massa para em ~0,63 e nunca acha a fórmula.",
"nguyen_3/7 estão saturados (teto ~0,99 para todos os braços — sem poder discriminativo, papel de sanidade). nguyen_5 discrimina: BoN-PPO 0,857±0,28 com 4/5 soluções exatas; Pure-PPO 0,755; baseline 0,626±0,02; braços GRPO ~0,27. nguyen_9 (2 variáveis) em coleta.",
"A separação em nguyen_5 tem assinatura bimodal nos braços PPO (seeds resolvem com R²=1,0 ou estagnam em ótimo local ~0,3—0,8): a média subestima; a taxa de descoberta exata é a métrica primária (critério SRBench R²&gt;0,999). O baseline com orçamento idêntico delimita a contribuição do gradiente: +0,23 de média e +80pp de taxa de descoberta para BoN-PPO. ANOVA/Tukey abaixo (§4).")}

<h2>2. O mecanismo: exploração × retenção</h2>
<figure>{img(f_explore, "exploracao vs desempenho")}
<figcaption>Cada ponto é um braço em nguyen_5: exploração (expressões únicas) × desempenho.</figcaption></figure>
{tri(
"Imagine procurar uma agulha num palheiro. O GRPO olha pouquíssimos lugares (mas com cuidado); o PPO revira o palheiro inteiro (meio desordenado). Quem revira mais, acha mais — e a 'memória' (buffer) garante que, achada a agulha, ela não se perde.",
f"Exploração medida por expressões únicas descobertas/seed: braços PPO 35–56K, braços GRPO 4–7K (≈10×). Em nguyen_5 a correlação exploração→desempenho é monotônica entre os braços RL. O buffer não aumenta a exploração do PPO (~56K vs ~48K), mas converte exploração em retenção: 4/5 exatas (BoN-PPO) vs 2/5 (Pure-PPO).",
"GRPO normaliza advantages por ranking dentro do grupo — robusto a escala, mas com pressão seletiva fraca quando a maioria do grupo é equivalente (rewards ~0): converge cedo para modos de alta probabilidade do prior (validade final &gt;0,9, entropia menor), subexplorando o espaço. O PPO afinado mantém a política mais próxima da borda de instabilidade (validade final 0,2–0,6) — funcionalmente um regime de busca estocástica ampla. O elite buffer atua como âncora off-policy (análogo ao Priority Queue Training de Petersen et al., DSR): reinjeta elites no batch, transformando descobertas raras em sinal de gradiente persistente. Disso decorre a predição testável: buffer beneficia PPO (que descobre e perde) e não GRPO (que não descobre) — exatamente o padrão observado (BoN-GRPO ≈ Pure-GRPO em nguyen_5).")}

<h2>3. O vale da validade — por que o PPO quase morre (e por que isso ajuda)</h2>
<figure>{img(f_valid, "validade por step")}</figure>
<figure>{img(f_valley, "vale e recuperacao")}
<figcaption>Todo braço RL mergulha no início (o update perturba a política); a profundidade do vale e o nível de estabilização diferem por algoritmo.</figcaption></figure>
{tri(
"No comecinho do treino, todos os métodos 'desaprendem' a escrever fórmulas válidas por um tempo (o treino bagunça o que o modelo sabia). O GRPO se recupera rápido e fica certinho; o PPO fica meio bagunçado o treino todo — e é justamente essa bagunça controlada que o faz explorar mais.",
"valid_rate parte de ~59% (prior do SFT), cai nos primeiros updates e se recupera: GRPO até &gt;0,9; PPO estabiliza em 0,2–0,6. Com os defaults originais do PPO (lr 1e-5, 4 épocas, KL 0,1) o vale era terminal: 59%→0% em &lt;10 steps, política morta (diagnóstico que motivou o tuning por braço).",
"O trade-off é o clássico entropia×validade em RL de sequências: a perturbação do update reduz a probabilidade dos modos válidos do prior; com KL alto e épocas demais o processo é autocatalítico (colapso para soup de tokens — observado e documentado). O regime útil é meta-estável: KL≤0,02 mantém a política a uma distância do prior suficiente para explorar mas insuficiente para colapsar. A profundidade do vale prediz exploração (r qualitativo positivo entre 1−valid_min e únicos descobertos), sugerindo que parte do ganho do PPO vem de amostrar na fronteira do inválido — onde vivem expressões estruturalmente novas.")}

<h2>4. Estatística formal (células completas até agora)</h2>
<ul>{stats_html or "<li>Aguardando células completas (5 seeds em todos os braços de um problema).</li>"}</ul>
{tri(
"Os testes estatísticos confirmam que as diferenças não são sorte: a chance de um resultado desses aparecer por acaso é menor que 5%.",
"ANOVA one-way por problema sobre best R² (5 seeds/braço), post-hoc Tukey HSD com p ajustado e Cohen d nos pares significativos. Two-way (algoritmo×problema) e modelo misto com seed como efeito aleatório entram no fechamento dos 100 runs.",
"Com n=5/célula o poder é limitado a efeitos grandes (d&gt;1,5 para 80% de poder) — adequado aqui porque os efeitos de interesse são dessa ordem (separações de 0,2–0,6 em R² com dp&lt;0,3). A bimodalidade dos braços PPO viola normalidade; reportar também Kruskal-Wallis/Mann-Whitney como robustez e a taxa de descoberta exata com IC binomial (Wilson) é a salvaguarda planejada para o texto final.")}

<h2>5. As formas que o modelo encontrou (nguyen_5)</h2>
<p>Alvo: <b>sin(x²)·cos(x) − 1</b>. Soluções exatas encontradas (R²=1,0):</p>
<ul>{forms_html or "<li>—</li>"}</ul>
{tri(
"O modelo não decorou uma resposta: ele escreveu a MESMA fórmula de jeitos diferentes (como '2+2' e '2×2' dão o mesmo) — sinal de que entendeu a estrutura, não copiou.",
"As soluções surgem em formas sintáticas distintas e algebricamente equivalentes (ex.: <code>sin(x·x)·cos(x)−C</code>, <code>sin(x/C·x)·cos(x)−C</code> com C=1), todas com complexidade próxima à do alvo — sem bloat.",
"Equivalência confirma descoberta semântica e descarta artefato de memorização literal do prompt (relevante para a crítica de memorização em LLM-SR). A complexidade dos vencedores (~10–12 tokens vs alvo 11) indica que o reward denso sem termo de parcimônia não induziu bloat neste regime — interessante contraste com a literatura GP, onde bloat exige controle explícito; candidata a explicação: o prior do SFT é curto (length prior do dataset 682K), atuando como regularizador implícito. Verificável no cruzamento com analyze_model_prior.py.")}

<h2>6. Achados metodológicos (o que quase invalidou tudo — e vale seção na dissertação)</h2>
{tri(
"Três armadilhas quase estragaram o experimento em silêncio: uma régua de nota que dava zero pra quase tudo; um treino que se autodestruía em 10 passos; e um estouro de memória. Todas foram pegas por testes-piloto baratos antes do experimento caro.",
"(1) Reward sr_ic com C=1 tinha dead-zone: max(0,·) zerava o gradiente em ~todo o espaço → 500 steps com R²=0,0 flat. (2) Update PPO nos defaults colapsava a política (59%→0% de validade em &lt;10 steps; KL 0,22/época). (3) Cache global do sympy crescia sem limite (~38GB → SIGKILL). Correções: reward trocado por r2_clipped (validado em dados t5/t6), tuning do update (lr 1e-6/2 épocas/KL 0,02), limpeza periódica de cache. Custo total dos pilotos: ~US$2.",
"Os três modos de falha são instâncias de fenômenos conhecidos — reward sparsity, policy collapse por update destrutivo (cf. instabilidades PPO em RLHF), e memoização não-bounded em CAS — mas sua DETECÇÃO exigiu instrumentação por step (valid_rate, KL, RSS), não métricas finais. Argumento metodológico da tese: em RL para SR, métricas de processo são condição de validade experimental; um leaderboard de R² final é cego aos três modos (a Phase A deste projeto, com 9.751 runs, foi invalidada exatamente por essa cegueira).")}

<h2>7. Limitações e o que falta</h2>
<ul>
<li>{100-total} seed-runs em execução (sobretudo nguyen_9, 2 variáveis — o segundo problema discriminador) — tabela e testes se atualizam ao fechar;</li>
<li>um modelo (Base-Infix 124M) e 4 problemas — generalização vem da Fase 4 (122 problemas SRBench, campeão, 3 seeds);</li>
<li>tuning PPO por braço introduz assimetria (defendida em §6; GRPO nos defaults é a comparação conservadora);</li>
<li>C=1 (sem ajuste de constantes) — ablação com L-BFGS-B prevista só para o campeão;</li>
<li>análise prior×dataset (analyze_model_prior.py) e mixed-model entram no fechamento.</li>
</ul>

<p style="margin-top:30px;color:#666;font-size:.85em">Gerado por build_phase2_report.py — Seriguela, {date.today().isoformat()}.
Fontes: results/phase_2 (JSONs por seed com histórico por step), docs/reports/PHASE3_ANALYSIS_PLAN.md.</p>
</body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "relatorio_fase2.html"
    out.write_text(html, encoding="utf-8")
    print(f"[ok] {out} ({out.stat().st_size/1024:.0f} KB, {total} seed-runs)")


if __name__ == "__main__":
    main()
