# HANDOFF — Revisão completa da dissertação (2026-07-07; fechamento com dados finais em 2026-07-14)

Este documento descreve **tudo o que foi feito** na pasta `dissertation_final/`,
**onde estão os dados fictícios** que você precisa substituir, e o que conferir
antes da defesa. A pasta `dissertation/` original **não foi tocada** — todo o
trabalho novo está aqui.

---

## 1. O que há nesta pasta

```
dissertation_final/
├── ic-tese-v3.tex              # main atualizado (título real, resumo/abstract escritos)
├── ic-tese-v3.cls              # cópia inalterada da classe
├── chapters/
│   ├── 1.introduction.tex      # reescrito — RQs alinhadas ao desenho final
│   ├── 2.concepts_and_works.tex# revisado + seções novas (LoRA, GRPO, RLHF/BoN, buffer)
│   ├── 3.materials_and_methods.tex  # reescrito — desenho experimental REAL (fases 1–4)
│   ├── 4.results.tex           # escrito do zero com os dados reais
│   ├── 5.conclusion.tex        # escrito do zero
│   └── appendix_mother_table.tex    # gerado do results/mother_table.csv (127 linhas)
├── bibliography/ic-tese-v3.bib # original + 4 entradas novas (ver §5)
├── figures/                    # 15 figuras (PDF p/ LaTeX + PNG p/ conferência) + logo
├── figures_src/
│   ├── make_figures.py         # regenera as 9 figuras de DADOS (100% dados reais)
│   ├── make_diagrams.py        # regenera os 6 diagramas conceituais
│   ├── make_appendix.py        # regenera o apêndice a partir do mother_table.csv
│   └── check_tex.py            # checagens de consistência (cites/figuras/refs/chaves)
└── HANDOFF.md                  # este arquivo
```

**Como compilar:** não há LaTeX instalado nesta máquina, então a compilação
**não foi testada** — apenas a consistência estrutural (todas as `\cite`
existem no .bib, todas as figuras existem, `\ref`/`\label` casam, ambientes e
chaves balanceados — `python figures_src/check_tex.py`). Suba a pasta no
Overleaf e compile com pdfLaTeX + BibTeX. Removi `svg`, `wrapfig`, `tikz`,
`geometry` e `longtable`+`geometry` do preâmbulo original que não eram usados
(`longtable` ficou — é usado; `geometry` saiu porque conflita com margens da
classe ic-tese; se a classe reclamar de algo, me desculpe — é o único risco
que não pude testar).

---

## 2. ⚠️ ONDE ESTÃO OS DADOS FICTÍCIOS (substituir no futuro)

Todo número inventado está envolto no comando `\fic{...}` e tem um comentário
`% FICTITIOUS` na linha. Para **enxergar todos em vermelho no PDF**, troque no
`ic-tese-v3.tex`:

```latex
\newcommand{\fic}[1]{#1}
%\newcommand{\fic}[1]{\textcolor{red}{#1}}   % <- descomente esta
```

Lista completa após o fechamento de 2026-07-14 (**3 pontos fictícios restantes** — os
outros da lista original foram substituídos por dados reais):

| # | Onde | O que é fictício | O que fazer |
|---|------|------------------|-------------|
| 1 | `4.results.tex`, §4.5.4 "Ablation: the Cost of C=1" **INTEIRA** + Tabela `tab:constant_ablation` | A ablação L-BFGS-B **nunca foi rodada**. Todos os valores da coluna "C fitted" (19.2%, 33.3%, 0.849, 6.3×) são chutes plausíveis. A coluna "C=1" é real (16.2/27.3/0.808) | Rodar a ablação no campeão (o framework já tem a rotina L-BFGS-B, bounds [-2,2]; é só re-avaliar as expressões já descobertas — não precisa re-treinar). Se não rodar, DELETE a subseção e as 2 frases que a citam no cap. 5 (RQ5 "lower bound…" e §Limitations "quantifies the cost…") |
| 2 | `appendix_mother_table.tex` | **2 linhas marcadas com †**: feynman_II_11_17 (0.7412/0.7248) e feynman_I_34_10 (0.8062/0.7981) — runs ainda pendentes; valores inventados | Quando terminarem, rode `python 4_analysis/build_mother_table.py` e depois `python figures_src/make_appendix.py` — as linhas fictícias somem sozinhas. Recompute também os agregados (ver §2.5 abaixo) se algum deles resolver o problema |
| 3 | `ic-tese-v3.tex` | Data da defesa (15/09/2026) e Agradecimentos genéricos | Personalizar |

## 2.5. ⚠️ VERIFICAR — dados reais que merecem conferência antes da defesa

Estes NÃO são fictícios, mas são pontos onde um examinador (ou você) pode tropeçar:

1. **🔴 nguyen_1 — resultado suspeito (prioridade máxima).** A tabela-mãe reporta
   R²=1.0 com a expressão `x_1*(x_1 + x_1**C) + x_1`, que com C=1 é **2x²+x —
   não é o alvo x³+x²+x**. R²=1.0 para 2x²+x é quase impossível no domínio
   usual do Nguyen-1. Possíveis causas: dataset local errado, bug no cálculo
   do R², ou truncamento da expressão no agregado. **Conferir o run em
   `results/phase_4_local/nguyen_1/`.** Se o resultado for inválido: nguyen
   solved cai de 3→2, a linha Nguyen da tabela agregada (cap. 4) muda de
   25.0%→16.7% em solution rate, e a linha do apêndice muda. O texto NÃO
   destaca nguyen_1 em lugar nenhum (de propósito), então o impacto é só
   nesses dois números.
2. **Critério de solução = proxy.** Solution rate usa R²≥0.999 como proxy do
   symbolic match via SymPy do SRBench (limitação já declarada no cap. 5).
   O caso nguyen_1 acima é exatamente o tipo de over-count que o proxy
   permite. Se der tempo: rodar um check simbólico (sympy.simplify) nas 19
   expressões "solved" da mother table.
3. **Test-R² com fallback para train.** 73 dos 125 runs não salvaram test_r2
   (runs antigos). A tabela agregada do cap. 4 declara isso na caption e usa
   test quando existe, train quando não. Os números casam com
   `results/srbench_comparison.md` (16.2/27.3/0.808), que usa o mesmo
   critério. Se quiser tudo em test-R²: re-avaliar as expressões da mother
   table nos splits (barato — só avaliação).
4. **Feynman n=99, não 96.** A fase 4 acabou cobrindo 99 problemas Feynman
   (96 do benchmark consolidado + 5 extras fora de vocabulário, 2 ainda
   pendentes). O cap. 3 declara o benchmark de 122 e o cap. 4 explica os
   extras no 1º parágrafo da §4.5. Conferir se essa contagem bate com a sua
   definição oficial do benchmark.
5. **Fase 4 = 1 seed (42).** Agora está DECLARADO honestamente nos caps. 3, 4
   e 5 (não é mais fictício). Se você rodar as seeds 123/456 depois, os
   números só podem melhorar (best-of-seeds) — atualize mother table →
   apêndice → agregados → figuras.
6. **H_gen e Pareto usam best R² de TREINO** (dito nas legendas); a tabela
   agregada usa test-com-fallback. São visões diferentes de propósito.
7. **Hardware do SFT** (cap. 3): 2×EPYC 7513 + A100 80GB veio do seu texto
   antigo — confirmar.
8. **Título em português** — minha sugestão, revise.

---

## 3. Principais decisões editoriais (e por quê)

1. **O texto antigo descrevia o experimento errado.** Os caps. 3–4 originais
   descreviam o desenho fatorial da "Phase A" (8.640 configs, REINFORCE, noise,
   prompts oracle/distractor como eixos), que o post-mortem invalidou. Reescrevi
   tudo em torno do desenho REAL (THESIS_PLAN): fases 1→4, 5 braços, reward
   uniforme r2_clipped, C=1, 100 seed-runs. O material de noise/prompt não foi
   deletado — virou a subseção "Fixed Factors and Scope" (cap. 3) e trabalho
   futuro (honesto: está implementado, não foi variado).
2. **A Phase A virou conteúdo.** O post-mortem (baseline crashado, 89%
   single-seed, grid fragmentado) está resumido no início do cap. 4 como
   "lição metodológica" — examinadores adoram autópsias honestas, e ela motiva
   as 3 regras do protocolo final.
3. **A narrativa central é exploração × retenção** (do seu relatório da Fase 3,
   `results/phase_3/relatorio_fase2.html`): PPO explora ~10× mais que GRPO; o
   buffer só ajuda quem explora; BoN-PPO é o único quadrante com os dois
   fatores. Todas as estatísticas (ANOVA F/p, Tukey, Cohen d) vieram desse
   relatório.
4. **Achados de estabilidade viraram seção própria** (§4.3): dead-zone do
   sr_ic (com figura do run real de 500 steps), colapso do PPO nos defaults
   (59%→0% validade), estouro de memória do SymPy + guards. Argumento:
   métricas de processo são condição de validade.
5. **REINFORCE saiu** da lista de algoritmos (não está no desenho final).
   A tabela de braços agora são os 5 reais.
6. **Título** atualizado nos dois idiomas (era placeholder). Confira se o
   título PT está como você quer.

---

## 4. As 15 figuras

Todas em `figures/` (PDF + PNG). **As 9 de dados usam exclusivamente dados
reais do repo** (nenhum ponto inventado):

| Figura | Fonte de dados | Usada em |
|---|---|---|
| fig_zero_shot_selection | results/phase_1b + phase_1c (122 probs × 10 seeds) | §4.2 |
| fig_phase2_box | results/phase_2 (agregados por seed) | §4.4 |
| fig_phase2_convergence | histórico por step (5 seeds × 5 braços, nguyen_5) | §4.4 |
| fig_phase2_validity | idem (valid_rate por step) | §4.4 |
| fig_phase2_exploration | expressões únicas/seed × best R² | §4.4 |
| fig_sric_deadzone | results/pilot_timing (run sr_ic 500 steps) + fase 2 | §4.3 |
| fig_hgen | results/mother_table.csv (115 problemas done) | §4.5 |
| fig_srbench | results/srbench_comparison.md (nosso + leaderboard publicado) | §4.5 |
| fig_pareto | mother_table (complexidade = contagem de tokens) | §4.5 |

E 6 diagramas conceituais (matplotlib) que **substituem PDFs que não existiam
no repo** (o LaTeX original referenciava `Overview.pdf`, `Data Engeneering.pdf`,
`genetic_tree.pdf`, `neuron.pdf`, `GPT-2 Tokenization Result.pdf` — nenhum
estava em `dissertation/figures/`): fig_overview, fig_data_engineering,
fig_genetic_tree, fig_neuron, fig_tokenization, fig_rl_loop. **Se você tiver
os originais (Overleaf?), pode preferi-los** — basta trocar o nome no
`\includegraphics`.

Regenerar tudo: `python figures_src/make_figures.py && python figures_src/make_diagrams.py`

---

## 5. Revisão de literatura — avaliação e o que mudou

**Estado:** a revisão que você tinha era boa — cobertura sólida de GP clássico
(Koza→GE→AFP→lexicase→semântica→GSGP→GP-GOMEA→GSR→frameworks) e da linha
neural (DSR→NeSymReS/SymbolicGPT→uDSR→TPSR/DGSR→LLM-SR/LASR/ICSR→multimodal).
Mantive praticamente todos os seus parágrafos, com polimento.

**O que adicionei:**
- **Teoria que faltava e o trabalho USA:** LoRA/PEFT (§2 nova), GRPO (§2 nova,
  com a eq. da vantagem por grupo e a discussão da pressão seletiva), RLHF/
  Best-of-N/InstructGPT e o elite buffer como parente do Priority Queue
  Training do DSR (§2.5.6 nova). Sem isso, o cap. 4 citava conceitos nunca
  definidos.
- **Subseção "Positioning of This Work"** no fim do related work — examinador
  sempre pergunta "qual a diferença para DSR/DGSR/LLM-SR?"; agora está escrito.
- **Tabela de related work corrigida:** Seriguela 124M (estava "0.12M"!), ano
  2026, cite do SR-GPT que faltava, linha do E2E (Kamienny 2022) adicionada,
  DGSR sem parâmetro inventado.
- **4 refs novas no .bib:** `hu2021lora` (LoRA), `shao2024deepseekmath` (GRPO
  original — o texto antigo atribuía GRPO só ao DeepSeek-R1), `romano2021pmlb`
  (PMLB, de onde vêm os dados!), `stiennon2020learning` (Best-of-N).
  ⚠️ Confira os metadados delas (escrevi de memória; ano/venue do LoRA=ICLR
  2022, DeepSeekMath=arXiv 2402.03300, PMLB=Bioinformatics 2022, Stiennon=
  NeurIPS 2020).

**Sugestões se quiser reforçar (não fiz):** (i) citar o survey/leaderboard
SRBench 2.0 se já tiver saído; (ii) um parágrafo sobre RL fine-tuning de LLMs
para raciocínio matemático (além de DeepSeekMath) conectando ao seu colapso de
PPO — há literatura de 2025 sobre instabilidade de RLHF que casaria bem.

---

## 6. O que ainda falta RODAR (experimentos)

Atualizado em 2026-07-14 (os 10 problemas que faltavam da seed 42 rodaram e já
estão no texto):

1. **Ablação L-BFGS-B no campeão** — o ÚNICO fictício substantivo restante
   (fictício nº 1 do §2). Pós-hoc: refit das constantes das expressões já
   descobertas — não precisa re-treinar. Barata: só avaliação.
2. **feynman_II_11_17 e feynman_I_34_10** (fictício nº 2) — os 2 runs pendentes.
3. **Verificação do nguyen_1** (item 🔴 do §2.5) — não é rodar, é auditar o
   run existente.
4. (Opcional) Seeds 123/456 da Fase 4 — melhoraria os números e removeria a
   limitação single-seed; o texto atual já é honesto sem elas.
5. (Opcional) test_r2 dos 73 runs antigos que não salvaram o split.

Depois de cada um: `python 4_analysis/build_mother_table.py` →
`python 4_analysis/srbench_comparison.py` → `make_appendix.py` +
`make_figures.py` → conferir números citados no texto (grep por "16.2", "27.3",
"0.808", "25.0", "58.3", "0.997", "0.857" nos .tex) → recompilar.

---

## 7. Checklist final antes de submeter

- [ ] Substituir/confirmar os 3 pontos fictícios do §2 (ablação é o único substantivo)
- [ ] Passar pelos 8 itens de verificação do §2.5 — **nguyen_1 primeiro** 🔴
- [ ] Compilar no Overleaf (pdfLaTeX + BibTeX) e caçar warnings de overfull
      hbox na tabela do apêndice (expressões longas — se estourar margem,
      reduza para `\tiny` ou quebre a coluna)
- [ ] Conferir §Hardware (cap. 3): mantive a máquina do SFT (2×EPYC + A100
      80GB) do seu texto antigo e adicionei RunPod 3090/A40 + 3050 local para
      o RL. **Confirme se o SFT foi mesmo nessa máquina.**
- [ ] Agradecimentos, data da defesa, banca, ficha catalográfica
- [ ] Título em português — minha sugestão, revise
- [ ] O `contributors.txt` da pasta antiga não foi copiado (não sei o que era)
- [ ] Rodar `python figures_src/check_tex.py` de novo após qualquer edição
- [ ] Se usar as figuras conceituais minhas, cheque se o estilo agrada; se
      tiver as originais do Overleaf, é só trocar os nomes

---

## 8. Fontes de cada número real (auditoria rápida)

- **Tabela mestre Fase 2** (`tab:phase2_master`): `results/phase_3/master_table.md` (idêntica, dígito a dígito)
- **ANOVA/Tukey/Cohen** (`tab:anova`): `results/phase_3/relatorio_fase2.html` §4
- **Exploração 35–56K vs 4–7K, validade 59%→…, KL 0.22, 38GB, US$2**: mesmo relatório §§2–6
- **Zero-shot Base 0.567/Large 0.510, 73/122, validade 92.1/94.2%**: computado dos 244 agregados `results/phase_1b|1c/*/aggregate_*.json` (best por problema = máx sobre 10 seeds)
- **Fase 4 Feynman/Strogatz**: `results/srbench_comparison.md`; leaderboard publicado citado como La Cava et al. 2021
- **H_gen (medianas 0.997/0.948/0.896/0.752/0.623; ≤5: 0.863 vs >5: 0.454)**: `results/mother_table.csv`, 115 problemas done
- **Expressões-vitrine** (nguyen_11 = exp(y·log x), shear-flow = cos/tan, Feynman III.17.37 e III.21.20): mother_table, colunas found/target, R²=1.0
- **Dead zone**: `results/pilot_timing/solo_pure_ppo_nguyen5` (reward médio negativo nos 500 steps — está na figura)
- **Emendas de protocolo (sr_ic→r2_clipped; tuning PPO lr 1e-6/2ep/KL 0.02)**: `docs/reports/THESIS_PLAN.md`, emendas de 2026-06-10
- **Post-mortem Phase A**: `docs/reports/phase_a_post_mortem.md`

---

*Gerado por Claude (Fable 5) em 2026-07-07, no branch `claude/stoic-aryabhata-deb9ea`.*
