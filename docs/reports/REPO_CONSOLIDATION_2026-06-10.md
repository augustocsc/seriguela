# Consolidação do Repositório — 2026-06-10

**Objetivo:** juntar todo o trabalho disperso em uma única branch (`main`), remover o
que não sustenta o core da tese e deixar o repo pronto para rodar as fases restantes
no RunPod.

**O core (inalterado):** um GPT-2 pequeno (124M), fine-tuned via LoRA em 682K
expressões, otimizado por RL para regressão simbólica; comparação controlada de
5 braços (best_of_n, pure_ppo, pure_grpo, bon_ppo, bon_grpo) com reward uniforme
(`sr_ic`, C=1) em benchmark consolidado (Nguyen + SRBench Feynman/Strogatz).
Fonte única de verdade: [THESIS_PLAN.md](THESIS_PLAN.md).

---

## 1. Estado encontrado

### Branches (6 locais + 4 remotas)
Todas apontavam para o mesmo commit (`5f92050`) ou estavam **atrás** dele.
Zero commits únicos fora de `main`. A "bagunça" não estava nos commits — estava em:

1. **Trabalho não-commitado** no worktree `optimistic-kilby`: o orquestrador RunPod
   (`runpod/launch_pilot.py`) e o piloto de timing (`experiments/run_pilot_timing.py`),
   criados em 2026-06-09 e nunca commitados. **Resgatados nesta consolidação.**
2. **4 worktrees** duplicando ~1.7GB de checkout cada (≈7GB de disco).
3. **Lixo acumulado** na raiz e em diretórios de eras encerradas (abaixo).

### Pod RunPod órfão
`optimistic-kilby/runpod/pod_info.json` registrava o pod `yt3wrpceru8j8j`
($0.22/h) criado em 2026-06-09. A etapa 0 da consolidação coleta o
`timing_report` (se existir no pod) e **termina o pod**.

### Dados
| Conjunto | Status | Decisão |
|---|---|---|
| Phase A (9.751 runs W&B, fev/2026) | **Inválido** — baseline crashado, 89% single-seed ([post-mortem](phase_a_post_mortem.md)) | Já fora do repo; scripts em `legacy/phase_a` |
| t5/t6 (36 runs RL, mar/2026) | **Válido** com confound de reward documentado | Mantido: `results/pre_phase__t5*` (brutos têm histórico por step → curvas de convergência da Fase 3) |
| Etapa 0 + Fase 1 (piloto 18 exp) | Válido (W&B artifact `phase1-results`) | Mantido: `results/etapa0/` |
| Fase 1b (Base, 122 problemas × 10 seeds, BoN) | **Válido — completo** | Mantido: `results/phase_1b/` |
| Fase 1c (Large, 122 problemas × 10 seeds, BoN) | **Válido — completo** | Mantido: `results/phase_1c/` |

As fases 1b/1c são, na prática, o Estágio 4.1 do plano ("tabela mãe" zero-shot)
para dois dos modelos — colhidas com C=1, consistentes com o protocolo da Fase 2.

---

## 2. O que saiu (e por quê)

| Item | Tamanho | Motivo |
|---|---|---|
| `1_data/benchmarks/feynman/*.csv` | 819MB | `feynman_loader.py` baixa e cacheia do PMLB sob demanda (testado em produção nas fases 1b/1c) |
| `1_data/benchmarks/srbench/` | 373MB | Fora do escopo: a Fase 4 usa apenas Feynman+Strogatz |
| `1_data/benchmarks/blackbox/` | 38MB | Idem (datasets caixa-preta não entram na tese) |
| ~60 `aws_launch_*.json` (raiz) | — | Registros de launches AWS de fev/2026 (era encerrada) |
| `temp_results.json`, `remaining/completed_base_configs.json` | ~3MB | Artefatos da Phase A inválida |
| `large_files*.txt/json`, `push_err.txt`, `compare_t6_t5_step50.py` | — | Diagnósticos one-off já consumidos |
| `dev/null/` | — | Bug de redirecionamento (criou diretório literal com hooks git) |
| `src/seriguela/` | 66K | Pacote morto: nenhum import no repo, duplicava `classes/expression.py` |
| `wandb/` (repo principal) | 19MB | Logs locais de runs de mai/2025 (gitignored) |

**Mantido em `1_data/benchmarks/`:** `nguyen/` (84K), `strogatz/` (400K),
`feynman/*.meta.json`, `benchmarks_metadata.json` e os scripts de download.

## 3. O que virou `legacy/` (histórico auditável, fora do caminho)

| De | Para |
|---|---|
| `colab/` (notebooks runner/ssh) | `legacy/colab/` — créditos Colab esgotados; o **daemon de fila continua vivo** (`experiments/queue_processor.py`, roda em qualquer VM com `NO_GIT=1`) |
| `experiments/{run_phase1c_*,fix_and_run_phase1c,download_phase1c,run_missing,test_colab_ready}.py`, `queue_1c_{a,b}.yaml` | `legacy/colab_phase1/` — one-offs das fases 1b/1c (concluídas) |
| `2_training/reinforcement/{run_pre_phase_t6,merge_t5_t6,final_t5_t6_compare}.py` | `legacy/pre_phase/` — proveniência dos dados t5/t6 |
| `scripts/pre_phase_b/` | `legacy/pre_phase_b/` |
| `aws/` | `legacy/aws/` — substituída por RunPod (custo) |
| `future/EXPERIMENT_MODERN_MODELS_COMPARISON.md` | `docs/proposals/` |
| `docs/FAST_PHASE_A_REPORT.md` | `docs/archive/` (fase inválida) |

## 4. Mudanças estruturais

1. **LFS desativado para arquivos novos** (`.gitattributes`). O blanket
   `*.json/*.csv` em LFS colocava até JSONs de resultado no LFS e quebrava clones
   sem `git-lfs` (pointer files — a causa-raiz dos hotfixes `df5273c`/`1bfac35`).
   Com os datasets grandes fora do repo, tudo que resta é pequeno; renormalizado
   para blobs normais. Objetos LFS antigos seguem resolvíveis no histórico.
2. **`runpod/` é o caminho oficial de compute** — ver [runpod/README.md](../../runpod/README.md):
   piloto de timing (Fase 1.5, define `MAX_STEPS`) → fila da Fase 2 via
   `experiments/build_phase2_queue.py` → daemon no pod com `NO_GIT=1`.
3. **Smoke test robusto a fila vazia** (`experiments/test_smoke.py`): não quebra
   mais quando todas as entradas da fila estão `done`.
4. **Dissertação, cap. 3:** removida a contradição texto×código sobre otimização
   de constantes — o protocolo é **C=1 uniforme** em todos os braços, com ablação
   L-BFGS-B (bounds [−2,2]) reservada ao campeão.
5. **Worktrees/branches obsoletos removidos**; `main` é a única branch viva.

## 5. Próximos passos (ordem)

1. `python runpod/launch_pilot.py` — piloto de timing (~3–5h, ~US$2).
2. Ler `results/pilot_timing/timing_report.md` → fixar `MAX_STEPS` da Fase 2.
3. `python experiments/build_phase2_queue.py --max-steps <N> --write` + smoke + push.
4. Fase 2 no pod (daemon, ~US$15–25) → análise (Fase 3: ANOVA, Tukey, effect sizes).
5. Fase 4.2: RL do campeão na Categoria A do Feynman (orçado pelo probe do piloto).

## 6. Limitações assumidas

GPT-2 é tecnologia de 2019 e os resultados absolutos serão modestos frente ao
estado da arte de SR. **Isso não compromete a contribuição**, que é comparativa e
metodológica: (i) comparação controlada PPO×GRPO×elite-buffer com reward uniforme
e 5 seeds; (ii) teste explícito de generalização dimensional (H_gen); (iii)
protocolo SRBench para comparabilidade. Modelos modernos ficam como trabalho
futuro ([docs/proposals/EXPERIMENT_MODERN_MODELS_COMPARISON.md](../proposals/EXPERIMENT_MODERN_MODELS_COMPARISON.md)).
