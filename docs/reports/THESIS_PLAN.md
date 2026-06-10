# THESIS_PLAN.md — Seriguela: Fonte Única de Verdade

**Última atualização:** 2026-05-18  
**Status do projeto:** Fase 0 (consolidação) → Fase 1 (piloto) em progresso  
**Branch ativo:** `claude/cool-kowalevski-103c25`

---

## 1. Contribuição Central

> **"Elite Buffer–Augmented Reinforcement Learning for LLM-based Symbolic Regression: PPO vs GRPO with GPT-2"**

Um modelo GPT-2 Base (124M parâmetros) fine-tuned via LoRA em 682K expressões sintéticas é otimizado via RL para resolver problemas de regressão simbólica. Comparamos quatro variantes algorítmicas (Pure-PPO, Pure-GRPO, BoN-PPO, BoN-GRPO) em problemas de dificuldade variável e avaliamos o campeão em um subconjunto do benchmark SRBench (Feynman + Strogatz). Testamos explicitamente a hipótese H_gen: modelos treinados em ≤5 variáveis generalizam para problemas de maior dimensionalidade.

---

## 2. Hipóteses Principais

| ID | Hipótese | Como testar |
|----|----------|-------------|
| **H_rl** | RL (qualquer variante) supera o baseline Best-of-N zero-shot | Fase 2: comparar best_r2 final vs. best_of_n corrigido nos mesmos problemas |
| **H_buffer** | Buffer elite (BoN-PPO, BoN-GRPO) supera RL puro (Pure-PPO, Pure-GRPO) no mesmo reward | Fase 2: mesmo reward_fn para todos — controla o confound reward identificado nos dados t5/t6 |
| **H_ppo_vs_grpo** | PPO e GRPO diferem em estabilidade de convergência além do R² final | Fase 3: análise de variância + curvas de convergência step-a-step |
| **H_gen** | O modelo generaliza para n_vars > 5 (além do escopo de treino SFT) | Fase 4: classificar Feynman/Strogatz por n_vars e reportar R² estratificado |

---

## 3. Estado Atual dos Dados

### 3.1 Dados válidos existentes (test5/test6, março 2026)

Localização: `results/pre_phase__t5_and_t6_merged/` — 36 JSONs agregados.

**Design:** 4 algoritmos × 3 problemas × 3 seeds = 36 runs  
**Modelo:** `augustocsc/gpt2_base_infix_682k` (Base Infix, 124M)

**⚠️ CONFOUND IDENTIFICADO:** Os algoritmos BoN usaram `reward_fn=sr_ic_lambda0.1` enquanto os algoritmos Pure usaram `reward_fn=r2_clipped`. Isso invalida a comparação direta BoN vs Pure nestes dados. A Fase 2 corrige isso usando o mesmo reward para todos.

**Resumo dos resultados (mean best_r2 por algoritmo × problema):**

| Algoritmo | nguyen_1 (fácil) | nguyen_5 (médio) | nguyen_9 (difícil, 2-var) |
|-----------|-----------------|-----------------|--------------------------|
| bon_grpo  | 0.993 | 0.164 | 0.545 |
| bon_ppo   | 0.993 | 0.205 | 0.745 |
| pure_grpo | 0.997 | 0.372 | 0.606 |
| pure_ppo  | 0.995 | **0.516** | 0.641 |

**Observações dos dados t5/t6:**
- Nguyen_1: todos os algoritmos convergem (R²≥0.99) — zero sinal para comparação
- Nguyen_5: alta variância entre seeds; confound de reward impede conclusões
- Nguyen_9: variância moderada; pure_ppo e bon_ppo mostram sinal mais consistente
- Sem baseline `best_of_n` (bug não corrigido nestes runs)

### 3.2 Dados comprometidos (Phase A, 9.751 runs W&B)

Documentados em `docs/reports/phase_a_post_mortem.md`. **Não usar para defesa.**  
Falhas críticas: baseline crashado, 89% single-seed, grid desequilibrado, reward/penalty não logados.

---

## 4. Design Experimental (Fases 1–4)

### Fase 1 — Piloto de seleção de modelo + correção de bugs (sem GPU intensiva)

**Objective:** Escolher o(s) modelo(s) que entram na Fase 2 e corrigir os bugs críticos.

**Bugs a corrigir antes de qualquer GPU:**
1. `2_training/reinforcement/algorithms/best_of_n.py:254` — `AttributeError: 'float' object has no attribute 'is_valid'` (wrapper RewardResult)
2. `2_training/reinforcement/run_experiment.py:163` — `local_vars["x"] = None` para n_vars>1 (invalida Nguyen 9-12)
3. `3_evaluation/core/metrics.py` — adicionar `symbolic_match()` via `sympy.simplify`

**Piloto zero-shot:** 6 modelos × Nguyen {1, 5, 9} × 5 seeds × N=512 amostras (Best-of-N puro)  
**Custo:** ~4-5h T4 free tier (sem usar units Pro)

**Critério de seleção:** Se Base ≥ Medium em R² médio → RL roda só em Base (economiza ~40% compute).

### Fase 2 — Experimento RL principal

**Design (controlado, sem confound de reward):**

| Eixo | Valores |
|------|---------|
| Modelo | Melhor(es) da Fase 1 (previsto: Base-Infix) |
| Algoritmo | best_of_n (**corrigido**), pure_ppo, pure_grpo, bon_ppo, bon_grpo |
| Reward | **Uniforme: `r2_clipped`** para todos os algoritmos (emenda 2026-06-10, ver abaixo) |
| Problema | nguyen_3, nguyen_5, nguyen_7, nguyen_9 (exclui nguyen_1 — sem sinal) |
| Seeds | 42, 123, 456, 789, 1011 (5 seeds para significância estatística) |
| MAX_STEPS | ~200 (determinado por scout runs de plateau na Fase 1.5) |
| **Total runs** | 5 × 4 × 5 = **100 runs** (1 modelo) |

**EMENDA 2026-06-10 — reward uniforme passa de `sr_ic` para `r2_clipped`.**
O piloto de timing (Fase 1.5, RunPod) revelou que `sr_ic` com C=1 não fornece
sinal de aprendizado: a normalização `max(0, (−log(MSE+ε) − 0.1·C)/25)` zera
exatamente a recompensa de qualquer expressão cuja penalidade de complexidade
supere o termo de ajuste — na prática, ~todo o espaço amostrado. Um run
pure_ppo de 500 steps (512K amostras) ficou com best R²=0,0 do início ao fim e
colapsou para constantes degeneradas, enquanto best_of_n com o MESMO modelo
atinge R²≈0,39 em 50 steps. Isso também reinterpreta os dados t5/t6: os braços
BoN (que usavam sr_ic) performavam ≈ baseline porque o gradiente era nulo — o
buffer fazia o trabalho. `r2_clipped` é denso em [0,1] e foi validado nos
braços pure do t5/t6 (pure_ppo 0,516 em nguyen_5, mesmo protocolo C=1). A
análise de complexidade das expressões vencedoras permanece na Fase 3, como
métrica descritiva (não embutida no reward). O dead-zone do sr_ic entra na
discussão da dissertação como achado de design de reward.

**Parâmetros fixos** (do experiment_plan_v2.md, com correção de temperature):
- penalty: `gradient`, prompt: `standard`, temperature: `cosine_annealing`
- batch_size: 1024, max_new_tokens: 50

**Otimizações de compute:**
- Runner unificado `run_unified_bon.py`: compartilha rollouts entre best_of_n/bon_ppo/bon_grpo (3 runs → 1 geração)
- Early-stopping: R²<0.1 em step 100 → kill; entropy collapse → kill
- Seeds em batch para runs de inferência pura

### Fase 3 — Análise estatística

- ANOVA two-way (algoritmo × problema) com seed como bloco aleatório
- Tukey HSD post-hoc para pares de algoritmos
- Effect sizes (Cohen's d): BoN vs Pure, PPO vs GRPO, RL vs baseline
- 95% CIs em todas as comparações principais
- Análise de complexidade das expressões vencedoras
- **Saída:** campeão `(modelo, algoritmo)` escolhido com justificativa estatística

### Fase 4 — Desafio SRBench (Feynman + Strogatz)

**Estratégia "Run All, Smart Where It Matters":**

**Classificação dos 113 problemas (99 Feynman + 14 Strogatz):**
- **Categoria A (vocab-covered, ~40-60):** operadores ⊆ {+,-,*,/,**,sin,cos,tan,log,sqrt,exp}
- **Categoria B (vocab-uncovered, ~50-70):** contém arccos, arctan, tanh, sinh, cosh

**Estágio 4.1 — Zero-shot em TODOS os 113 (custo ~$10 L4):**
- Inferência pura, N=256, 2 seeds — sem RL
- Produz "tabela mãe" de 113 linhas

**Estágio 4.2 — RL em Categoria A (~$30-65 L4):**
- Algoritmo campeão da Fase 3 em ~40-60 problemas × 3 seeds
- Protocolo SRBench: 75/25 split, 10K samples, 1h wall-clock

**Estágio 4.3 — Baselines locais (CPU, gratuito):**
- gplearn e PySR em Categoria A via presets SRBench

**Deliverables da Fase 4:**
1. Tabela mãe 113 linhas (Apêndice)
2. Tabela "Champion vs. SRBench Leaderboard" (Categoria A)
3. Figura Pareto: R² médio vs. complexidade
4. **Figura H_gen: R² médio em função de n_vars (1–9)** — contribuição original

---

## 5. Arquivos Críticos

**Bugs a consertar (Fase 1):**
- `2_training/reinforcement/algorithms/best_of_n.py:254`
- `2_training/reinforcement/run_experiment.py:163`
- `3_evaluation/core/metrics.py` (adicionar symbolic_match)

**Pipelines a reusar:**
- `3_evaluation/cli.py` — avaliação zero-shot
- `2_training/reinforcement/run_pre_phase_t6.py` — runner Colab+Drive
- `2_training/reinforcement/algorithms/base_trainer.py` — backbone PPO/GRPO
- `2_training/reinforcement/rewards/sr_ic.py` — reward canônico
- `4_analysis/statistical/aggregate_nguyen_results.py` — modelo de agregação

**Pipelines a criar:**
- `experiments/queue.yaml` + `experiments/queue_processor.py` — daemon Colab
- `colab/seriguela_runner.ipynb` — notebook daemon
- `colab/ssh_bootstrap.ipynb` — debug SSH
- `2_training/reinforcement/run_unified_bon.py` — runner que compartilha rollouts
- `4_analysis/notebooks/t5_t6_analysis.ipynb` — análise dos 36 JSONs existentes
- `4_analysis/challenge/filter_feynman.py`, `var_remapper.py`, `srbench_protocol.py`

---

## 6. Cronograma

| Semana | Fase | Marcos chave |
|--------|------|--------------|
| 1 (atual) | 0 | Repo limpo ✅, THESIS_PLAN ✅, notebook t5_t6 (em progresso), pipeline Colab |
| 2 | 1 | Bug fixes + piloto 6 modelos + scout de plateau + decisão de modelo |
| 3–5 | 2 | RL principal (100 runs, Base, sr_ic uniforme, MAX_STEPS=200) |
| 6–7 | 3 | Análise estatística + escolha do campeão |
| 7–10 | 4 | SRBench (113 problemas, Estágios 4.1-4.3) |
| 11–14 | 5 | Dissertação + draft de paper |

---

## 7. Custo Estimado de Compute

| Fase | Compute | Custo units |
|------|---------|-------------|
| 1 — Piloto (T4 free) | ~5h T4 | 0 |
| 1.5 — Scout plateau (T4 Pro) | ~6h T4 | ~12 units |
| 2 — RL principal Base (T4 Pro) | ~50-60h T4 | ~120 units |
| 4.1 — Zero-shot 113 problemas (L4) | ~19h L4 | ~92 units |
| 4.2 — RL Categoria A (L4) | ~60-135h L4 | ~290-650 units |
| **Total mínimo** | | **~520 units (~$55)** |

Colab Pro = 100 units/mês. Plano mínimo: 5.2 ciclos (~$52). Pack de 25 units custa ~$10.

---

## 8. Targets de Publicação

**Dissertação:** defesa prevista em ~14 semanas a partir de agora.

**Paper de conferência (draft extraível da dissertação):**
- GECCO 2027 SR track (CFP costuma fechar Jan/Fev)
- AutoML Conference
- IEEE TEVC (jornal, revisão longa mas impacto maior)

**Contribuição original defensável:**
1. Comparação controlada PPO vs GRPO com elite buffer em modelos GPT-2
2. Teste empírico de H_gen (generalização dimensional fora do escopo de treino)
3. Uso do protocolo SRBench padronizado para comparação justa com o estado da arte

---

*Este arquivo substitui os 14 relatórios arquivados em `docs/archive/`. Para histórico completo, consultar `docs/archive/`.*
