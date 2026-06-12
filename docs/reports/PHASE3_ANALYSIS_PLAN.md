# Fase 3 — Plano de Análise Profunda (o "porquê", não só o ranking)

**Criado:** 2026-06-12, durante a execução da Fase 2.
**Princípio:** cada comparação responde a um mecanismo, não só a uma média.
Ferramenta principal: [4_analysis/statistical/analyze_phase2.py](../../4_analysis/statistical/analyze_phase2.py)
(roda local, CPU, sobre `results/phase_2`).

## 1. Núcleo estatístico (o "quem ganhou", com rigor)
- Tabela mestre: média ± dp do best R² por braço × problema (5 seeds)
- ANOVA two-way (algoritmo × problema), Tukey HSD, Cohen's d para os pares
  centrais (RL×baseline, buffer×puro, PPO×GRPO), ICs de 95%
- **Taxa de descoberta simbólica** (`symbolic_match` com a expressão alvo) por
  braço — métrica de descoberta, mais exigente que R² (critério SRBench R²>0.999
  reportado em paralelo)

## 2. Dinâmica de treino (o "porquê" — vem do history por step)
- **Curvas de convergência**: best R² vs step (média ± IC por braço×problema).
  Pergunta: o ganho do RL vem de refino tardio ou de sorte amostral cedo?
  (distribuição de `best_step` por braço)
- **Dinâmica de validade**: valid_rate vs step. Já sabemos que PPO mergulha
  (59%→8%) e se recupera; GRPO mergulha menos e recupera mais rápido. Isso é
  ASSINATURA de estabilidade de cada update — quantificar: profundidade do
  vale, step de recuperação, validade final
- **Exploração**: expressões únicas por step e acumuladas; entropia da política.
  Pergunta: o buffer reduz exploração (exploit) ou a direciona?
- **Contribuição do buffer** (braços bon): `buffer_samples_used` vs ganho de R²
  no step seguinte — o elite buffer ancora a política ou só repete elites?
- **Estabilidade do update**: KL por step; correlação entre picos de KL e
  quedas de validade (a espiral que matou os defaults do PPO)

## 3. Nível de expressão (o que os modelos efetivamente produzem)
- Complexidade (tokens) dos vencedores por braço — RL encontra soluções mais
  simples ou mais complexas que a amostragem pura?
- Motivos estruturais: frequência de operadores nas top-expressions por braço
  vs alvo; formas equivalentes da solução (ex.: as 4 formas de sin(x²)cos(x)−1
  que o bon_ppo achou)
- Análise dos fracassos: a seed que falha quando as outras resolvem (ex.:
  bon_ppo/nguyen_5 seed com 0.286) — em que ótimo local ela travou?

## 4. Ligação com o modelo e o dataset (prior → comportamento RL)
Ferramenta: [4_analysis/statistical/analyze_model_prior.py](../../4_analysis/statistical/analyze_model_prior.py)
(amostra o modelo SFT localmente em CPU + estatísticas do dataset 682K via
HF streaming — sem download integral, disco do notebook é apertado).
- **Prior do modelo**: distribuição de operadores, comprimento e validade de
  N amostras zero-shot do `gpt2_base_infix_682k` — comparar com a distribuição
  do dataset de treino (o SFT reproduz o dataset?) e com o que o RL converge
  (o RL amplifica o prior ou luta contra ele?)
- **Dificuldade explicada pelo prior**: nguyen_7 fácil porque log/sqrt são
  frequentes no 682K? nguyen_9 difícil porque expressões 2-var são raras?
  Cruzar frequência no dataset × desempenho por problema
- Conexão com H_gen: o prior de aridade/n-variáveis do dataset explica o
  decaimento de R² com n_vars na Fase 4

## 5. Saídas
- `results/phase_3/master_table.csv` + `.md` (tabela mestre + testes)
- `results/phase_3/figures/` (convergência, validade, KL, complexidade,
  best_step, prior×desempenho)
- Texto-base para o Capítulo 4 da dissertação (resultados) com os números
  citáveis prontos
