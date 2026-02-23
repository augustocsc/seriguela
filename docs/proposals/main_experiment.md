# Plano Experimental: Otimização de LLMs para Regressão Simbólica via Aprendizado por Reforço

## 1. Visão Geral e Escopo do Projeto

O objetivo desta pesquisa é investigar a eficácia de diferentes algoritmos de Aprendizado por Reforço (RL), com ênfase em métodos híbridos (Best-of-N + RL), na otimização de Modelos de Linguagem de Grande Escala (LLMs) para a tarefa de Regressão Simbólica.

A pesquisa utiliza a arquitetura GPT-2 (modelos Base, Medium e Large) previamente submetida a *Supervised Fine-Tuning* (SFT) com 682 mil expressões matemáticas sintéticas. Uma premissa fundamental deste experimento é que **as constantes numéricas (C) não serão otimizadas em tempo de inferência**. A remoção do gargalo de otimização de constantes acelera exponencialmente o cálculo de recompensa, permitindo o uso de uma infraestrutura robusta de nuvem (AWS) para testar amostragem em larga escala e algoritmos de RL mais intensivos.

**Notação Matemática:** Todos os experimentos serão executados em paralelo para ambas as notações (Infix e Prefix). A comparação entre notações será realizada na fase de análise final.

## 2. Arquitetura e Fluxo do Pipeline Experimental

O experimento é operado por um pipeline modular que separa a geração, a avaliação e a otimização. O fluxo de dados ocorre na seguinte ordem lógica:

1. **Motor de Dados (Data Engine):** Instancia o problema matemático (ex: do benchmark de Nguyen). Gera os tensores de treino (In-Distribution) e de teste (Out-of-Distribution), aplicando injeção de ruído quando configurado.
2. **Motor de Inferência Massiva (Inference Engine):** Recebe os dados e o prompt configurado. Utilizando os modelos GPT-2 SFT, gera milhares de expressões candidatas em lote (Best-of-N). Durante a geração, registra e armazena a log-probabilidade média de cada sequência (confiança do modelo).
3. **Motor de Avaliação Offline (Eval Engine):** Opera estritamente na CPU. Converte as strings geradas no passo anterior em árvores sintáticas avaliáveis. Calcula o ajuste matemático de cada expressão válida contra os dados de treino utilizando múltiplas funções de recompensa simultaneamente (R² Puro, Length-Penalized R², SR-IC) e ranqueia as melhores soluções.
4. **Motor de Otimização (RL Engine):** Utiliza as recompensas calculadas no passo 3 como sinal (Reward) para guiar o treinamento dos algoritmos de Reinforcement Learning (PPO, GRPO, BoN-RL), atualizando os pesos da política de geração.
5. **Motor de Validação (Validation Engine):** Submete os modelos otimizados pelo RL a testes de estresse, avaliando a robustez contra prompts distratores, dados fora de distribuição (OOD) e ruído estocástico.

## 3. Fase 1: Estabelecimento da Linha de Base (Best-of-N puro)

Antes de introduzir a otimização de pesos (RL), é necessário estabelecer o limite superior de performance do modelo atuando apenas via amostragem (inferência de força bruta).

* **Modelos Avaliados:** GPT-2 Base (124M), Medium (355M) e Large (774M), nas variantes Infix e Prefix.
* **Amostragem Massiva (N):** Serão geradas 100, 1.000 e 10.000 expressões para cada problema.
* **Hiperparâmetros:** Temperatura fixada em 0.7 e 0.9, com decodificação top-k e top-p.
* **Condicionamento Semântico (Prompt):** Utilização estrita de um **Prompt Padrão / Expandido**. O prompt fornecerá a biblioteca matemática completa do tokenizador para o modelo (todas as operações básicas e transcendentes). Prompts restritivos ou adversariais não serão testados nesta fase.
* **Benchmark:** Os 12 problemas de Nguyen, avaliados nos domínios originais de treinamento (ex: x ∈ [0, 2]).

## 4. Especificação das Funções de Recompensa (Reward Functions)

A métrica padrão R² é insuficiente para treinar modelos em larga escala, pois incentiva o *overfitting* através da geração de equações longas. Para o treinamento de RL na Fase 2, as seguintes funções de recompensa serão implementadas no Eval Engine e comparadas:

### 4.1. R² Clipado Puro (Linha de Base do RL)

Mede exclusivamente o ajuste da curva, limitando penalidades extremas para manter a estabilidade do gradiente.

```
R_clip = max(0, R²)
```

### 4.2. R² com Penalidade de Comprimento (Length-Penalized R²)

Subtrai do R² um valor proporcional à quantidade de nós matemáticos (ou tokens) gerados, forçando o modelo a preferir soluções mais curtas com o mesmo nível de precisão.

```
R_length = R² - α * L
```

*(Onde L é o número de tokens da expressão e α é um hiperparâmetro de penalidade, ex: 0.01).*

### 4.3. Critério de Informação de Regressão Simbólica (SR-IC)

Equilibra o Erro Quadrático Médio (MSE) com a complexidade intrínseca da árvore sintática. Esta função transforma o erro e pune a complexidade de forma logarítmica.

```
R_SRIC = -log(MSE + ε) - λ * C
```

*(Onde C é a complexidade total calculada pelos nós da árvore de expressão, λ é o peso da complexidade e ε previne log de zero. O sinal negativo transforma o critério de minimização em maximização para o algoritmo de RL).*

### 4.4. Tratamento de Expressões Inválidas (Ablation Study)

Expressões que falham no parsing ou produzem resultados inválidos (NaN, Inf, erro de execução) receberão penalidade. **Duas estratégias serão testadas e comparadas:**

#### A. Penalidade Binária Fixa
Todas as expressões inválidas recebem a mesma penalidade constante:
```
R_invalid = -1.0
```

#### B. Penalidade Gradiente (Proposta)
Penalidades diferenciadas baseadas no tipo de falha, fornecendo sinal de gradiente mais informativo:

| Tipo de Falha | Penalidade | Justificativa |
|---------------|------------|---------------|
| Erro de parsing (sintaxe inválida) | -1.0 | Expressão completamente malformada |
| Variáveis incorretas (usa x_2 quando só x_1 existe) | -0.7 | Estrutura válida, semântica incorreta |
| Produz NaN/Inf (divisão por zero, log negativo) | -0.5 | Expressão válida, instabilidade numérica |
| R² negativo (pior que média) | -0.3 | Expressão funcional mas inútil |
| R² ∈ [0, 0.5) (ajuste muito fraco) | 0.0 | Sinal fraco positivo |

A hipótese é que a penalidade gradiente fornecerá um sinal de treinamento mais rico, permitindo que o modelo aprenda a evitar erros específicos de forma mais eficiente.

## 5. Fase 2: Otimização por Aprendizado por Reforço (RL)

Esta fase investiga se o RL consegue superar a amostragem massiva, testando também a resiliência do modelo a contextos de prompt inadequados.

**Nota:** Cada experimento de RL inicia com o modelo SFT original (sem transferência entre runs). Não há estratégia de curriculum - cada configuração é independente.

### 5.1. Algoritmos Avaliados

Os algoritmos serão configurados para maximizar a função de recompensa escolhida na etapa anterior.

* **RL Puro:** PPO e GRPO.
* **Métodos Híbridos (Foco do Estudo):** BoN-PPO e BoN-GRPO (mantendo um *buffer* das melhores expressões para ancorar as atualizações da política).

### 5.2. Estratégias de Temperatura (Ablation Study)

Duas abordagens de temperatura serão testadas durante o treinamento de RL:

#### A. Temperatura Fixa
Manter temperatura constante durante todo o treinamento:
* T = 0.7 (mais conservador)
* T = 0.9 (mais exploratório)

#### B. Temperature Annealing (Proposta)
Redução gradual da temperatura ao longo do treinamento para balancear exploração inicial e exploração final:

```
T(step) = T_max - (T_max - T_min) * (step / total_steps)
```

Configurações a testar:
* **Annealing Linear:** T_max = 1.0 → T_min = 0.5
* **Annealing Cosine:** T(step) = T_min + 0.5 * (T_max - T_min) * (1 + cos(π * step / total_steps))

A hipótese é que o annealing permitirá exploração ampla no início (descoberta de estruturas) e refinamento no final (otimização de detalhes).

### 5.3. Condicionamento Semântico e Resiliência (Teste de Prompts)

É durante o treinamento e avaliação de RL que os prompts variados serão introduzidos. O objetivo é mensurar se a política otimizada consegue encontrar a equação correta mesmo quando as dicas iniciais são falhas ou excessivas.

* **Prompt Oráculo:** Fornece exatamente o conjunto de operadores necessários.
* **Prompt Distrator:** Injeta deliberadamente operadores não relacionados à solução alvo (ex: sugerir logaritmos para uma função trigonométrica pura).
* **Análise:** Avaliar a degradação da *Constraint Adherence* (aderência ao prompt) versus a precisão do ajuste matemático.

### 5.4. Early Stopping

Para evitar overfitting e otimizar recursos computacionais, as seguintes estratégias de early stopping serão implementadas:

#### A. Critério de Convergência de Recompensa
Parar o treinamento quando a recompensa média não melhorar por N épocas consecutivas:
```
if mean_reward[epoch] <= mean_reward[epoch - patience] + δ:
    stop_training()
```
* **Patience:** 5 épocas
* **Delta (δ):** 0.01 (melhoria mínima significativa)

#### B. Critério de Recuperação Exata
Parar imediatamente se o modelo recuperar a expressão ground-truth (ou equivalente simbólico) com R² > 0.999:
```
if best_r2 >= 0.999 and expression_matches_ground_truth:
    stop_training(reason="exact_recovery")
```

#### C. Critério de Colapso de Política
Parar se a entropia da política cair abaixo de um limiar crítico (indicando mode collapse):
```
if policy_entropy < entropy_threshold:
    stop_training(reason="policy_collapse")
```
* **Entropy Threshold:** 0.1

#### D. Limite Máximo de Steps
Independente dos outros critérios, limitar o treinamento a um número máximo de steps:
* **Max Steps:** 10.000 steps por problema

## 6. Fase 3: Validação de Robustez e Extrapolação (OOD)

As políticas resultantes do treinamento de RL e a melhor configuração da Fase 1 serão submetidas a uma bateria rigorosa de testes no Validation Engine para comprovar a generalização.

### 6.1. Tolerância a Ruído

Os dados alvo (y) dos problemas de Nguyen receberão injeção de ruído gaussiano para simular medições reais.

* Nível 1: Sem ruído (σ = 0).
* Nível 2: Ruído de 1% (σ = 0.01 * std(y)).
* Nível 3: Ruído de 5% (σ = 0.05 * std(y)).

### 6.2. Avaliação de Generalização (Out-of-Distribution - OOD)

Para garantir que o LLM não memorizou um interpolador polinomial local, as expressões finais serão testadas nos seguintes cenários OOD:

**A. OOD de Domínio (Extrapolação Espacial):**
Os 12 problemas de Nguyen serão avaliados em um intervalo deslocado em relação ao treino.

* *Treinamento:* x ∈ [0, 2].
* *Teste OOD:* x ∈ [2, 4].

**B. OOD Estrutural (Novas Equações):**
A avaliação incluirá 3 equações não presentes na distribuição do benchmark de Nguyen para testar a adaptabilidade estrutural profunda da política de RL:

1. **Trigonometria Multivariada Complexa:** f(x₁, x₂) = sin(x₁) * cos(x₂) + sin(x₁ * x₂)
2. **Racional e Exponencial:** f(x) = (x² + 1) / (exp(-x) + 1)
3. **Distância Euclidiana Não-Linear:** f(x₁, x₂) = sqrt(x₁² + x₂²) * log(x₁² + x₂² + 1)

### 6.3. Calibração de Confiança (Log-Probabilidade)

O pipeline registrará a log-probabilidade média das sequências geradas. A análise calculará a correlação entre a confiança do modelo (probabilidade de geração) e o *fitness* da expressão, verificando se os modelos otimizados via BoN-RL desenvolvem a capacidade intrínseca de prever quando uma equação está correta.

## 7. Métricas de Sucesso e Critérios de Avaliação

### 7.1. Métricas Primárias

| Métrica | Definição | Threshold de Sucesso |
|---------|-----------|---------------------|
| **Taxa de Recuperação Exata** | % de problemas onde a expressão ground-truth (ou equivalente) foi encontrada | > 50% dos Nguyen |
| **R² Médio** | Média do melhor R² obtido em cada problema | > 0.95 |
| **R² > 0.99** | % de problemas com R² acima de 0.99 | > 70% |
| **Taxa de Expressões Válidas** | % de expressões geradas que passam no parsing | > 80% após RL |

### 7.2. Métricas Secundárias

| Métrica | Definição |
|---------|-----------|
| **Complexidade Relativa** | Razão entre complexidade da expressão gerada e do ground-truth |
| **Tempo até Convergência** | Número de steps até atingir R² > 0.99 |
| **Estabilidade** | Desvio padrão do R² entre múltiplas seeds |
| **Correlação Confiança-Fitness** | Pearson correlation entre log-prob e R² |

### 7.3. Comparações Estatísticas

Para todas as comparações entre métodos (PPO vs GRPO, BoN vs RL puro, Infix vs Prefix):
* **Número de Seeds:** Mínimo 5 seeds por configuração
* **Teste Estatístico:** Wilcoxon signed-rank test (não-paramétrico)
* **Nível de Significância:** α = 0.05
* **Intervalo de Confiança:** Bootstrap 95% CI para todas as métricas reportadas

## 8. Reprodutibilidade e Hiperparâmetros

### 8.1. Configuração de RL (PPO)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Learning Rate | 1e-5 | Conservador para evitar catastrophic forgetting |
| Batch Size | 64 | Balanceamento memória/variância |
| Mini-batch Size | 16 | 4 updates por batch |
| PPO Epochs | 4 | Padrão para estabilidade |
| Clip Range (ε) | 0.2 | Padrão PPO |
| KL Coefficient | 0.1 | Prevenir divergência excessiva do SFT |
| Value Function Coef | 0.5 | Padrão |
| Entropy Coef | 0.01 | Encorajar exploração |
| Max Grad Norm | 0.5 | Gradient clipping |
| GAE Lambda | 0.95 | Padrão |
| Discount (γ) | 1.0 | Episódio de um passo |

### 8.2. Configuração de RL (GRPO)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Learning Rate | 1e-5 | Consistente com PPO |
| Batch Size | 64 | Consistente com PPO |
| Group Size | 8 | Número de amostras por grupo para ranking |
| KL Coefficient | 0.1 | Consistente com PPO |
| Temperature | 0.7 / Annealing | Variável de teste |

### 8.3. Configuração de BoN-RL

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Buffer Size | 1000 | Manter top-1000 expressões |
| Buffer Strategy | Top-K by R² | Priorizar melhores soluções |
| Refresh Rate | A cada 100 steps | Atualizar buffer periodicamente |
| Sampling from Buffer | 20% do batch | Misturar novas gerações com elite |

### 8.4. Seeds e Reprodutibilidade

* **Random Seeds:** [42, 123, 456, 789, 1337]
* **Torch Deterministic:** Habilitado quando possível
* **Logging de Seeds:** Todas as seeds serão registradas nos metadados

## 9. Logging, Armazenamento e Preservação de Dados

### 9.1. Logging em Tempo Real (Weights & Biases)

Cada experimento registrará em tempo real via W&B:

**Métricas por Step:**
* Recompensa média, mínima, máxima
* R² médio, melhor R² do batch
* Taxa de expressões válidas
* Entropia da política
* KL divergence do modelo base
* Loss (policy, value, total)
* Learning rate atual
* Temperatura atual (se annealing)

**Métricas por Época:**
* Melhor expressão encontrada (string)
* R² da melhor expressão
* Complexidade da melhor expressão
* Distribuição de comprimentos das expressões
* Histograma de recompensas

**Artefatos:**
* Checkpoint do modelo a cada 1000 steps
* Expressões top-10 por época
* Configuração completa do experimento

### 9.2. Estrutura de Diretórios para Resultados

```
results/
└── {experiment_date}/
    └── {algorithm}_{model_size}_{notation}/
        └── {nguyen_problem}/
            └── seed_{seed}/
                ├── config.json          # Configuração completa
                ├── metrics.csv          # Métricas por step
                ├── best_expressions.json # Top expressões
                ├── checkpoints/         # Model checkpoints
                │   ├── step_1000/
                │   ├── step_2000/
                │   └── final/
                └── logs/
                    ├── training.log
                    └── wandb_run_id.txt
```

### 9.3. Upload para HuggingFace Hub

Ao final de cada experimento completo, os seguintes artefatos serão automaticamente enviados ao HuggingFace:

**Modelo Final:**
* Repositório: `augustocsc/seriguela-{algorithm}-{model_size}-{notation}-{problem}`
* Conteúdo: Adapter LoRA final, tokenizer, config

**Dataset de Resultados:**
* Repositório: `augustocsc/seriguela-results-{experiment_date}`
* Conteúdo: Todas as métricas agregadas, melhores expressões, configurações

**Model Card Automático:**
Cada modelo uploaded incluirá model card com:
* Configuração de treinamento
* Métricas de performance
* Exemplos de uso
* Limitações conhecidas

### 9.4. Backup e Redundância

* **Backup Local:** Todos os resultados copiados para disco local antes de terminar instância AWS
* **Backup S3:** Sync automático para bucket S3 a cada 1 hora
* **Versionamento:** Git tags para cada experimento completo

## 10. Cronograma de Execução

### Fase 1: Baseline Best-of-N
* 6 modelos × 12 problemas × 3 valores de N × 2 temperaturas = 432 configurações
* Estimativa: Paralelizável, execução em lote

### Fase 2: RL Training
* 4 algoritmos × 6 modelos × 12 problemas × 5 seeds × 2 penalty strategies × 3 temperature strategies = 8.640 runs
* Estimativa: Distribuído em múltiplas instâncias AWS

### Fase 3: Validação
* Executado apenas nos melhores modelos da Fase 2
* 3 níveis de ruído × 2 domínios (ID + OOD) × 3 equações estruturais

## 11. Implementation Status and Usage Guide

> **For Future Agents:** This section documents what has been implemented and how to run the experiments. All code is located in `2_training/reinforcement/`.

### 11.1. Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| **Reward Functions** | ✅ Complete | `rewards/` |
| R² Clipped Pure | ✅ | `rewards/r2_clipped.py` |
| Length-Penalized R² | ✅ | `rewards/length_penalized.py` |
| SR-IC | ✅ | `rewards/sr_ic.py` |
| **Penalty Strategies** | ✅ Complete | `rewards/penalty.py` |
| Binary Penalty | ✅ | `PenaltyStrategy.BINARY` |
| Gradient Penalty | ✅ | `PenaltyStrategy.GRADIENT` |
| **Temperature Schedulers** | ✅ Complete | `schedulers/temperature.py` |
| Fixed Temperature | ✅ | `FixedTemperature` |
| Linear Annealing | ✅ | `LinearAnnealing` |
| Cosine Annealing | ✅ | `CosineAnnealing` |
| **Early Stopping** | ✅ Complete | `callbacks/early_stopping.py` |
| Convergence Criterion | ✅ | `StopReason.CONVERGENCE` |
| Exact Recovery | ✅ | `StopReason.EXACT_RECOVERY` |
| Policy Collapse | ✅ | `StopReason.POLICY_COLLAPSE` |
| Max Steps | ✅ | `StopReason.MAX_STEPS` |
| **RL Algorithms** | ✅ Complete | `algorithms/` |
| BoN-PPO | ✅ | `algorithms/bon_ppo.py` |
| BoN-GRPO | ✅ | `algorithms/bon_grpo.py` |
| **Elite Buffer** | ✅ Complete | `buffers/elite_buffer.py` |
| **Experiment Runner** | ✅ Complete | `run_experiment.py` |
| **HuggingFace Upload** | ✅ Complete | `utils/hf_upload.py` |
| **AWS Launcher** | ✅ Complete | `aws/launch_rl_experiment.py` |

### 11.2. Directory Structure

```
2_training/reinforcement/
├── algorithms/                 # RL algorithms
│   ├── __init__.py
│   ├── base_trainer.py        # Base trainer with model loading, rollouts
│   ├── bon_ppo.py             # BoN-PPO: PPO + elite buffer hybrid
│   └── bon_grpo.py            # BoN-GRPO: GRPO + elite buffer hybrid
├── rewards/                    # Reward functions
│   ├── __init__.py
│   ├── base.py                # BaseReward, RewardResult, ErrorType
│   ├── r2_clipped.py          # R² Clipped Pure: max(0, R²)
│   ├── length_penalized.py    # R² - α*L
│   ├── sr_ic.py               # -log(MSE+ε) - λ*C
│   └── penalty.py             # PenaltyHandler (Binary/Gradient)
├── schedulers/                 # Temperature scheduling
│   ├── __init__.py
│   └── temperature.py         # Fixed, Linear, Cosine annealing
├── callbacks/                  # Training callbacks
│   ├── __init__.py
│   └── early_stopping.py      # 4 early stopping criteria
├── buffers/                    # Experience buffers
│   ├── __init__.py
│   └── elite_buffer.py        # Top-K expressions by R²
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── hf_upload.py           # HuggingFace model/results upload
│   └── logger.py              # CSV/JSONL experiment logging
├── run_experiment.py           # Main experiment runner (CLI)
└── test_components.py          # Component tests

aws/
└── launch_rl_experiment.py     # AWS EC2 launcher
```

### 11.3. How to Run Experiments

#### Local Testing
```bash
cd 2_training/reinforcement

# Test all components first
python test_components.py

# Run single experiment
python run_experiment.py \
    --algorithm bon_ppo \
    --model augustocsc/gpt2_base_infix_682k \
    --problem nguyen_5 \
    --reward length_penalized \
    --penalty gradient \
    --temperature cosine_annealing \
    --max_steps 5000 \
    --seeds 42 123 456 \
    --use_wandb
```

#### CLI Arguments Reference
| Argument | Values | Default |
|----------|--------|---------|
| `--algorithm` | `bon_ppo`, `bon_grpo` | `bon_ppo` |
| `--model` | HuggingFace repo | `augustocsc/gpt2_base_infix_682k` |
| `--problem` | `nguyen_1` to `nguyen_12` | `nguyen_5` |
| `--reward` | `r2_clipped`, `length_penalized`, `sr_ic` | `length_penalized` |
| `--penalty` | `binary`, `gradient` | `gradient` |
| `--temperature` | `fixed_0.7`, `fixed_0.9`, `linear_annealing`, `cosine_annealing` | `fixed_0.7` |
| `--max_steps` | int | `10000` |
| `--seeds` | list of ints | `[42]` |
| `--use_wandb` | flag | False |
| `--upload_hf` | flag | False |

### 11.4. AWS Deployment

#### Pre-defined Experiments
```bash
cd aws

# List available experiments
python launch_rl_experiment.py --list

# Launch specific experiment
python launch_rl_experiment.py --experiment reward_ablation --instance_type g5.xlarge

# Dry run (show commands without launching)
python launch_rl_experiment.py --experiment nguyen_5_test --dry_run
```

#### Available Experiment Presets
| Preset | Description | Commands |
|--------|-------------|----------|
| `nguyen_5_test` | Quick test on Nguyen-5 | 1 |
| `reward_ablation` | Compare all 3 reward functions | 3 |
| `penalty_ablation` | Compare binary vs gradient penalty | 2 |
| `temperature_ablation` | Compare temperature strategies | 4 |
| `algorithm_comparison` | Compare BoN-PPO vs BoN-GRPO | 2 |
| `all_nguyen` | Best config on all 12 Nguyen problems | 12 |

#### Monitoring AWS Instance
```bash
# SSH to instance
ssh -i ~/.ssh/chave-gpu-nova.pem ubuntu@<PUBLIC_IP>

# Monitor training log
tail -f /var/log/user-data.log

# Check W&B runs
# Visit: https://wandb.ai/seriguela
```

### 11.5. Key Implementation Details

#### Gradient Penalty Values (from `rewards/penalty.py`)
```python
ErrorType.PARSING: -1.0      # Completely malformed
ErrorType.VARIABLES: -0.7    # Wrong variables used
ErrorType.NAN_INF: -0.5      # Numerical instability
ErrorType.NEGATIVE_R2: -0.3  # Worse than baseline
ErrorType.WEAK_R2: 0.0       # Weak but positive
```

#### Temperature Annealing Formulas (from `schedulers/temperature.py`)
- **Linear:** `T(step) = T_max - (T_max - T_min) * (step / total_steps)`
- **Cosine:** `T(step) = T_min + 0.5 * (T_max - T_min) * (1 + cos(π * step / total_steps))`

#### Early Stopping Defaults (from `callbacks/early_stopping.py`)
```python
patience = 5           # Epochs without improvement
delta = 0.01           # Minimum improvement
r2_threshold = 0.999   # Exact recovery threshold
entropy_threshold = 0.1 # Policy collapse detection
max_steps = 10000      # Hard limit
```

### 11.6. Mapping Plan Sections to Code

| Plan Section | Implementation File |
|--------------|---------------------|
| §4.1 R² Clipado Puro | `rewards/r2_clipped.py` |
| §4.2 Length-Penalized R² | `rewards/length_penalized.py` |
| §4.3 SR-IC | `rewards/sr_ic.py` |
| §4.4 Penalidade Binária/Gradiente | `rewards/penalty.py` |
| §5.1 PPO/GRPO/BoN-RL | `algorithms/bon_ppo.py`, `algorithms/bon_grpo.py` |
| §5.2 Temperature Annealing | `schedulers/temperature.py` |
| §5.4 Early Stopping | `callbacks/early_stopping.py` |
| §8.3 Buffer Configuration | `buffers/elite_buffer.py` |
| §9.1-9.3 Logging/Upload | `utils/logger.py`, `utils/hf_upload.py` |

### 11.7. Current Experiment Status (Updated: 2026-02-23)

> **IMPORTANT:** The experiment strategy has been updated to a two-phase approach for efficiency.

#### Phase A: Configuration Search (Base Models Only) - IN PROGRESS

Due to the large configuration space (8,640 configs per model size), we first run the **full factorial experiment on Base models only** to identify the best hyperparameter configurations. This approach:

1. **Reduces compute costs** - Base models run ~2x faster than Large models
2. **Enables faster iteration** - Can identify promising configurations quickly
3. **Provides transferable insights** - Best configs likely generalize across model sizes

**Factorial Design (Base Models):**
- **Models:** 2 (base_infix, base_prefix)
- **Problems:** 3 (nguyen_1, nguyen_5, nguyen_9)
- **Algorithms:** 5 (bon_ppo, bon_grpo, pure_ppo, pure_grpo, best_of_n)
- **Rewards:** 3 (r2_clipped, length_penalized, sr_ic)
- **Penalties:** 2 (binary, gradient)
- **Temperatures:** 4 (fixed_0.7, fixed_0.9, linear_annealing, cosine_annealing)
- **Prompts:** 3 (standard, oracle, distractor)
- **Noise levels:** 4 (0.0, 0.01, 0.05, 0.1)
- **Total:** 2 × 3 × 5 × 3 × 2 × 4 × 3 × 4 = **8,640 configurations**

**Current Progress:**
- Completed: ~2,500 / 8,640 configs (~29%)
- Running: 6 AWS instances (g5.xlarge)
- ETA: ~8-10 hours remaining
- Results: Uploading to `augustocsc/seriguela-results` on HuggingFace

**Key Files:**
- `run_remaining_experiment.py` - Runs only remaining configs from JSON
- `remaining_base_configs.json` - List of configs not yet completed
- `aws/launch_rl_experiment.py` - AWS launcher with `rem_base_*` presets

#### Phase B: Full Benchmark Evaluation (Medium/Large Models) - PENDING

After Phase A completes, the **top-K configurations** will be applied to:

1. **Medium models** (355M params) - `gpt2_medium_infix_682k`, `gpt2_medium_prefix_682k`
2. **Large models** (774M params) - `gpt2_large_infix_682k`, `gpt2_large_prefix_682k`

**Full Benchmark Coverage:**
- All 12 Nguyen problems (nguyen_1 through nguyen_12)
- Top 3-5 configurations from Phase A
- Multiple seeds for statistical significance

This two-phase approach reduces total compute from ~50,000+ runs to ~10,000 runs while maintaining scientific rigor.

### 11.8. Next Steps for Future Agents

1. **Complete Phase A:** Monitor the 6 running instances until all 8,640 Base model configs complete.

2. **Analyze Phase A Results:** Identify best configurations by:
   - Highest test R² per problem
   - Best generalization (train vs test gap)
   - Robustness across noise levels
   - Prompt sensitivity analysis

3. **Launch Phase B:** Apply top configurations to Medium/Large models on full Nguyen benchmark (12 problems).

4. **Statistical Analysis:** Aggregate results from W&B and HuggingFace for Wilcoxon tests, bootstrap CI.

5. **Phase 3 Validation:** Implement OOD testing and noise robustness evaluation using the best configurations.

### 11.9. Related Documentation

- **Implementation Plan:** `docs/proposals/implementation_plan.md` - Detailed technical plan
- **Training Guide:** `2_training/README.md` - General training documentation
- **CLAUDE.md:** Root-level instructions for working with the codebase

## 12. Referências e Recursos

* **Código:** https://github.com/augustocsc/seriguela
* **Modelos SFT:** https://huggingface.co/augustocsc (6 modelos base)
* **Dataset:** https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
* **W&B Project:** seriguela
