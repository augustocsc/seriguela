# Plano de Implementação: Pipeline Experimental de RL para Regressão Simbólica

**Baseado em:** `docs/proposals/main_experiment.md`
**Data:** 2026-02-21
**Status:** A implementar

---

## Resumo Executivo

A base de código está ~85% completa. Os 6 modelos SFT estão treinados e disponíveis no HuggingFace. O framework de avaliação está funcional. As principais lacunas são:

1. **Sistema de Recompensas** - Apenas R² básico implementado
2. **Métodos Híbridos BoN-RL** - Não implementados
3. **Temperature Annealing** - Não implementado
4. **Validação OOD** - Não implementada
5. **Estatísticas Avançadas** - Wilcoxon/Bootstrap faltando

---

## Fase 0: Limpeza e Reorganização

### 0.1. Scripts a Remover (Deprecated)

```
2_training/reinforcement/ppo_experiment_legacy.py    # Substituído por ppo_symbolic.py
2_training/reinforcement/ppo_experiment_v2.py       # Versão intermediária
2_training/reinforcement/debug_reinforce.py         # Script de debug
2_training/supervised/train_test.py                 # Se existir, é teste
src/seriguela/                                      # Duplica classes/ - avaliar remoção
```

### 0.2. Estrutura de Diretórios Final

```
seriguela/
├── 1_data/                          # ✅ Completo
│   ├── benchmarks/                  # Nguyen, Feynman, PMLB
│   ├── processed/                   # 682K dataset
│   └── raw/
├── 2_training/
│   ├── supervised/                  # ✅ Completo
│   │   └── train_with_json.py       # Script principal
│   ├── reinforcement/               # 🔧 A completar
│   │   ├── algorithms/              # NOVO: Módulo de algoritmos
│   │   │   ├── __init__.py
│   │   │   ├── ppo.py               # PPO refatorado
│   │   │   ├── grpo.py              # GRPO refatorado
│   │   │   ├── bon_ppo.py           # NOVO: BoN-PPO híbrido
│   │   │   └── bon_grpo.py          # NOVO: BoN-GRPO híbrido
│   │   ├── rewards/                 # NOVO: Sistema de recompensas
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # Interface base
│   │   │   ├── r2_clipped.py        # R² clipado
│   │   │   ├── length_penalized.py  # R² com penalidade de comprimento
│   │   │   ├── sr_ic.py             # SR Information Criterion
│   │   │   └── penalty.py           # Estratégias de penalidade
│   │   ├── schedulers/              # NOVO: Schedulers
│   │   │   ├── __init__.py
│   │   │   └── temperature.py       # Annealing strategies
│   │   ├── buffers/                 # NOVO: Replay buffers
│   │   │   ├── __init__.py
│   │   │   └── elite_buffer.py      # Buffer de melhores expressões
│   │   ├── callbacks/               # NOVO: Callbacks de treino
│   │   │   ├── __init__.py
│   │   │   └── early_stopping.py    # 4 critérios de parada
│   │   └── run_experiment.py        # NOVO: Script principal unificado
│   └── configs/
├── 3_evaluation/                    # ✅ Completo (minor additions)
│   ├── core/                        # Métricas e validação
│   ├── benchmarks/                  # Nguyen benchmarks
│   ├── validation/                  # NOVO: Validação OOD
│   │   ├── __init__.py
│   │   ├── ood_domain.py            # Extrapolação de domínio
│   │   ├── ood_structural.py        # Novas equações
│   │   ├── noise_robustness.py      # Injeção de ruído
│   │   └── confidence.py            # Calibração log-prob
│   ├── cli.py
│   └── commands/
├── 4_analysis/                      # ✅ Parcial (additions)
│   ├── statistical/
│   │   ├── wilcoxon_tests.py        # NOVO: Testes não-paramétricos
│   │   └── bootstrap_ci.py          # NOVO: Intervalos de confiança
│   ├── visualization/
│   └── complexity/
├── classes/                         # ✅ Completo
│   ├── expression.py                # Parser de expressões
│   └── dataset.py                   # Dataset utilities
├── configs/                         # ✅ Completo
└── docs/
    └── proposals/
        ├── main_experiment.md       # Plano experimental
        └── implementation_plan.md   # Este arquivo
```

---

## Fase 1: Sistema de Recompensas

**Prioridade:** CRÍTICA
**Objetivo:** Implementar as 3 funções de recompensa + estratégias de penalidade

### 1.1. Interface Base de Recompensa

**Arquivo:** `2_training/reinforcement/rewards/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RewardResult:
    """Resultado do cálculo de recompensa."""
    reward: float
    r2: float
    is_valid: bool
    complexity: int
    error_type: Optional[str] = None  # "parsing", "variables", "nan_inf", "negative_r2"

class BaseReward(ABC):
    """Interface base para funções de recompensa."""

    @abstractmethod
    def compute(self, expression: str, x: np.ndarray, y: np.ndarray) -> RewardResult:
        """Calcula a recompensa para uma expressão."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Nome da função de recompensa."""
        pass
```

### 1.2. R² Clipado Puro

**Arquivo:** `2_training/reinforcement/rewards/r2_clipped.py`

```python
class R2ClippedReward(BaseReward):
    """R² clipado em [0, 1]."""

    def compute(self, expression: str, x: np.ndarray, y: np.ndarray) -> RewardResult:
        # 1. Parse expression
        # 2. Fit constants
        # 3. Calculate R²
        # 4. Return max(0, R²)
        pass

    def name(self) -> str:
        return "r2_clipped"
```

### 1.3. R² com Penalidade de Comprimento

**Arquivo:** `2_training/reinforcement/rewards/length_penalized.py`

```python
class LengthPenalizedReward(BaseReward):
    """R² - α * L (penaliza expressões longas)."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def compute(self, expression: str, x: np.ndarray, y: np.ndarray) -> RewardResult:
        # R_length = R² - α * num_tokens
        pass
```

### 1.4. SR Information Criterion

**Arquivo:** `2_training/reinforcement/rewards/sr_ic.py`

```python
class SRICReward(BaseReward):
    """Symbolic Regression Information Criterion."""

    def __init__(self, lambda_complexity: float = 0.1, epsilon: float = 1e-10):
        self.lambda_c = lambda_complexity
        self.epsilon = epsilon

    def compute(self, expression: str, x: np.ndarray, y: np.ndarray) -> RewardResult:
        # R_SRIC = -log(MSE + ε) - λ * complexity
        pass
```

### 1.5. Estratégias de Penalidade

**Arquivo:** `2_training/reinforcement/rewards/penalty.py`

```python
from enum import Enum

class PenaltyStrategy(Enum):
    BINARY = "binary"      # Tudo inválido = -1.0
    GRADIENT = "gradient"  # Penalidades diferenciadas

class PenaltyHandler:
    """Gerencia penalidades para expressões inválidas."""

    GRADIENT_PENALTIES = {
        "parsing": -1.0,      # Sintaxe inválida
        "variables": -0.7,    # Variáveis incorretas
        "nan_inf": -0.5,      # Produz NaN/Inf
        "negative_r2": -0.3,  # R² < 0
        "weak_r2": 0.0,       # R² ∈ [0, 0.5)
    }

    def __init__(self, strategy: PenaltyStrategy = PenaltyStrategy.BINARY):
        self.strategy = strategy

    def get_penalty(self, error_type: str) -> float:
        if self.strategy == PenaltyStrategy.BINARY:
            return -1.0
        return self.GRADIENT_PENALTIES.get(error_type, -1.0)
```

---

## Fase 2: Temperature Annealing

**Prioridade:** ALTA
**Objetivo:** Implementar schedulers de temperatura

### 2.1. Schedulers de Temperatura

**Arquivo:** `2_training/reinforcement/schedulers/temperature.py`

```python
from abc import ABC, abstractmethod
import math

class TemperatureScheduler(ABC):
    """Interface para schedulers de temperatura."""

    @abstractmethod
    def get_temperature(self, step: int, total_steps: int) -> float:
        pass

class FixedTemperature(TemperatureScheduler):
    """Temperatura fixa durante todo treinamento."""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    def get_temperature(self, step: int, total_steps: int) -> float:
        return self.temperature

class LinearAnnealing(TemperatureScheduler):
    """Redução linear: T_max → T_min."""

    def __init__(self, t_max: float = 1.0, t_min: float = 0.5):
        self.t_max = t_max
        self.t_min = t_min

    def get_temperature(self, step: int, total_steps: int) -> float:
        progress = step / max(total_steps, 1)
        return self.t_max - (self.t_max - self.t_min) * progress

class CosineAnnealing(TemperatureScheduler):
    """Redução cosine (mais suave)."""

    def __init__(self, t_max: float = 1.0, t_min: float = 0.5):
        self.t_max = t_max
        self.t_min = t_min

    def get_temperature(self, step: int, total_steps: int) -> float:
        progress = step / max(total_steps, 1)
        return self.t_min + 0.5 * (self.t_max - self.t_min) * (1 + math.cos(math.pi * progress))
```

---

## Fase 3: Early Stopping

**Prioridade:** ALTA
**Objetivo:** Implementar os 4 critérios de parada

### 3.1. Callbacks de Early Stopping

**Arquivo:** `2_training/reinforcement/callbacks/early_stopping.py`

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class StopReason(Enum):
    CONVERGENCE = "convergence"
    EXACT_RECOVERY = "exact_recovery"
    POLICY_COLLAPSE = "policy_collapse"
    MAX_STEPS = "max_steps"
    NONE = "none"

@dataclass
class EarlyStoppingConfig:
    # Convergência
    patience: int = 5
    delta: float = 0.01

    # Recuperação exata
    r2_threshold: float = 0.999
    check_symbolic_match: bool = True

    # Colapso de política
    entropy_threshold: float = 0.1

    # Limite máximo
    max_steps: int = 10000

class EarlyStoppingCallback:
    """Gerencia múltiplos critérios de early stopping."""

    def __init__(self, config: EarlyStoppingConfig, ground_truth: Optional[str] = None):
        self.config = config
        self.ground_truth = ground_truth
        self.reward_history: List[float] = []
        self.best_r2 = -float('inf')
        self.best_expression = None

    def check(self, step: int, mean_reward: float, best_r2: float,
              best_expr: str, policy_entropy: float) -> StopReason:
        """Verifica todos os critérios de parada."""

        # 1. Max steps
        if step >= self.config.max_steps:
            return StopReason.MAX_STEPS

        # 2. Exact recovery
        if best_r2 >= self.config.r2_threshold:
            if self._check_symbolic_match(best_expr):
                return StopReason.EXACT_RECOVERY

        # 3. Policy collapse
        if policy_entropy < self.config.entropy_threshold:
            return StopReason.POLICY_COLLAPSE

        # 4. Convergence
        self.reward_history.append(mean_reward)
        if len(self.reward_history) > self.config.patience:
            old_reward = self.reward_history[-self.config.patience - 1]
            if mean_reward <= old_reward + self.config.delta:
                return StopReason.CONVERGENCE

        return StopReason.NONE

    def _check_symbolic_match(self, expression: str) -> bool:
        """Verifica equivalência simbólica com ground truth."""
        if not self.config.check_symbolic_match or not self.ground_truth:
            return True  # Se não verificar, assume match
        # TODO: Usar SymPy para verificar equivalência
        return True
```

---

## Fase 4: Buffer de Elite (BoN-RL)

**Prioridade:** CRÍTICA
**Objetivo:** Implementar buffer para métodos híbridos BoN-RL

### 4.1. Elite Buffer

**Arquivo:** `2_training/reinforcement/buffers/elite_buffer.py`

```python
from dataclasses import dataclass
from typing import List, Tuple
import heapq

@dataclass
class BufferEntry:
    expression: str
    r2: float
    reward: float
    log_prob: float

    def __lt__(self, other):
        return self.r2 < other.r2  # Min-heap by R²

class EliteBuffer:
    """Buffer das melhores expressões para BoN-RL."""

    def __init__(self, max_size: int = 1000, sample_ratio: float = 0.2):
        self.max_size = max_size
        self.sample_ratio = sample_ratio  # % do batch vindo do buffer
        self.buffer: List[BufferEntry] = []

    def add(self, entry: BufferEntry):
        """Adiciona entrada ao buffer (mantém top-K)."""
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, entry)
        elif entry.r2 > self.buffer[0].r2:
            heapq.heapreplace(self.buffer, entry)

    def add_batch(self, expressions: List[str], r2_scores: List[float],
                  rewards: List[float], log_probs: List[float]):
        """Adiciona batch de expressões."""
        for expr, r2, reward, lp in zip(expressions, r2_scores, rewards, log_probs):
            if r2 > 0:  # Só adiciona expressões válidas
                self.add(BufferEntry(expr, r2, reward, lp))

    def sample(self, batch_size: int) -> List[BufferEntry]:
        """Amostra do buffer para compor batch de treino."""
        n_from_buffer = int(batch_size * self.sample_ratio)
        if len(self.buffer) == 0:
            return []
        # Amostra aleatória do buffer
        import random
        return random.sample(self.buffer, min(n_from_buffer, len(self.buffer)))

    def get_best(self, k: int = 10) -> List[BufferEntry]:
        """Retorna as K melhores expressões."""
        return heapq.nlargest(k, self.buffer, key=lambda x: x.r2)

    def stats(self) -> dict:
        """Estatísticas do buffer."""
        if not self.buffer:
            return {"size": 0, "mean_r2": 0, "max_r2": 0, "min_r2": 0}
        r2_values = [e.r2 for e in self.buffer]
        return {
            "size": len(self.buffer),
            "mean_r2": sum(r2_values) / len(r2_values),
            "max_r2": max(r2_values),
            "min_r2": min(r2_values),
        }
```

---

## Fase 5: Algoritmos BoN-RL Híbridos

**Prioridade:** CRÍTICA
**Objetivo:** Implementar BoN-PPO e BoN-GRPO

### 5.1. BoN-PPO

**Arquivo:** `2_training/reinforcement/algorithms/bon_ppo.py`

```python
class BoNPPOTrainer:
    """PPO com buffer de Best-of-N."""

    def __init__(
        self,
        model,
        tokenizer,
        reward_fn: BaseReward,
        penalty_strategy: PenaltyStrategy,
        temp_scheduler: TemperatureScheduler,
        early_stopping: EarlyStoppingCallback,
        elite_buffer: EliteBuffer,
        config: PPOConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.penalty = PenaltyHandler(penalty_strategy)
        self.temp_scheduler = temp_scheduler
        self.early_stopping = early_stopping
        self.buffer = elite_buffer
        self.config = config

    def train_step(self, step: int, total_steps: int, x: np.ndarray, y: np.ndarray, prompt: str):
        """Um passo de treinamento BoN-PPO."""

        # 1. Get current temperature
        temperature = self.temp_scheduler.get_temperature(step, total_steps)

        # 2. Generate new expressions
        new_expressions = self._generate(prompt, temperature, self.config.batch_size)

        # 3. Sample from elite buffer
        buffer_samples = self.buffer.sample(self.config.batch_size)

        # 4. Combine: 80% new + 20% buffer
        batch_expressions = new_expressions + [s.expression for s in buffer_samples]

        # 5. Compute rewards
        rewards = [self.reward_fn.compute(expr, x, y) for expr in batch_expressions]

        # 6. Apply penalty strategy
        final_rewards = self._apply_penalties(rewards)

        # 7. Update buffer with good expressions
        self.buffer.add_batch(...)

        # 8. PPO update
        loss = self._ppo_update(batch_expressions, final_rewards)

        # 9. Check early stopping
        stop_reason = self.early_stopping.check(...)

        return loss, stop_reason
```

### 5.2. BoN-GRPO

**Arquivo:** `2_training/reinforcement/algorithms/bon_grpo.py`

Similar ao BoN-PPO, mas usando GRPO (group-relative advantages) para o update.

---

## Fase 6: Validação OOD

**Prioridade:** MÉDIA
**Objetivo:** Implementar testes fora de distribuição

### 6.1. OOD de Domínio

**Arquivo:** `3_evaluation/validation/ood_domain.py`

```python
class DomainOODValidator:
    """Testa extrapolação em domínio diferente."""

    def __init__(self, train_domain: Tuple[float, float] = (0, 2),
                 test_domain: Tuple[float, float] = (2, 4)):
        self.train_domain = train_domain
        self.test_domain = test_domain

    def evaluate(self, model, nguyen_problems: List[str]) -> dict:
        """Avalia modelo em domínio OOD."""
        results = {}
        for problem in nguyen_problems:
            # Generate test data in OOD domain
            x_ood = np.linspace(self.test_domain[0], self.test_domain[1], 100)
            y_ood = self._get_ground_truth(problem, x_ood)

            # Generate expressions and evaluate
            ...
        return results
```

### 6.2. OOD Estrutural

**Arquivo:** `3_evaluation/validation/ood_structural.py`

```python
STRUCTURAL_OOD_EQUATIONS = [
    {
        "name": "trig_multivar",
        "equation": "sin(x_1) * cos(x_2) + sin(x_1 * x_2)",
        "vars": ["x_1", "x_2"],
    },
    {
        "name": "rational_exp",
        "equation": "(x**2 + 1) / (exp(-x) + 1)",
        "vars": ["x"],
    },
    {
        "name": "euclidean_nonlinear",
        "equation": "sqrt(x_1**2 + x_2**2) * log(x_1**2 + x_2**2 + 1)",
        "vars": ["x_1", "x_2"],
    },
]

class StructuralOODValidator:
    """Testa em equações fora da distribuição de Nguyen."""

    def evaluate(self, model, equations: List[dict] = STRUCTURAL_OOD_EQUATIONS) -> dict:
        ...
```

### 6.3. Robustez a Ruído

**Arquivo:** `3_evaluation/validation/noise_robustness.py`

```python
class NoiseRobustnessValidator:
    """Testa robustez a ruído gaussiano."""

    NOISE_LEVELS = [0.0, 0.01, 0.05]  # 0%, 1%, 5%

    def add_noise(self, y: np.ndarray, noise_level: float) -> np.ndarray:
        """Adiciona ruído gaussiano proporcional."""
        if noise_level == 0:
            return y
        sigma = noise_level * np.std(y)
        noise = np.random.normal(0, sigma, y.shape)
        return y + noise

    def evaluate(self, model, problem: str) -> dict:
        """Avalia em múltiplos níveis de ruído."""
        results = {}
        for noise_level in self.NOISE_LEVELS:
            y_noisy = self.add_noise(y_true, noise_level)
            ...
        return results
```

### 6.4. Calibração de Confiança

**Arquivo:** `3_evaluation/validation/confidence.py`

```python
class ConfidenceCalibrator:
    """Analisa correlação entre log-prob e fitness."""

    def analyze(self, expressions: List[str], log_probs: List[float],
                r2_scores: List[float]) -> dict:
        """Calcula métricas de calibração."""
        from scipy.stats import pearsonr, spearmanr

        # Filtra expressões válidas
        valid_mask = [r2 >= 0 for r2 in r2_scores]
        valid_lp = [lp for lp, v in zip(log_probs, valid_mask) if v]
        valid_r2 = [r2 for r2, v in zip(r2_scores, valid_mask) if v]

        pearson_corr, p_value = pearsonr(valid_lp, valid_r2)
        spearman_corr, _ = spearmanr(valid_lp, valid_r2)

        return {
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "p_value": p_value,
            "n_valid": len(valid_r2),
        }
```

---

## Fase 7: Estatísticas Avançadas

**Prioridade:** MÉDIA
**Objetivo:** Implementar testes não-paramétricos e bootstrap CI

### 7.1. Testes Não-Paramétricos

**Arquivo:** `4_analysis/statistical/wilcoxon_tests.py`

```python
from scipy.stats import wilcoxon, mannwhitneyu
import numpy as np

def compare_methods(method_a_scores: List[float], method_b_scores: List[float],
                    alpha: float = 0.05) -> dict:
    """Compara dois métodos usando Wilcoxon signed-rank test."""

    stat, p_value = wilcoxon(method_a_scores, method_b_scores)

    # Effect size (r = Z / sqrt(N))
    n = len(method_a_scores)
    z = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
    effect_size = abs(z) / np.sqrt(n)

    return {
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size_r": effect_size,
        "method_a_median": np.median(method_a_scores),
        "method_b_median": np.median(method_b_scores),
    }
```

### 7.2. Bootstrap Confidence Intervals

**Arquivo:** `4_analysis/statistical/bootstrap_ci.py`

```python
import numpy as np
from typing import Callable

def bootstrap_ci(data: np.ndarray, statistic: Callable = np.mean,
                 n_bootstrap: int = 10000, confidence: float = 0.95) -> dict:
    """Calcula intervalo de confiança via bootstrap."""

    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence

    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        "point_estimate": statistic(data),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "std_error": np.std(bootstrap_stats),
    }
```

---

## Fase 8: Script Principal Unificado

**Prioridade:** ALTA
**Objetivo:** Script único para executar todos os experimentos

### 8.1. Script de Experimento

**Arquivo:** `2_training/reinforcement/run_experiment.py`

```python
"""
Script principal para executar experimentos de RL.

Uso:
    python run_experiment.py --config experiment_config.yaml

    python run_experiment.py \
        --algorithm bon_ppo \
        --model augustocsc/gpt2_base_infix_682k \
        --reward length_penalized \
        --penalty gradient \
        --temperature cosine_annealing \
        --problem nguyen_5 \
        --seeds 42 123 456 789 1337
"""

import argparse
from pathlib import Path
import wandb
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()

    # Algoritmo
    parser.add_argument("--algorithm", choices=["ppo", "grpo", "bon_ppo", "bon_grpo"],
                        required=True)

    # Modelo
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model repo")

    # Recompensa
    parser.add_argument("--reward", choices=["r2_clipped", "length_penalized", "sr_ic"],
                        default="r2_clipped")
    parser.add_argument("--reward_alpha", type=float, default=0.01,
                        help="Alpha for length penalty")
    parser.add_argument("--reward_lambda", type=float, default=0.1,
                        help="Lambda for SR-IC complexity")

    # Penalidade
    parser.add_argument("--penalty", choices=["binary", "gradient"],
                        default="binary")

    # Temperatura
    parser.add_argument("--temperature", choices=["fixed_0.7", "fixed_0.9",
                                                   "linear_annealing", "cosine_annealing"],
                        default="fixed_0.7")

    # Problema
    parser.add_argument("--problem", type=str, required=True,
                        help="Nguyen problem (e.g., nguyen_5)")

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1337])

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--upload_hf", action="store_true")

    args = parser.parse_args()

    # Run experiment for each seed
    for seed in args.seeds:
        run_single_experiment(args, seed)

def run_single_experiment(args, seed: int):
    """Executa um experimento com seed específica."""

    # 1. Setup
    set_seed(seed)

    # 2. Initialize WandB
    run_name = f"seriguela-{args.algorithm}-{args.model.split('/')[-1]}-{args.problem}-seed{seed}"
    wandb.init(project="seriguela", name=run_name, config=vars(args))

    # 3. Load model
    model, tokenizer = load_model(args.model)

    # 4. Setup reward function
    reward_fn = create_reward(args.reward, args.reward_alpha, args.reward_lambda)

    # 5. Setup trainer
    trainer = create_trainer(args.algorithm, model, tokenizer, reward_fn, ...)

    # 6. Train
    results = trainer.train()

    # 7. Save results
    save_results(results, args.output_dir, args, seed)

    # 8. Upload to HuggingFace
    if args.upload_hf:
        upload_to_hf(results, args)

    wandb.finish()
```

---

## Fase 9: Logging e HuggingFace Upload

**Prioridade:** ALTA
**Objetivo:** Sistema completo de logging e preservação

### 9.1. Logger Estruturado

**Arquivo:** `2_training/reinforcement/utils/logger.py`

```python
import json
import csv
from pathlib import Path
from datetime import datetime
import wandb

class ExperimentLogger:
    """Logger estruturado para experimentos."""

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.metrics_file = self.output_dir / "metrics.csv"
        self.expressions_file = self.output_dir / "expressions.jsonl"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def log_step(self, step: int, metrics: dict):
        """Log métricas de um step."""
        # CSV
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + list(metrics.keys()))
            writer.writerow({"step": step, **metrics})

        # WandB
        wandb.log(metrics, step=step)

    def log_expression(self, step: int, expression: str, r2: float, reward: float):
        """Log uma expressão gerada."""
        with open(self.expressions_file, "a") as f:
            json.dump({"step": step, "expr": expression, "r2": r2, "reward": reward}, f)
            f.write("\n")

    def save_checkpoint(self, model, step: int):
        """Salva checkpoint do modelo."""
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
```

### 9.2. HuggingFace Uploader

**Arquivo:** `2_training/reinforcement/utils/hf_upload.py`

```python
from huggingface_hub import HfApi, create_repo
from pathlib import Path

class HuggingFaceUploader:
    """Upload automático para HuggingFace Hub."""

    def __init__(self, username: str = "augustocsc"):
        self.api = HfApi()
        self.username = username

    def upload_model(self, model_dir: Path, repo_name: str):
        """Upload modelo treinado."""
        repo_id = f"{self.username}/{repo_name}"

        # Create repo if not exists
        create_repo(repo_id, exist_ok=True, private=False)

        # Upload
        self.api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
        )

        return f"https://huggingface.co/{repo_id}"

    def upload_results(self, results_dir: Path, dataset_name: str):
        """Upload dataset de resultados."""
        repo_id = f"{self.username}/{dataset_name}"

        create_repo(repo_id, exist_ok=True, private=False, repo_type="dataset")

        self.api.upload_folder(
            folder_path=str(results_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )

        return f"https://huggingface.co/datasets/{repo_id}"
```

---

## Cronograma de Implementação

| Fase | Componente | Arquivos | Dependências |
|------|------------|----------|--------------|
| 0 | Limpeza | - | Nenhuma |
| 1 | Sistema de Recompensas | 5 arquivos | Fase 0 |
| 2 | Temperature Annealing | 1 arquivo | Nenhuma |
| 3 | Early Stopping | 1 arquivo | Nenhuma |
| 4 | Elite Buffer | 1 arquivo | Nenhuma |
| 5 | BoN-RL Algoritmos | 2 arquivos | Fases 1-4 |
| 6 | Validação OOD | 4 arquivos | Fase 5 |
| 7 | Estatísticas | 2 arquivos | Nenhuma |
| 8 | Script Principal | 1 arquivo | Fases 1-7 |
| 9 | Logging/Upload | 2 arquivos | Fase 8 |

---

## Ordem de Execução Recomendada

1. **Fase 0:** Limpar scripts deprecated
2. **Fases 1-4:** Implementar em paralelo (independentes)
3. **Fase 5:** Integrar algoritmos BoN-RL
4. **Fase 8:** Script principal
5. **Fase 9:** Logging e upload
6. **Fases 6-7:** Validação e estatísticas
7. **Testes:** Rodar experimento piloto em Nguyen-1

---

## Comandos de Exemplo (Pós-Implementação)

```bash
# Experimento completo: BoN-PPO com todas as variações
python 2_training/reinforcement/run_experiment.py \
    --algorithm bon_ppo \
    --model augustocsc/gpt2_base_infix_682k \
    --reward length_penalized \
    --penalty gradient \
    --temperature cosine_annealing \
    --problem nguyen_5 \
    --seeds 42 123 456 789 1337 \
    --upload_hf

# Baseline Best-of-N
python 2_training/reinforcement/best_of_n_experiment.py \
    --model augustocsc/gpt2_large_prefix_682k \
    --n_samples 10000 \
    --temperature 0.7 \
    --problem nguyen_1

# Validação OOD
python 3_evaluation/validation/run_ood_suite.py \
    --model_dir results/bon_ppo_base_infix/nguyen_5/seed_42/final \
    --output results/ood_validation/
```

---

## Verificação de Completude

Após implementação, verificar:

- [ ] 3 funções de recompensa funcionando
- [ ] 2 estratégias de penalidade testadas
- [ ] 3 schedulers de temperatura implementados
- [ ] 4 critérios de early stopping ativos
- [ ] Buffer de elite funcionando
- [ ] BoN-PPO e BoN-GRPO treinando
- [ ] Logs salvos em formato correto
- [ ] Upload automático para HuggingFace
- [ ] Validação OOD executando
- [ ] Estatísticas Wilcoxon e Bootstrap funcionando
