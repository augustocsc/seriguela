# Plano de Avaliação: PPO Finetuning (Bloco 3)

**Data:** 2026-02-01
**Modelo Base:** V2 (`augustocsc/Se124M_700K_infix_v2`) - 90% valid rate
**Objetivo:** Avaliar se o PPO está funcionando para encontrar expressões específicas

---

## Entendimento da Arquitetura

### Os 3 Blocos do Projeto

#### Bloco 1: Dados ✅ (Completo)
- **Objetivo:** Preparar e analisar dados de expressões matemáticas
- **Output:** 947K expressões com markers `<|endofex|>`
- **Status:** 100% validado
- **Localização:** `data/processed/700K_fixed/`

#### Bloco 2: Treinamento (Supervised Fine-tuning) ✅ (Completo)
- **Objetivo:** Treinar LLM para **gerar expressões válidas** (sintaxe correta)
- **Método:** LoRA fine-tuning com dataset sintético
- **Modelos:** V1 (83.3%), V2 (90% com nucleus sampling)
- **Status:** Production-ready
- **Script:** `scripts/train.py`

#### Bloco 3: Finetuning PPO (Reinforcement Learning) 🔍 (A Avaliar)
- **Objetivo:** Aprimorar modelo para **encontrar expressões específicas** que fitam dados reais
- **Método:** PPO (Proximal Policy Optimization) com reward R²
- **Tarefa:** Symbolic Regression (regressão simbólica)
- **Status:** Implementado, precisa avaliação
- **Script:** `scripts/symbolic_rl/trainer.py`

---

## Como o PPO Funciona no Projeto

### Diferença Chave: Bloco 2 vs Bloco 3

| Aspecto | Bloco 2 (Supervised) | Bloco 3 (PPO/RL) |
|---------|---------------------|------------------|
| **Objetivo** | Gerar expressões **válidas** | Gerar expressões que **fitam dados** |
| **Entrada** | Prompt com vars/operators | Prompt + Dataset (X, y) |
| **Saída** | Expressão sintática | Expressão que aproxima f(X) ≈ y |
| **Feedback** | Loss (cross-entropy) | Reward (R² score) |
| **Exemplo** | `x_1*sin(x_2) + C` | `1.5*x_1*sin(0.7*x_2) + 0.3` |
| **Avaliação** | Sintaxe correta? | Quão bem fita os dados? |

### Symbolic Regression: O Que Significa "Encontrar Expressão Específica"

**Problema:**
Dado um dataset (X, y) onde:
- X = variáveis de entrada (e.g., x_1, x_2, x_3)
- y = variável alvo

**Objetivo:**
Encontrar expressão matemática f(X) tal que f(X) ≈ y

**Exemplo Concreto (Feynman Equation):**
```
Dataset: feynman-i.18.16.txt
X = [x_1, x_2]  # Input variables
y = target      # Output to fit

Expressão Ideal: x_1*sin(x_2)
Reward: R² = 0.95 (95% da variância explicada)
```

### Reward Function (O Coração do PPO)

```python
def compute_reward(expression_str: str, X, y) -> float:
    """
    1. Parse expression: "x_1*sin(x_2) + C"
    2. Check validity: No NaN/Inf when evaluated on X
    3. Fit constants: Optimize C to minimize error
    4. Compute R²: How well the expression fits y
    5. Return reward: -1 (invalid) to 1.0 (perfect fit)
    """
    try:
        expr = Expression(expression_str)

        # Validate on dataset
        if expr.is_valid_on_dataset(X):
            # Fit constants using L-BFGS-B
            r2_score = expr.fit_constants(X, y)
            return max(0.1, r2_score)
        else:
            return -1.0  # Invalid expression
    except:
        return -1.0
```

**Reward Scale:**
- **1.0**: Perfeito (100% fit)
- **0.9**: Excelente (90% da variância explicada)
- **0.5**: Razoável
- **0.0**: Baseline (média)
- **-1.0**: Inválida (NaN/Inf)

### PPO Training Loop

```
1. Generate expression from prompt
   Input:  "vars: x_1, x_2\noper: *, sin\ncons: C\nexpr:"
   Output: "x_1*sin(x_2) + C"

2. Evaluate on dataset
   - Load data: X, y
   - Compute reward: R² = 0.85

3. PPO Update
   - Adjust model parameters to increase reward
   - Maintain KL divergence from reference model

4. Repeat until reward >= 0.9 (stopping criterion)
```

---

## Estratégia de Avaliação do PPO

### Questões a Responder

1. **O PPO consegue melhorar as expressões?**
   - Comparar: Expressões antes vs depois do PPO
   - Métrica: R² score médio

2. **O modelo converge para soluções corretas?**
   - Verificar se encontra equações conhecidas (Feynman)
   - Métrica: Reward >= 0.9 atingido?

3. **V2 é melhor base que V1 para PPO?**
   - Comparar PPO usando V2 vs V1
   - Métrica: Taxa de convergência, reward final

4. **O PPO generaliza para novos datasets?**
   - Testar em datasets não vistos
   - Métrica: R² em test set

### Experimentos Propostos

#### Experimento 1: Baseline (Sem PPO) - V2 Zero-Shot

**Objetivo:** Ver quão bem V2 faz symbolic regression sem PPO

**Método:**
```python
# Use V2 com nucleus sampling (90% valid)
model = "augustocsc/Se124M_700K_infix_v2"
config = {"temperature": 0.7, "top_p": 0.8}

# Generate 100 expressions for dataset
for i in range(100):
    expr = generate(prompt, model, config)
    reward = compute_reward(expr, X, y)

# Metrics
best_reward = max(rewards)
avg_reward = mean(rewards)
valid_rate = count(rewards > 0) / 100
```

**Expectativa:**
- Valid rate: ~90% (já sabemos)
- Avg reward: Baixo (~0.1-0.3), porque não foi otimizado para fitagem
- Best reward: Talvez 0.5-0.6 com sorte

#### Experimento 2: PPO Training - V2 como Base

**Objetivo:** Ver se PPO melhora o V2 para encontrar expressões específicas

**Método:**
```python
# Start from V2 model
base_model = load_model("augustocsc/Se124M_700K_infix_v2")
ppo_trainer = PPOTrainer(base_model, ppo_config)

# Train on Feynman dataset
for epoch in range(10):
    # Generate expressions with current policy
    expressions = generate_batch(prompts)

    # Compute rewards on target dataset
    rewards = [compute_reward(expr, X, y) for expr in expressions]

    # PPO update
    ppo_trainer.step(prompts, expressions, rewards)

    # Log progress
    print(f"Epoch {epoch}: Avg Reward = {mean(rewards):.3f}")

    # Early stopping if excellent fit found
    if max(rewards) >= 0.9:
        break
```

**Métricas a Coletar:**
- **Per-epoch average reward**: Deve aumentar monotonicamente
- **Per-epoch best reward**: Deve atingir >= 0.9
- **Convergence speed**: Quantos epochs até reward >= 0.9?
- **Invalid rate**: Deve permanecer baixa (~10%)
- **Final expression**: Deve ser interpretável e correta

**Expectativa:**
- Epoch 0 (baseline): Avg reward ~0.2
- Epoch 5: Avg reward ~0.6
- Epoch 10: Avg reward ~0.85, best >= 0.9

#### Experimento 3: Comparação V1 vs V2 para PPO

**Objetivo:** Determinar se V2 (90% valid) é melhor base que V1 (83.3%)

**Método:**
```python
# Same PPO training on both models
results_v1 = train_ppo("augustocsc/Se124M_700K_infix", dataset)
results_v2 = train_ppo("augustocsc/Se124M_700K_infix_v2", dataset)

# Compare
compare_metrics(results_v1, results_v2)
```

**Métricas:**
- **Convergence speed**: V2 deve convergir mais rápido (menos epochs)
- **Final reward**: V2 deve atingir reward maior
- **Stability**: V2 deve ter menos colapsos (reward -> 0)
- **Sample efficiency**: V2 precisa menos samples para aprender

**Hipótese:** V2 será significativamente melhor porque:
1. Maior taxa de valid (90% vs 83%)
2. Treinado com end markers (stop correto)
3. Melhor generalização

#### Experimento 4: Generalização Multi-Dataset

**Objetivo:** Ver se PPO generaliza para diferentes problemas

**Método:**
```python
# Train on Feynman Easy dataset 1
ppo_model = train_ppo(v2_base, feynman_easy_1)

# Test on other Feynman datasets
results = {
    "train": evaluate(ppo_model, feynman_easy_1),  # Train dataset
    "test_easy": evaluate(ppo_model, feynman_easy_2),  # Similar dataset
    "test_hard": evaluate(ppo_model, feynman_hard_1),  # Harder dataset
}
```

**Métricas:**
- **Train reward**: Deve ser alto (>= 0.9)
- **Test easy reward**: Deve ser moderado (~0.7)
- **Test hard reward**: Pode ser baixo (<0.5)
- **Transferência**: (test_reward / train_reward) indica generalização

#### Experimento 5: Análise Qualitativa

**Objetivo:** Entender **quais** expressões o PPO está gerando

**Método:**
```python
# Generate top-10 expressions per epoch
for epoch in ppo_training:
    expressions_with_rewards = sorted_by_reward(generated)

    # Analyze top expressions
    for expr, reward in top_10:
        print(f"Reward: {reward:.3f} | Expression: {expr}")

        # Check against known solutions
        if is_feynman_dataset:
            ground_truth = get_feynman_formula(dataset_id)
            similarity = compare_expressions(expr, ground_truth)
            print(f"Similarity to ground truth: {similarity}")
```

**Análises:**
- **Redescoberta:** PPO encontrou a fórmula correta de Feynman?
- **Simplicidade:** Expressões são simples ou overfit complexas?
- **Interpretabilidade:** Fazem sentido físico?
- **Diversidade:** PPO explora diferentes formas ou colapsa?

---

## Métricas de Sucesso

### Critérios Quantitativos

| Métrica | Baseline (V2 sem PPO) | Alvo (V2 com PPO) | Excelente |
|---------|----------------------|-------------------|-----------|
| **Avg Reward** | 0.1-0.3 | >= 0.7 | >= 0.85 |
| **Best Reward** | 0.3-0.5 | >= 0.9 | >= 0.95 |
| **Valid Rate** | ~90% | >= 85% | >= 90% |
| **Convergence** | N/A | <= 10 epochs | <= 5 epochs |
| **Sample Efficiency** | N/A | <= 500 samples | <= 200 samples |

### Critérios Qualitativos

1. **Redescoberta de Equações Conhecidas**
   - ✅ PPO encontra fórmulas de Feynman corretas
   - ✅ Expressões são simplificáveis para forma canônica

2. **Robustez**
   - ✅ PPO não colapsa (reward não cai para -1)
   - ✅ Mantém diversidade de expressões

3. **Interpretabilidade**
   - ✅ Expressões finais são simples (não overfit)
   - ✅ Constantes otimizadas fazem sentido físico

---

## Implementação Prática

### Código de Avaliação Proposto

```python
# scripts/evaluate_ppo.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from classes.expression import Expression
from classes.dataset import RegressionDataset
import json
from datetime import datetime

class PPOEvaluator:
    def __init__(self, base_model_path, dataset_path):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        self.dataset = RegressionDataset(dataset_path)
        self.X, self.y = self.dataset.get_numpy()

    def compute_reward(self, expression_str):
        """Compute R² score for expression"""
        try:
            expr = Expression(expression_str)
            if expr.is_valid_on_dataset(self.X):
                r2 = expr.fit_constants(self.X, self.y)
                return max(0.1, r2)
            return -1.0
        except:
            return -1.0

    def evaluate_baseline(self, n_samples=100):
        """Evaluate V2 without PPO"""
        print("="*60)
        print("BASELINE EVALUATION (No PPO)")
        print("="*60)

        rewards = []
        valid_exprs = []

        for i in range(n_samples):
            # Generate with V2's best config (nucleus sampling)
            expr = self.generate_expression(
                temperature=0.7,
                top_p=0.8,
            )
            reward = self.compute_reward(expr)
            rewards.append(reward)

            if reward > 0:
                valid_exprs.append((expr, reward))

        # Compute metrics
        results = {
            "avg_reward": np.mean(rewards),
            "median_reward": np.median(rewards),
            "best_reward": np.max(rewards),
            "valid_rate": sum(r > 0 for r in rewards) / n_samples,
            "top_expressions": sorted(valid_exprs, key=lambda x: x[1], reverse=True)[:10]
        }

        print(f"Avg Reward: {results['avg_reward']:.3f}")
        print(f"Best Reward: {results['best_reward']:.3f}")
        print(f"Valid Rate: {results['valid_rate']:.1%}")
        print(f"\nTop 3 Expressions:")
        for expr, reward in results['top_expressions'][:3]:
            print(f"  R²={reward:.3f}: {expr}")

        return results

    def train_ppo(self, n_epochs=10, stopping_reward=0.9):
        """Train with PPO"""
        print("\n" + "="*60)
        print("PPO TRAINING")
        print("="*60)

        # Setup PPO
        ppo_config = PPOConfig(
            model_name=None,
            learning_rate=1e-5,
            batch_size=32,
            mini_batch_size=8,
            ppo_epochs=4,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.base_model,
            ref_model=None,  # Will create reference model
            tokenizer=self.tokenizer,
        )

        epoch_results = []

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")

            # Generate batch
            queries = self.create_prompts(batch_size=32)
            responses = []
            rewards = []

            for query in queries:
                expr = self.generate_from_query(query)
                reward = self.compute_reward(expr)
                responses.append(expr)
                rewards.append(reward)

            # PPO step
            ppo_trainer.step(queries, responses, torch.tensor(rewards))

            # Log metrics
            avg_reward = np.mean(rewards)
            best_reward = np.max(rewards)
            valid_rate = sum(r > 0 for r in rewards) / len(rewards)

            epoch_result = {
                "epoch": epoch + 1,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "valid_rate": valid_rate,
            }
            epoch_results.append(epoch_result)

            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Best Reward: {best_reward:.3f}")
            print(f"  Valid Rate: {valid_rate:.1%}")

            # Early stopping
            if best_reward >= stopping_reward:
                print(f"\n✅ Target reward {stopping_reward} achieved!")
                break

        return {
            "epoch_results": epoch_results,
            "final_model": ppo_trainer.model,
            "converged": best_reward >= stopping_reward,
            "epochs_to_converge": epoch + 1 if best_reward >= stopping_reward else None,
        }

    def compare_v1_vs_v2(self):
        """Compare PPO on V1 vs V2"""
        print("\n" + "="*60)
        print("V1 vs V2 COMPARISON")
        print("="*60)

        # Train PPO on V1
        print("\nTraining PPO on V1 (83.3% base valid rate)...")
        v1_results = self.train_ppo_on_model("augustocsc/Se124M_700K_infix")

        # Train PPO on V2
        print("\nTraining PPO on V2 (90% base valid rate)...")
        v2_results = self.train_ppo_on_model("augustocsc/Se124M_700K_infix_v2")

        # Compare
        comparison = {
            "v1": {
                "final_reward": v1_results["epoch_results"][-1]["avg_reward"],
                "converged": v1_results["converged"],
                "epochs": v1_results["epochs_to_converge"],
            },
            "v2": {
                "final_reward": v2_results["epoch_results"][-1]["avg_reward"],
                "converged": v2_results["converged"],
                "epochs": v2_results["epochs_to_converge"],
            },
        }

        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"V1 Final Reward: {comparison['v1']['final_reward']:.3f}")
        print(f"V1 Converged: {comparison['v1']['converged']}")
        print(f"V1 Epochs: {comparison['v1']['epochs']}")
        print()
        print(f"V2 Final Reward: {comparison['v2']['final_reward']:.3f}")
        print(f"V2 Converged: {comparison['v2']['converged']}")
        print(f"V2 Epochs: {comparison['v2']['epochs']}")
        print()
        print(f"🏆 Winner: {'V2' if comparison['v2']['final_reward'] > comparison['v1']['final_reward'] else 'V1'}")

        return comparison

    def full_evaluation(self):
        """Run complete evaluation pipeline"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Baseline
        baseline_results = self.evaluate_baseline()

        # 2. PPO Training
        ppo_results = self.train_ppo()

        # 3. Save results
        results = {
            "timestamp": timestamp,
            "dataset": self.dataset.path,
            "baseline": baseline_results,
            "ppo": ppo_results,
        }

        output_file = f"ppo_evaluation_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        return results


# Usage
if __name__ == "__main__":
    evaluator = PPOEvaluator(
        base_model_path="augustocsc/Se124M_700K_infix_v2",
        dataset_path="./data/evaluate/srsd-feynman_easy/train/feynman-i.18.16.txt"
    )

    results = evaluator.full_evaluation()
```

---

## Datasets de Teste Recomendados

### Feynman Equations (Fáceis)
```
data/evaluate/srsd-feynman_easy/train/
├── feynman-i.12.1.txt     # μ*N (força de fricção)
├── feynman-i.18.16.txt    # m*c² (energia relativística)
├── feynman-i.6.2.txt      # exp(-θ²/2)/√(2π) (Gaussiana)
└── ...
```

**Características:**
- 2-3 variáveis
- Operadores simples (*, +, exp, sqrt)
- Soluções conhecidas (validação ground-truth)

### Feynman Equations (Difíceis)
```
data/evaluate/srsd-feynman_hard/train/
├── feynman-bonus.12.txt   # Equações mais complexas
└── ...
```

**Características:**
- 4+ variáveis
- Operadores compostos (sin, cos, log)
- Interações não-lineares

### Ordem de Teste Recomendada

1. **feynman-i.12.1** (mais fácil) - μ*N
2. **feynman-i.18.16** (médio) - m*c²
3. **feynman-i.6.2** (médio) - Gaussiana
4. **feynman-bonus.12** (difícil) - Complexo

---

## Cronograma de Avaliação

### Fase 1: Baseline (1 hora)
- [x] Configurar ambiente
- [ ] Rodar V2 sem PPO em 3 datasets
- [ ] Coletar métricas baseline
- [ ] Confirmar: Avg reward baixo (~0.2)

### Fase 2: PPO Single Dataset (2-3 horas)
- [ ] Treinar PPO em feynman-i.12.1
- [ ] Monitorar convergência
- [ ] Avaliar expressões finais
- [ ] Confirmar: Reward >= 0.9 alcançado

### Fase 3: Comparação V1 vs V2 (3-4 horas)
- [ ] Treinar PPO em V1
- [ ] Treinar PPO em V2
- [ ] Comparar métricas
- [ ] Confirmar: V2 converge mais rápido

### Fase 4: Generalização (2 horas)
- [ ] Testar modelo PPO em dataset diferente
- [ ] Avaliar transferência
- [ ] Documentar resultados

**Total: ~8-10 horas de computação (AWS g5.xlarge)**

---

## Resultados Esperados

### Cenário Ideal (PPO Funcionando Bem)

```
BASELINE (V2 sem PPO):
  Avg Reward: 0.18
  Best Reward: 0.42
  Valid Rate: 90%

PPO TRAINING (V2 com PPO):
  Epoch 1: Avg=0.21, Best=0.48
  Epoch 3: Avg=0.45, Best=0.72
  Epoch 6: Avg=0.71, Best=0.93 ✅ TARGET!

  ✅ Converged in 6 epochs
  ✅ Found expression: 1.02*x_1*x_2 (R²=0.93)
  ✅ Ground truth: x_1*x_2 (similarity: 98%)

V1 vs V2:
  V1: Converged in 12 epochs, Final R²=0.89
  V2: Converged in 6 epochs, Final R²=0.93
  🏆 V2 is 2x faster and more accurate
```

### Cenário Problemático (PPO Não Funcionando)

```
PPO TRAINING:
  Epoch 1: Avg=0.15, Best=0.35
  Epoch 5: Avg=0.18, Best=0.41
  Epoch 10: Avg=0.16, Best=0.38

  ❌ No improvement over baseline
  ❌ Reward not increasing
  ❌ No convergence

Possíveis Problemas:
  - Learning rate muito alto/baixo
  - Reward function mal calibrada
  - Model collapse (mode collapse)
  - Insufficient exploration
```

---

## Métricas de Debug

Se PPO não funcionar, diagnosticar com:

### 1. KL Divergence
```python
# Monitor KL from reference model
if kl_divergence > 0.1:
    print("⚠️ Model diverging too much from reference")
```

### 2. Policy Entropy
```python
# Check exploration
if entropy < 0.5:
    print("⚠️ Policy too deterministic, not exploring")
```

### 3. Value Function Accuracy
```python
# Check if value head predicting rewards well
value_error = abs(predicted_value - actual_reward)
if value_error > 0.5:
    print("⚠️ Value function not learning")
```

### 4. Expression Diversity
```python
# Check if generating different expressions
unique_rate = len(set(expressions)) / len(expressions)
if unique_rate < 0.5:
    print("⚠️ Mode collapse, generating same expressions")
```

---

## Próximos Passos Recomendados

### Opção A: Avaliação Rápida (Hoje - 2 horas)
1. ✅ Entender código PPO (feito)
2. Rodar baseline V2 em 1 dataset fácil
3. Rodar PPO por 5 epochs
4. Ver se reward aumenta

**Objetivo:** Validação rápida se PPO básico funciona

### Opção B: Avaliação Completa (Esta Semana - 10 horas)
1. Implementar script `evaluate_ppo.py` completo
2. Rodar em 3 datasets
3. Comparar V1 vs V2
4. Gerar relatório completo

**Objetivo:** Publicação/documentação completa

### Opção C: Investigação Profunda (Projeto - 2 semanas)
1. Avaliar completamente
2. Tunear hiperparâmetros PPO
3. Experimentar reward functions alternativas
4. Benchmarkar contra state-of-the-art symbolic regression

**Objetivo:** Contribuição científica

---

## Conclusão

**Como avaliar o PPO:**

1. **Baseline sem PPO**: Ver que V2 gera expressões válidas mas com R² baixo (~0.2)
2. **Training com PPO**: Ver que reward aumenta epoch-a-epoch até >= 0.9
3. **Comparação V1 vs V2**: Ver que V2 converge mais rápido (melhor base)
4. **Qualitativo**: Ver que PPO encontra fórmulas corretas (Feynman)

**Métrica Principal:** **R² Score**
- Baseline: ~0.2
- Target: >= 0.9
- Success: Atingir target em <= 10 epochs

**Código já está implementado em:** `scripts/symbolic_rl/trainer.py`

**Você quer que eu:**
- [ ] Rode avaliação rápida agora (2h)?
- [ ] Prepare script completo de avaliação?
- [ ] Outra abordagem?

