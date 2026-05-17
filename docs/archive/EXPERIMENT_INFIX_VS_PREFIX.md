# Plano de Experimento: Comparativo Infix vs Prefix

**Data**: 2026-02-12 (Atualizado: 2026-02-19)
**Objetivo**: Comparar modelos treinados com notação infix vs prefix para symbolic regression

---

## ⚠️ PRIORIDADE ZERO: ARRUMAR OS MODELOS

Antes de qualquer experimento comparativo, precisamos ter os 6 modelos funcionando:

### Status Atual dos Modelos

| Modelo | Notação | Parâmetros | Status | Ação Necessária |
|--------|---------|------------|--------|-----------------|
| **Base** | Infix | 124M | ⚠️ Erro de carregamento | Corrigir scripts RL |
| **Medium** | Infix | 355M | ❌ NÃO EXISTE | Treinar do zero |
| **Large** | Infix | 774M | ❌ NÃO EXISTE | Treinar do zero |
| **Base** | Prefix | 124M | ✅ OK (local) | Upload HuggingFace |
| **Medium** | Prefix | 355M | ⚠️ Mode collapse | Retreinar com r=16 |
| **Large** | Prefix | 774M | ✅ OK (local) | Upload HuggingFace |

### Plano de Ação Sequencial

```
FASE 0: Corrigir carregamento infix (1h)
    ↓
FASE 1: Treinar modelos infix Medium/Large (4-6h AWS)
    ↓
FASE 2: Retreinar prefix Medium com r=16 (2-3h AWS)
    ↓
FASE 3: Upload todos modelos para HuggingFace
    ↓
FASE 4: Experimento comparativo (6 modelos x 12 benchmarks)
```

---

## FASE 0: Corrigir Carregamento do Modelo Infix (LOCAL)

### Problema
```
RuntimeError: size mismatch for transformer.wte.weight:
  checkpoint: torch.Size([50259, 768])  # Modelo treinado com 2 tokens extras
  current:    torch.Size([50257, 768])  # GPT-2 base padrao
```

### Solução
Modificar `ppo_symbolic_enhanced.py` e `grpo_symbolic_enhanced.py`:

```python
# ANTES (problematico)
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# DEPOIS (corrigido)
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Carrega tokenizer do modelo
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Se tokenizer tem mais tokens, redimensionar
if len(tokenizer) > base_model.config.vocab_size:
    base_model.resize_token_embeddings(len(tokenizer))

base_model.enable_input_require_grads()
model = PeftModel.from_pretrained(base_model, model_path)
```

### Arquivos a Modificar
1. `scripts/ppo_symbolic_enhanced.py`
2. `scripts/grpo_symbolic_enhanced.py`

---

## FASE 1: Treinar Modelos Infix Medium e Large (AWS)

### IMPORTANTE: Dataset Unificado

**TODOS os modelos usam o MESMO dataset** para garantir comparabilidade:

```
Dataset: augustocsc/sintetico_natural_prefix_682k
Train: 682,429 exemplos
Val: 75,826 exemplos
```

| Notação | Coluna | Mesma expressão? |
|---------|--------|------------------|
| INFIX | `i_prompt_n` | ✅ Sim |
| PREFIX | `p_prompt_n_converted` | ✅ Sim |

**Documentação completa**: Ver `TRAINING_CONFIG_REGISTRY.md`

### 1.1 Configuração para Infix Medium (355M)

```bash
# Lançar treinamento na AWS
bash scripts/aws/launch_medium_infix_training.sh \
  --hf-token $HF_TOKEN \
  --wandb-key $WANDB_KEY
```

**Hiperparâmetros**:
- Base model: gpt2-medium
- Dataset: augustocsc/sintetico_natural_prefix_682k
- Coluna: i_prompt_n (formato infix)
- LoRA: r=16, alpha=64 (maior para modelo maior)
- Learning rate: 3e-5 (menor para estabilidade)
- Epochs: 3 (com early stopping)
- Instance: g5.xlarge

### 1.2 Configuração para Infix Large (774M)

```bash
bash scripts/aws/launch_large_infix_training.sh \
  --hf-token $HF_TOKEN \
  --wandb-key $WANDB_KEY
```

**Hiperparâmetros**:
- Base model: gpt2-large
- Dataset: augustocsc/sintetico_natural_prefix_682k
- Coluna: i_prompt_n (formato infix)
- LoRA: r=16, alpha=64
- Learning rate: 2e-5
- Instance: g5.2xlarge (48GB VRAM)

---

## FASE 2: Retreinar Prefix Medium com r=16 (AWS)

### Problema Atual
- Valid rate: 0.5% (3/640 expressões)
- Expressões corrompidas: `xt_4`, `xt_1` ao invés de `x_1`, `x_4`
- Melhor expressão "válida": apenas `C` (constante)

### Solução
Retreinar com LoRA maior:

```bash
bash scripts/aws/launch_medium_prefix_v2_training.sh \
  --hf-token $HF_TOKEN \
  --wandb-key $WANDB_KEY
```

**Configuração v2**:
- LoRA: r=16 (dobrar de 8)
- alpha: 64 (manter proporção 4x)
- Learning rate: 3e-5 (reduzir de 5e-5)
- Warmup: 1000 steps (dobrar)

---

## FASE 3: Upload para HuggingFace

### Modelos Locais para Upload

```bash
# Base Prefix
cd output/gpt2_base_prefix_682k
huggingface-cli upload augustocsc/gpt2_base_prefix_682k . .

# Large Prefix
cd output/gpt2_large_prefix_682k
huggingface-cli upload augustocsc/gpt2_large_prefix_682k . .

# Medium Prefix v2 (após retreino)
cd output/gpt2_medium_prefix_v2_682k
huggingface-cli upload augustocsc/gpt2_medium_prefix_v2_682k . .

# Medium Infix (após treino)
huggingface-cli upload augustocsc/gpt2_medium_infix_700k . .

# Large Infix (após treino)
huggingface-cli upload augustocsc/gpt2_large_infix_700k . .
```

---

## 1. Resumo da Investigação (Contexto)

### 1.1 Problemas Identificados

#### Problema 1: Modelo Infix Falhou ao Carregar
**Causa raiz**: Incompatibilidade de tamanho do vocabulário
**Status**: 🔧 Solução definida (FASE 0)

#### Problema 2: Mode Collapse no Medium Prefix (355M)
**Evidência**:
- Valid rate: 0.5% (3/640 expressões)
- Expressões corrompidas: `xt_4`, `xt_1` ao invés de `x_1`, `x_4`
**Status**: 🔧 Retreinar com r=16 (FASE 2)

#### Problema 3: Modelos Infix Medium/Large Não Existem
**Status**: 🔧 Treinar do zero (FASE 1)

#### Problema 4: Evolução por Época

| Modelo | Valid Rate Total | Best R2 Inicial | Best R2 Final | Melhoria |
|--------|------------------|-----------------|---------------|----------|
| Base (124M) | 9.8% | 0.2173 | 0.8638 | +0.6464 |
| Medium (355M) | 0.5% | -1.0000 | -0.5879 | +0.4121 |
| Large (774M) | 1.9% | 0.2173 | 0.6275 | +0.4102 |

**Observação**: O modelo **melhora ao longo das épocas**, mas com alta variância. Base tem melhor convergência.

---

## 2. Plano de Experimento Comparativo (APÓS FASES 0-3)

### 2.1 Objetivo
Responder: **Modelos prefix são melhores que infix para symbolic regression com RL?**

### 2.2 Configuração Final (6 Modelos)

| Modelo | Tamanho | Notação | Fonte | Status |
|--------|---------|---------|-------|--------|
| base_infix | 124M | Infix | `augustocsc/Se124M_700K_infix_v3_json` | FASE 0 |
| medium_infix | 355M | Infix | `augustocsc/gpt2_medium_infix_700k` | FASE 1 |
| large_infix | 774M | Infix | `augustocsc/gpt2_large_infix_700k` | FASE 1 |
| base_prefix | 124M | Prefix | `augustocsc/gpt2_base_prefix_682k` | FASE 3 |
| medium_prefix_v2 | 355M | Prefix | `augustocsc/gpt2_medium_prefix_v2_682k` | FASE 2+3 |
| large_prefix | 774M | Prefix | `augustocsc/gpt2_large_prefix_682k` | FASE 3 |

### 2.3 Benchmarks
- Nguyen 1-12 (12 problemas)
- Foco especial: Nguyen-5 `sin(x^2)*cos(x) - 1` (mais difícil)

### 2.4 Algoritmos RL
- PPO (mais consistente)
- GRPO (pico mais alto)

### 2.5 Métricas
1. **Valid Rate (%)**: Expressões sintaticamente corretas
2. **Best R2**: Melhor fit alcançado
3. **Epochs to Best**: Quantas épocas até o melhor resultado
4. **Melhoria por Época**: Delta R2 médio por época

### 2.3 Abordagens de Prompt

#### Abordagem A: Prompt Geral (Todos os Operadores)
Inclui TODOS os operadores e variaveis possiveis do dataset:

**Prefix**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, sin, cos, tan, exp, log, sqrt, abs, **
cons: C
expr:
```

**Infix (JSON)**:
```json
{"vars": ["x_1", "x_2", "x_3", "x_4", "x_5"], "ops": ["*", "+", "-", "/", "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "**"], "cons": "C", "expr": "
```

**Vantagens**:
- Modelo pode explorar todo o espaco de expressoes
- Mais flexivel para benchmarks diferentes

**Desvantagens**:
- Maior espaco de busca
- Pode gerar expressoes desnecessariamente complexas

#### Abordagem B: Prompt Propositivo (Operadores Necessarios)
Inclui APENAS os operadores necessarios para resolver o benchmark especifico:

**Exemplo para Nguyen-1** (`x^3 + x^2 + x`):
```
vars: x_1
oper: *, +, **
cons: C
expr:
```

**Exemplo para Nguyen-5** (`sin(x^2)*cos(x) - 1`):
```
vars: x_1
oper: *, -, sin, cos, **
cons: C
expr:
```

**Vantagens**:
- Espaco de busca reduzido
- Guia o modelo para solucao correta
- Maior probabilidade de encontrar expressao valida

**Desvantagens**:
- Requer conhecimento previo da solucao
- Menos realista para casos de uso reais

#### Abordagem C: Hibrido Adaptativo
O modelo comeca com prompt geral e, se nao encontrar boa solucao apos N epocas,
refina o prompt removendo operadores nao utilizados.

---

## 3. Correcoes Necessarias

### 3.1 Corrigir Carregamento do Modelo Infix

**Arquivo**: `scripts/ppo_symbolic_enhanced.py` e `scripts/grpo_symbolic_enhanced.py`

**Mudanca**: Carregar tokenizer do modelo HuggingFace e redimensionar embeddings

```python
# ANTES (problematico)
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# DEPOIS (corrigido)
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Carrega tokenizer do modelo
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Se tokenizer tem mais tokens, redimensionar
if len(tokenizer) > base_model.config.vocab_size:
    base_model.resize_token_embeddings(len(tokenizer))

base_model.enable_input_require_grads()
model = PeftModel.from_pretrained(base_model, model_path)
```

### 3.2 Retreinar Medium com LoRA Maior

**Configuracao proposta para Medium v2**:
```json
{
  "base_model": "gpt2-medium",
  "lora_r": 16,          // Dobrar de 8 para 16
  "lora_alpha": 64,      // Manter proporcao 4x
  "learning_rate": 3e-5, // Reduzir de 5e-5
  "warmup_steps": 1000   // Dobrar warmup
}
```

### 3.3 Ajustar Hiperparametros de RL

**Para modelos maiores (Medium, Large)**:
```python
# PPO
learning_rate = 1e-5    # Reduzir de 3e-5
clip_epsilon = 0.1      # Mais conservador (de 0.2)
epochs_per_update = 2   # Menos updates

# GRPO
group_size = 16         # Maior grupo (de 8)
temperature = 0.8       # Mais exploracao (de 0.7)
```

---

## 4. Pipeline de Execucao

### Fase 1: Correcoes (Local - 1-2h)
1. [ ] Corrigir carregamento do modelo infix em `ppo_symbolic_enhanced.py`
2. [ ] Corrigir carregamento do modelo infix em `grpo_symbolic_enhanced.py`
3. [ ] Testar localmente com 1 experimento

### Fase 2: Retreinar Medium (AWS - 3-4h)
1. [ ] Criar script `launch_medium_prefix_v2_training.sh` com r=16
2. [ ] Treinar medium_prefix_v2 na AWS
3. [ ] Download e validar modelo

### Fase 3: Experimento Comparativo (AWS - 6-8h)
1. [ ] Rodar suite completa: 4 modelos x 12 benchmarks x 2 algoritmos x 2 prompts
2. [ ] Total: 192 experimentos
3. [ ] Coletar dados completos (evolucao por epoca)

### Fase 4: Analise (Local - 2-3h)
1. [ ] Comparar infix vs prefix
2. [ ] Comparar prompt geral vs propositivo
3. [ ] Analisar evolucao por epoca
4. [ ] Gerar relatorio final

---

## 5. Estimativa de Custos

| Fase | Instancia | Duracao | Custo |
|------|-----------|---------|-------|
| Retreino Medium v2 | g5.xlarge | 3-4h | ~$4 |
| Experimento (192 exp) | g5.2xlarge | 8-10h | ~$12 |
| **TOTAL** | | | **~$16** |

---

## 6. Metricas de Sucesso

### Hipoteses a Testar

**H1**: Prefix > Infix em valid rate (devido a estrutura mais simples)
**H2**: Prompt propositivo > Prompt geral em R2 (espaco de busca menor)
**H3**: Medium v2 (r=16) > Medium v1 (r=8) em valid rate (mais capacidade)
**H4**: Modelos melhoram monotonicamente com epocas (evidencia de aprendizado)

### Criterios de Sucesso

| Metrica | Minimo Aceitavel | Objetivo |
|---------|------------------|----------|
| Valid Rate (Base) | > 10% | > 20% |
| Best R2 (qualquer modelo) | > 0.90 | > 0.99 |
| Melhoria por epoca | > 0 | > 0.02 |
| Benchmarks resolvidos (R2 > 0.95) | 6/12 | 10/12 |

---

## 7. Workflow do Sistema Final

```
                    +------------------+
                    |  Base de Dados   |
                    |  (x, y) pairs    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Criar Prompt    |
                    | (geral ou prop)  |
                    +--------+---------+
                             |
                             v
            +----------------+----------------+
            |                                 |
            v                                 v
    +-------+-------+                 +-------+-------+
    | Modelo Infix  |                 | Modelo Prefix |
    | (JSON format) |                 | (Polish nota) |
    +-------+-------+                 +-------+-------+
            |                                 |
            +----------------+----------------+
                             |
                             v
                    +------------------+
                    | Gerar N hipotes  |
                    | (expressoes)     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Avaliar R2      |
                    |  em (x, y)       |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Calcular Reward  |
                    | (R2 + bonus)     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Atualizar LLM   |
                    |  (PPO/GRPO)      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Nova Epoca      |
                    |  (repetir)       |
                    +--------+---------+
                             |
                    (apos N epocas)
                             |
                             v
                    +------------------+
                    | Melhor Expressao |
                    | (maior R2)       |
                    +------------------+
```

---

## 8. Proximos Passos Imediatos

1. **Implementar correcao do modelo infix** (30 min)
2. **Testar localmente** (15 min)
3. **Criar script de retreino Medium v2** (30 min)
4. **Lancar experimento comparativo** (apos aprovacao)

---

**Aguardando aprovacao para prosseguir com a implementacao.**
