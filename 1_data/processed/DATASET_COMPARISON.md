# Comparação de Datasets: Sintetico Natural

## 📊 Visão Geral dos Datasets Disponíveis

| Dataset | Exemplos | Notação | Split | Uso Recomendado |
|---------|----------|---------|-------|-----------------|
| `sintetico_natural` (padrão) | 12,221 | Infix | test apenas | ❌ Muito pequeno para treino |
| `sintetico_natural` (data_dir='700K') | 947,876 | Infix | train/val/test | ⚠️ Nome confuso, splits não usados |
| `sintetico_natural_prefix` | 12,221 | Prefix | test apenas | ❌ Muito pequeno para treino |
| **`sintetico_natural_prefix_682k`** | **758,255** | **Prefix** | **train/val (90/10)** | ✅ **Recomendado para treino** |

## 🎯 Qual Dataset Usar?

### Para Treinar Modelos

**Recomendado**: `augustocsc/sintetico_natural_prefix_682k`

```python
from datasets import load_dataset

# Dataset correto com split usado no treinamento
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')

train_data = dataset['train']      # 682,429 exemplos
val_data = dataset['validation']   # 75,826 exemplos
```

**Por quê?**
- ✅ Split correto (90/10, seed=42) usado no treinamento dos modelos publicados
- ✅ Tamanho adequado (~682K exemplos)
- ✅ Reprodutibilidade perfeita
- ✅ Disponível em infix e prefix

### Para Comparar Infix vs Prefix

Use **ambos** com o mesmo split:

```python
# Infix
dataset_infix = load_dataset('augustocsc/sintetico_natural', data_dir='700K', split='train')
split_infix = dataset_infix.train_test_split(test_size=0.1, seed=42)

# Prefix (já com split correto)
dataset_prefix = load_dataset('augustocsc/sintetico_natural_prefix_682k')

# Mesmos 682K exemplos!
assert len(split_infix['train']) == len(dataset_prefix['train'])  # 682,429
```

### Para Testes Rápidos

Use o dataset pequeno:

```python
# Apenas 12K exemplos - bom para testes
dataset = load_dataset('augustocsc/sintetico_natural')  # split='test'
```

## 🔍 Detalhamento: Dataset Original (`sintetico_natural`)

### Configuração Confusa

O dataset `augustocsc/sintetico_natural` tem **dois modos de carregamento**:

#### Modo 1: Padrão (sem `data_dir`)
```python
dataset = load_dataset('augustocsc/sintetico_natural')
# Resultado: {'test': 12,221 exemplos}
```

**Problema**: Apenas split `test`, muito pequeno para treinar.

#### Modo 2: Com `data_dir='700K'`
```python
dataset = load_dataset('augustocsc/sintetico_natural', data_dir='700K')
# Resultado:
# - train: 758,255 exemplos
# - validation: 95,616 exemplos
# - test: 94,005 exemplos
# TOTAL: 947,876 exemplos
```

**Problemas**:
1. ❌ Nome "700K" é enganoso (são quase 1 milhão)
2. ❌ Scripts de treinamento **ignoram** os splits validation e test
3. ❌ Cada script faz seu próprio split 90/10 do `train`
4. ❌ Inconsistência entre experimentos se não usar mesmo seed

## 🔄 Como os Modelos Foram Treinados

Todos os modelos publicados (`Se124M_700K_infix_v3_json`, etc.) usaram:

```python
# Código de 2_training/supervised/train_with_json.py
dataset = load_dataset('augustocsc/sintetico_natural', data_dir='700K', split='train')
# 758,255 exemplos

# Script faz split interno 90/10
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']  # 682,429 exemplos ← USADO
eval_dataset = split_dataset['test']    # 75,826 exemplos  ← USADO
```

**Os splits `validation` e `test` do HuggingFace (95K + 94K) NÃO foram usados!**

## ✅ Solução: Dataset `sintetico_natural_prefix_682k`

Este dataset resolve todos os problemas:

### Características
- ✅ **682,429 exemplos de treino** (exatos usados no treinamento)
- ✅ **75,826 exemplos de validação** (exatos usados no treinamento)
- ✅ **Split fixo (seed=42)** - reprodutibilidade perfeita
- ✅ **Notação prefix** para comparar com infix
- ✅ **Nome correto** (682K, não "700K")
- ✅ **Sem splits não utilizados** (apenas train/val necessários)

### Comparação Direta Infix vs Prefix

| Aspecto | Infix (Original) | Prefix (682K) |
|---------|------------------|---------------|
| **Carga** | 2 passos (load + split) | 1 passo (load) |
| **Splits** | Manual (train_test_split) | Automático (train/val) |
| **Seed** | Precisa especificar | Fixo (42) |
| **Reprodutibilidade** | Difícil | Fácil |
| **Exemplos treino** | 682,429 | 682,429 ✅ |
| **Exemplos val** | 75,826 | 75,826 ✅ |

## 📝 Exemplo Completo de Uso

### Treinar Modelo Infix

```python
from datasets import load_dataset

# Carregar e fazer split manual (jeito antigo)
dataset = load_dataset('augustocsc/sintetico_natural', data_dir='700K', split='train')
split = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split['train'].map(lambda x: {'text': x['i_prompt_n']})
eval_dataset = split['test'].map(lambda x: {'text': x['i_prompt_n']})

# Treinar...
```

### Treinar Modelo Prefix (Mais Simples)

```python
from datasets import load_dataset

# Carregar com split já pronto
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')

train_dataset = dataset['train'].map(lambda x: {'text': x['p_prompt_n_converted']})
eval_dataset = dataset['validation'].map(lambda x: {'text': x['p_prompt_n_converted']})

# Treinar...
```

## 🚨 Armadilhas Comuns

### ❌ ERRADO: Usar `p_prompt_n` para comparação

```python
# ISTO ESTÁ ERRADO!
dataset = load_dataset('augustocsc/sintetico_natural', data_dir='700K')
infix_expr = dataset['train'][0]['i_prompt_n']   # "expr: x_1 + x_2"
prefix_expr = dataset['train'][0]['p_prompt_n']  # "expr: * x_3 x_4"  ← DIFERENTE!
```

**Problema**: `i_prompt_n` e `p_prompt_n` são **expressões completamente diferentes**, não conversões!

### ✅ CORRETO: Usar `p_prompt_n_converted` para comparação

```python
# ISTO ESTÁ CORRETO!
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')
infix_expr = dataset['train'][0]['i_prompt_n']             # "expr: x_1 + x_2"
prefix_expr = dataset['train'][0]['p_prompt_n_converted']  # "expr: + x_1 x_2"  ← MESMA EXPRESSÃO!
```

## 📈 Histórico de Problemas e Soluções

### Problema 1: Dataset "700K" tem 950K exemplos
**Descoberta**: 2026-02-09
**Causa**: Nome histórico não reflete conteúdo atual
**Solução**: Novo dataset com nome correto (682K)

### Problema 2: Splits do HF não são usados
**Descoberta**: 2026-02-09
**Causa**: Scripts fazem split interno do `train`
**Solução**: Dataset com split pré-definido igual ao usado no treinamento

### Problema 3: `p_prompt_n` ≠ conversão de `i_prompt_n`
**Descoberta**: 2026-02-09
**Causa**: São expressões diferentes no dataset original
**Solução**: Nova coluna `p_prompt_n_converted` com conversão exata

## 🔧 Migração de Código

### De: Código Antigo (Infix)
```python
dataset = load_dataset('augustocsc/sintetico_natural', data_dir='700K', split='train')
split = dataset.train_test_split(test_size=0.1, seed=42)
train = split['train']
val = split['test']
```

### Para: Código Novo (Infix - Compatível)
```python
# Mesma coisa, mas mais explícito
dataset = load_dataset('augustocsc/sintetico_natural', data_dir='700K', split='train')
split = dataset.train_test_split(test_size=0.1, seed=42)
train = split['train']  # 682,429 exemplos
val = split['test']      # 75,826 exemplos
```

### Para: Código Novo (Prefix - Simplificado)
```python
# Muito mais simples!
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')
train = dataset['train']       # 682,429 exemplos
val = dataset['validation']    # 75,826 exemplos
```

## 📚 Referências

- **Dataset Original**: https://huggingface.co/datasets/augustocsc/sintetico_natural
- **Dataset Prefix (12K)**: https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix
- **Dataset Prefix (682K)**: https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
- **Código de Conversão**: `scripts/data/convert_infix_to_prefix.py`

---

**Última Atualização**: 2026-02-09
**Versão**: 1.0
