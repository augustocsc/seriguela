# Plano de Experimentos: Formatos de Treino

**Data:** 2026-02-01
**Objetivo:** Testar duas abordagens para resolver o problema de stopping

---

## Contexto

### Problema Identificado
- Dados de treino não têm marcador de fim (0% com qualquer marker)
- Modelo não aprende quando parar
- Gera garbage tokens do vocabulário GPT-2

### Experimentos Propostos
1. **EXP-A:** Formato estruturado (JSON-like)
2. **EXP-B:** Token EOS do GPT-2 (`<|endoftext|>`)

---

## EXP-A: Formato Estruturado

### Formato dos Dados
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "expr": "C*sin(x_1) + x_2"}
```

### Vantagens
- Estrutura clara e parseável
- Fácil validação (JSON válido = formato correto)
- Modelo aprende estrutura rígida

### Desvantagens
- Mais tokens por exemplo
- Pode ser mais difícil de aprender

### Preparação de Dados
```python
# Transformar de:
"vars: x_1, x_2\noper: *, +, sin\ncons: C\nexpr: C*sin(x_1) + x_2"

# Para:
'{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "C*sin(x_1) + x_2"}'
```

### Inferência
```python
prompt = '{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "'
# Modelo completa com: C*sin(x_1) + x_2"}
# Extrair: tudo entre 'expr": "' e '"}'
```

### Critério de Sucesso
- JSON parseável em >90% dos casos
- Expressão extraída válida em >80% dos casos

---

## EXP-B: Token EOS do GPT-2

### Formato dos Dados
```
vars: x_1, x_2
oper: *, +, sin
cons: C
expr: C*sin(x_1) + x_2<|endoftext|>
```

### Vantagens
- Token já existe no modelo (ID 50256)
- GPT-2 já entende como "fim de sequência"
- Não precisa resize de embeddings
- Formato similar ao atual

### Desvantagens
- Pode conflitar com outros usos do EOS
- Menos explícito que marker dedicado

### Preparação de Dados
```python
# Adicionar <|endoftext|> no final de cada expressão
text = original_text + "<|endoftext|>"
```

### Inferência
```python
# Usar eos_token_id como stopping criteria
output = model.generate(
    **inputs,
    eos_token_id=tokenizer.eos_token_id,  # 50256
    max_new_tokens=128
)
```

### Critério de Sucesso
- Modelo gera `<|endoftext|>` em >90% dos casos
- Expressão antes do EOS válida em >80% dos casos

---

## Plano de Execução

### Fase 1: Preparação de Dados (Local)

#### 1.1 Criar script de preparação
```
scripts/data/prepare_experiment_data.py
```
- Entrada: dataset augustocsc/sintetico_natural (700K)
- Saída A: data/exp_a_json/train.csv, validation.csv
- Saída B: data/exp_b_eos/train.csv, validation.csv

#### 1.2 Validar dados preparados
- Verificar formato correto em 100% dos exemplos
- Amostrar e inspecionar manualmente

### Fase 2: Treino (AWS)

#### 2.1 Treinar EXP-A (JSON)
```bash
python scripts/train.py \
  --use_local_csvs \
  --train_file ./data/exp_a_json/train.csv \
  --output_dir ./output/exp_a_json \
  --num_train_epochs 3
```

#### 2.2 Treinar EXP-B (EOS)
```bash
python scripts/train.py \
  --use_local_csvs \
  --train_file ./data/exp_b_eos/train.csv \
  --output_dir ./output/exp_b_eos \
  --num_train_epochs 3
```

### Fase 3: Avaliação

#### 3.1 Métricas
- **Valid Rate:** % expressões parseáveis
- **Stopping Rate:** % que param corretamente (JSON fechado ou EOS)
- **Symbol Accuracy:** % que usam apenas símbolos do prompt
- **Garbage Rate:** % com tokens não-matemáticos

#### 3.2 Comparação
| Métrica | EXP-A (JSON) | EXP-B (EOS) |
|---------|--------------|-------------|
| Valid Rate | ? | ? |
| Stopping Rate | ? | ? |
| Symbol Accuracy | ? | ? |
| Garbage Rate | ? | ? |

### Fase 4: Decisão

- Se EXP-A melhor → usar formato JSON
- Se EXP-B melhor → usar EOS token
- Se ambos ruins → investigar outras opções

---

## Estimativas

| Fase | Tempo | Custo AWS |
|------|-------|-----------|
| Preparação dados | 30 min | $0 |
| Treino EXP-A | 2-3h | ~$3 |
| Treino EXP-B | 2-3h | ~$3 |
| Avaliação | 30 min | ~$0.50 |
| **Total** | **6-7h** | **~$6.50** |

---

## Arquivos a Criar

```
scripts/data/prepare_experiment_data.py  # Preparação
data/exp_a_json/train.csv                # Dados JSON
data/exp_a_json/validation.csv
data/exp_b_eos/train.csv                 # Dados EOS
data/exp_b_eos/validation.csv
scripts/evaluate_experiments.py          # Avaliação
```

---

## Critério de Sucesso Final

**Experimento bem-sucedido se:**
- Valid Rate > 80%
- Stopping Rate > 90%
- Garbage Rate < 5%

**Próximo passo após sucesso:**
- Usar formato vencedor para treinar modelo final
- Prosseguir para Block 3 (PPO)
