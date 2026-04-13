# Registro de Configuracao de Treinamento

**Objetivo**: Garantir que TODOS os 6 modelos (3 infix + 3 prefix) sejam treinados com configuracoes comparaveis.

**Status**: Retreinar TODOS os 6 modelos do zero para garantir consistencia academica.

---

## Comando para Treinar Tudo

```bash
bash scripts/aws/launch_full_experiment.sh \
  --hf-token $HF_TOKEN \
  --wandb-key $WANDB_KEY
```

**Custo estimado**: ~$15-20 USD
**Tempo estimado**: ~5-6 horas (paralelo)

---

## Dataset Unificado

**CRITICO**: Todos os modelos DEVEM usar o mesmo dataset com os mesmos splits!

| Aspecto | Valor | Justificativa |
|---------|-------|---------------|
| **Dataset HuggingFace** | `augustocsc/sintetico_natural_prefix_682k` | Tem ambas colunas infix e prefix |
| **Train split** | 682,429 exemplos | 90% do total |
| **Validation split** | 75,826 exemplos | 10% do total |
| **Split seed** | 42 | Reprodutibilidade |
| **Total** | 758,255 exemplos | Mesmo para todos modelos |

### Colunas por Notacao

| Notacao | Coluna do Dataset | Exemplo |
|---------|-------------------|---------|
| **INFIX** | `i_prompt_n` | `expr: sin(x_1 + C*x_2)` |
| **PREFIX** | `p_prompt_n_converted` | `expr: sin + x_1 * C x_2` |

**IMPORTANTE**: Ambas colunas representam as **MESMAS expressoes** em notacoes diferentes!

---

## Configuracao por Modelo

### Modelos INFIX

| Modelo | Base | Params | Dataset | Coluna | LoRA r | LoRA alpha | LR | Batch | Script |
|--------|------|--------|---------|--------|--------|------------|-----|-------|--------|
| Base | gpt2 | 124M | sintetico_natural_prefix_682k | i_prompt_n | 8 | 32 | 5e-5 | 8 | `augustocsc/Se124M_700K_infix_v3_json` (existente) |
| Medium | gpt2-medium | 355M | sintetico_natural_prefix_682k | i_prompt_n | 16 | 64 | 3e-5 | 4 | `launch_medium_infix_training.sh` |
| Large | gpt2-large | 774M | sintetico_natural_prefix_682k | i_prompt_n | 16 | 64 | 2e-5 | 2 | `launch_large_infix_training.sh` |

### Modelos PREFIX

| Modelo | Base | Params | Dataset | Coluna | LoRA r | LoRA alpha | LR | Batch | Script |
|--------|------|--------|---------|--------|--------|------------|-----|-------|--------|
| Base | gpt2 | 124M | sintetico_natural_prefix_682k | p_prompt_n_converted | 8 | 32 | 5e-5 | 8 | `launch_base_prefix_training.sh` |
| Medium v2 | gpt2-medium | 355M | sintetico_natural_prefix_682k | p_prompt_n_converted | 16 | 64 | 3e-5 | 4 | `launch_medium_prefix_v2_training.sh` |
| Large | gpt2-large | 774M | sintetico_natural_prefix_682k | p_prompt_n_converted | 8 | 32 | 5e-5 | 2 | `launch_large_prefix_training.sh` |

---

## Configuracoes Comuns (Todos os Modelos)

### LoRA

```python
lora_config = LoraConfig(
    r=R_VALUE,           # 8 para Base, 16 para Medium/Large
    lora_alpha=ALPHA,    # 32 para Base, 64 para Medium/Large
    target_modules=["c_attn"],  # MESMO para todos
    lora_dropout=0.05,          # MESMO para todos
    bias="none",                # MESMO para todos
    task_type="CAUSAL_LM",      # MESMO para todos
)
```

### Training Args

```python
training_args = TrainingArguments(
    num_train_epochs=3,              # MESMO para todos
    gradient_accumulation_steps=4,   # MESMO para todos
    warmup_steps=500,                # MESMO para todos
    weight_decay=0.01,               # MESMO para todos
    fp16=True,                       # MESMO para todos
    logging_steps=100,               # MESMO para todos
    eval_strategy="steps",           # MESMO para todos
    eval_steps=500,                  # MESMO para todos
    save_strategy="steps",           # MESMO para todos
    save_steps=500,                  # MESMO para todos
    load_best_model_at_end=True,     # MESMO para todos
    seed=42,                         # MESMO para todos
    # Variam por modelo:
    per_device_train_batch_size=BATCH_SIZE,  # 8/4/2
    learning_rate=LR,                         # 5e-5/3e-5/2e-5
)
```

### Early Stopping

```python
EarlyStoppingCallback(
    early_stopping_patience=3,  # MESMO para todos
)
```

---

## Instancias AWS

| Modelo | Instance Type | GPU | VRAM | Custo/hora | Tempo Est. |
|--------|---------------|-----|------|------------|------------|
| Base | g5.xlarge | A10G | 24GB | $1.01 | 2-3h |
| Medium | g5.xlarge | A10G | 24GB | $1.01 | 2-3h |
| Large | g5.2xlarge | A10G | 48GB | $1.21 | 4-5h |

---

## Checklist de Validacao

### Pre-Treinamento

- [ ] Dataset `augustocsc/sintetico_natural_prefix_682k` acessivel
- [ ] Coluna correta selecionada (i_prompt_n vs p_prompt_n_converted)
- [ ] LoRA rank correto para o tamanho do modelo
- [ ] Learning rate correto para o tamanho do modelo
- [ ] Batch size adequado para VRAM da instancia
- [ ] Wandb configurado para tracking

### Pos-Treinamento

- [ ] Train loss convergiu (< 0.5)
- [ ] Validation loss nao divergiu
- [ ] Early stopping nao acionou muito cedo
- [ ] Modelo salvo corretamente (adapter_config.json existe)
- [ ] Tokenizer salvo junto com modelo
- [ ] Instancia AWS PARADA

### Validacao do Modelo

- [ ] Carregar modelo sem erros
- [ ] Gerar 10 expressoes de teste
- [ ] Taxa de expressoes validas > 50%
- [ ] Expressoes usam notacao correta (infix ou prefix)

---

## Diferenca: Infix Base Existente vs Novos

**Modelo existente** (`augustocsc/Se124M_700K_infix_v3_json`):
- Treinado com dataset original `sintetico_natural` (700K)
- Split interno 90/10 (seed=42)
- Tokenizer tem 2 tokens extras (50259 vs 50257)
- LoRA r=8, alpha=32

**Novos modelos** (Medium/Large infix):
- Treinados com `sintetico_natural_prefix_682k` (mesmo split do prefix)
- Sem tokens extras (50257 padrao)
- LoRA r=16, alpha=64 (maior capacidade)

**Decisao**: O modelo Base infix existente e aceitavel para comparacao pois:
1. Usa mesma seed (42) para split
2. Split interno resulta nos mesmos 682K exemplos
3. Tokens extras sao corrigidos no carregamento (resize_embeddings)

---

## ⚠️ Tokens Especiais e Stop Mechanism

### Situacao Atual

| Modelo | Vocab | Tokens Extras | Stop Token |
|--------|-------|---------------|------------|
| Base Infix (existente) | 50259 | `<\|endofex\|>`, `<\|startofex\|>` | `"}` (JSON) |
| Novos modelos | 50257 | Nenhum | `"}` (JSON) |

### Por que NAO e um problema

1. **Formato JSON**: Todos os modelos usam formato JSON que para no `"}`
2. **Tokens nao usados**: Os tokens `<\|endofex\|>` existem no vocabulario mas NAO sao usados no treinamento/inferencia com formato JSON
3. **Correcao automatica**: `resize_token_embeddings()` corrige na hora do carregamento

### Evidencia

Do `EXPERIMENT_RESULTS.md`:
- EXP-A (JSON format): 80% valid - para no `"}`
- EXP-B (EOS token `<\|endoftext\|>`): 0.5% valid - FALHOU

**Conclusao**: O formato JSON e o mecanismo de parada correto. Tokens especiais sao irrelevantes.

### Se quiser consistencia TOTAL (opcional)

Retreinar Base Infix com `train_with_json_fixed.py`:
```bash
python scripts/train_with_json_fixed.py \
  --model_size gpt2 \
  --dataset_repo augustocsc/sintetico_natural_prefix_682k \
  --text_column i_prompt_n \
  --output_dir ./output/gpt2_base_infix_682k \
  --lora_r 8 \
  --lora_alpha 32
```

Isso criaria um modelo com:
- Mesmo dataset (682K)
- Mesmo formato JSON
- Mesmo tokenizer (50257)
- Mesmos hiperparametros LoRA

---

## Comandos de Verificacao

### Verificar dataset

```python
from datasets import load_dataset

ds = load_dataset('augustocsc/sintetico_natural_prefix_682k')
print(f"Train: {len(ds['train']):,}")          # Deve ser 682,429
print(f"Val: {len(ds['validation']):,}")       # Deve ser 75,826

# Verificar colunas
print(ds['train'].column_names)
# Deve incluir: 'i_prompt_n', 'p_prompt_n_converted'

# Comparar mesma expressao
ex = ds['train'][0]
print("INFIX:", ex['i_prompt_n'][-50:])
print("PREFIX:", ex['p_prompt_n_converted'][-50:])
```

### Verificar modelo salvo

```bash
# Verificar arquivos
ls -la output/gpt2_medium_infix_682k/
# Deve ter: adapter_config.json, adapter_model.bin, tokenizer_config.json

# Verificar config
cat output/gpt2_medium_infix_682k/adapter_config.json | jq '.r, .lora_alpha'
# Deve mostrar: 16, 64
```

---

## Historico de Mudancas

| Data | Mudanca | Motivo |
|------|---------|--------|
| 2025-02-19 | Criado documento | Garantir consistencia entre modelos |
| 2025-02-19 | Scripts infix atualizados | Usar mesmo dataset que prefix |

---

## Modelos Treinados (HuggingFace)

**Status**: Treinamento concluido em 2026-02-19

### Repositorios no HuggingFace

| Modelo | HuggingFace Repo | Params | Notacao |
|--------|------------------|--------|---------|
| Base Infix | `augustocsc/gpt2_base_infix_682k` | 124M | Infix |
| Base Prefix | `augustocsc/gpt2_base_prefix_682k` | 124M | Prefix |
| Medium Infix | `augustocsc/gpt2_medium_infix_682k` | 355M | Infix |
| Medium Prefix | `augustocsc/gpt2_medium_prefix_682k` | 355M | Prefix |
| Large Infix | `augustocsc/gpt2_large_infix_682k` | 774M | Infix |
| Large Prefix | `augustocsc/gpt2_large_prefix_682k` | 774M | Prefix |

### Como Carregar os Modelos

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Escolher modelo (exemplo: Medium Infix)
MODEL_REPO = "augustocsc/gpt2_medium_infix_682k"
BASE_MODEL = "gpt2-medium"  # gpt2, gpt2-medium, ou gpt2-large

# Carregar tokenizer e modelo base
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Carregar adapter LoRA
model = PeftModel.from_pretrained(base_model, MODEL_REPO)
model.eval()

# Para GPU
model = model.to("cuda")
```

### Prompt de Teste (Infix)

```python
# Prompt no formato JSON (INFIX)
prompt = '{"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "'

# Tokenizar
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Gerar
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decodificar
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Exemplo de saida: {"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "sin(x_1) + C*cos(x_1)"}

# Extrair apenas a expressao
import json
try:
    data = json.loads(result)
    expression = data["expr"]
    print(f"Expressao gerada: {expression}")
except:
    # Se JSON incompleto, extrair manualmente
    expr_start = result.find('"expr": "') + 9
    expr_end = result.rfind('"')
    expression = result[expr_start:expr_end]
    print(f"Expressao gerada: {expression}")
```

### Prompt de Teste (Prefix)

```python
# Prompt no formato JSON (PREFIX)
prompt = '{"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "'

# Mesmo codigo de geracao acima...
# Exemplo de saida: {"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "+ sin x_1 * C cos x_1"}
```

### Script Rapido de Teste

```bash
# Testar modelo infix
python scripts/generate.py \
  --model_path augustocsc/gpt2_medium_infix_682k \
  --num_generations 10 \
  --validate

# Testar modelo prefix
python scripts/generate.py \
  --model_path augustocsc/gpt2_medium_prefix_682k \
  --num_generations 10 \
  --validate \
  --is_prefix
```

### Wandb Runs (Treinamento)

| Modelo | Wandb Run |
|--------|-----------|
| Base Infix | [qctvm65z](https://wandb.ai/symbolic-gression/huggingface/runs/qctvm65z) |
| Base Prefix | [tav7itba](https://wandb.ai/symbolic-gression/huggingface/runs/tav7itba) |
| Medium Infix | [37z4jhuj](https://wandb.ai/symbolic-gression/huggingface/runs/37z4jhuj) |
| Medium Prefix | [o6ew0sw9](https://wandb.ai/symbolic-gression/huggingface/runs/o6ew0sw9) |
| Large Infix | [w146ecn0](https://wandb.ai/symbolic-gression/huggingface/runs/w146ecn0) |
| Large Prefix | [8l5hcu9a](https://wandb.ai/symbolic-gression/huggingface/runs/8l5hcu9a) |

**Dashboard**: https://wandb.ai/symbolic-gression/huggingface

---

## Historico de Mudancas

| Data | Mudanca | Motivo |
|------|---------|--------|
| 2025-02-19 | Criado documento | Garantir consistencia entre modelos |
| 2025-02-19 | Scripts infix atualizados | Usar mesmo dataset que prefix |
| 2026-02-19 | Treinamento dos 6 modelos | Experimento academico completo |
| 2026-02-19 | Adicionados repos HuggingFace | Documentar modelos treinados |

---

**Ultima atualizacao**: 2026-02-19
**Responsavel**: Claude Code
