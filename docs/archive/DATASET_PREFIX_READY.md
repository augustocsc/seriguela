# ✅ Dataset Prefix Pronto para Uso!

**Data**: 2026-02-09
**Status**: ✅ COMPLETO E PUBLICADO

---

## 🎯 O Que Foi Feito

### 1. Conversão Completa ✅
- **12,221 expressões** convertidas de infix para prefix
- **Taxa de sucesso**: 100%
- **Tempo**: ~8 segundos
- **Nova coluna**: `p_prompt_n_converted`

### 2. Upload para HuggingFace ✅
- **Repositório**: `augustocsc/sintetico_natural_prefix`
- **URL**: https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix
- **Tamanho**: 2.30 MB (comprimido)
- **Tempo de upload**: ~3 segundos

---

## 🚀 Como Usar o Dataset

### Carregar do HuggingFace Hub

```python
from datasets import load_dataset

# Carregar dataset convertido
ds = load_dataset('augustocsc/sintetico_natural_prefix', split='train')

print(f"Total de exemplos: {len(ds)}")
print(f"Colunas: {ds.column_names}")

# Ver exemplo
print("\nExemplo:")
print("INFIX:", ds[0]['i_prompt_n'])
print("PREFIX:", ds[0]['p_prompt_n_converted'])
```

### Carregar Localmente (se preferir)

```python
from datasets import load_from_disk

ds = load_from_disk('./1_data/processed/700K_prefix_converted')
```

---

## 🏋️ Treinar Modelo com Formato Prefix

### Opção 1: Usando Script Existente (Adaptar)

O script `2_training/supervised/train.py` precisa ser adaptado para usar a nova coluna.

**Comando sugerido** (após adaptação):
```bash
python 2_training/supervised/train.py \
  --model_name_or_path gpt2 \
  --dataset_repo_id augustocsc/sintetico_natural_prefix \
  --data_dir . \
  --data_column p_prompt_n_converted \
  --approach prefix \
  --output_dir ./output/gpt2_prefix_converted \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-5 \
  --fp16 \
  --wandb_project seriguela \
  --wandb_run_name gpt2-prefix-converted
```

### Opção 2: Script Customizado

```python
#!/usr/bin/env python
"""Train GPT-2 with prefix notation dataset."""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Load dataset
print("Loading prefix dataset...")
dataset = load_dataset('augustocsc/sintetico_natural_prefix', split='train')

# Use p_prompt_n_converted column
dataset = dataset.rename_column('p_prompt_n_converted', 'text')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained('gpt2')

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output/gpt2_prefix_converted",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="wandb",
    run_name="gpt2-prefix-converted"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save
trainer.save_model()
print("Model saved to ./output/gpt2_prefix_converted")
```

---

## 📊 Comparação: Infix vs Prefix

Agora você pode treinar dois modelos com a **MESMA expressão** em notações diferentes:

### Modelo A: Infix (Baseline)
```bash
python 2_training/supervised/train.py \
  --dataset_repo_id augustocsc/sintetico_natural \
  --data_column i_prompt_n \
  --approach infix \
  --output_dir ./output/gpt2_infix_baseline
```

### Modelo B: Prefix (Novo)
```bash
python 2_training/supervised/train.py \
  --dataset_repo_id augustocsc/sintetico_natural_prefix \
  --data_column p_prompt_n_converted \
  --approach prefix \
  --output_dir ./output/gpt2_prefix_converted
```

### Comparar Resultados
```bash
python 3_evaluation/comparison/compare_trained_models.py \
  --model_base ./output/gpt2_infix_baseline \
  --model_medium ./output/gpt2_prefix_converted \
  --dataset 1_data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 10
```

**Pergunta de Pesquisa**: Qual notação o modelo aprende melhor?

---

## 🔍 Validar Conversão

### Script de Validação

```python
#!/usr/bin/env python
"""Validate that prefix conversion is correct."""

from datasets import load_dataset
from classes.expression import Expression
import numpy as np

# Load dataset
ds = load_dataset('augustocsc/sintetico_natural_prefix', split='train')

print("Validating prefix conversions...")
errors = 0

for i in range(min(100, len(ds))):  # Test first 100
    try:
        # Parse infix
        infix_text = ds[i]['i_prompt_n'].split('expr:')[1].strip()
        expr_infix = Expression(infix_text, is_prefix=False)

        # Parse prefix converted
        prefix_text = ds[i]['p_prompt_n_converted'].split('expr:')[1].strip()
        expr_prefix = Expression(prefix_text, is_prefix=True)

        # Test on random data
        x = np.random.rand(10, 5)  # 10 samples, 5 variables

        result_infix = expr_infix.evaluate(x)
        result_prefix = expr_prefix.evaluate(x)

        # Compare
        if not np.allclose(result_infix, result_prefix, rtol=1e-5):
            print(f"[ERROR] Example {i}: Results don't match!")
            print(f"  Infix:  {infix_text}")
            print(f"  Prefix: {prefix_text}")
            errors += 1

    except Exception as e:
        print(f"[ERROR] Example {i}: {e}")
        errors += 1

if errors == 0:
    print(f"\n✅ All 100 conversions validated successfully!")
else:
    print(f"\n❌ {errors}/100 conversions had errors")
```

---

## 📝 Exemplos do Dataset

### Exemplo 1: Expressão Complexa

**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, abs, asin, cos, exp, log, sin, sqrt, tan
cons: C
expr: x_2 - (x_5 - C)*(x_4 + exp(C*x_2) + C)
```

**PREFIX CONVERTIDO**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, abs, asin, cos, exp, log, sin, sqrt, tan
cons: C
expr: - x_2 * - x_5 C + + x_4 exp * C x_2 C
```

### Exemplo 2: Expressão com Funções Aninhadas

**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: **, +, -, /, cos, exp, sin, sqrt
cons: C
expr: x_2 - x_1 + sin(exp(x_9))
```

**PREFIX CONVERTIDO**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: **, +, -, /, cos, exp, sin, sqrt
cons: C
expr: + + x_2 * -1 x_1 sin exp x_9
```

### Exemplo 3: Expressão Simples

**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, +, /, asin, sin, tan
cons: C
expr: (tan(x_7) + C)*(asin(x_5) + C)
```

**PREFIX CONVERTIDO**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, +, /, asin, sin, tan
cons: C
expr: * + tan x_7 C + asin x_5 C
```

---

## 🎓 Vantagens do Formato Prefix

### 1. Estrutura Mais Clara
- Operador sempre vem primeiro
- Não precisa de parênteses
- Ordem de avaliação explícita

### 2. Parsing Mais Simples
- Algoritmo stack-based
- Sem ambiguidade de precedência
- Mais eficiente

### 3. Comparabilidade
- Agora pode comparar infix vs prefix com mesmas expressões
- Isola efeito da notação do efeito da expressão

---

## 📚 Arquivos Criados

1. ✅ `scripts/data/convert_infix_to_prefix.py` - Script de conversão
2. ✅ `1_data/processed/700K_prefix_converted/` - Dataset local
3. ✅ `1_data/processed/PREFIX_CONVERSION_README.md` - Guia técnico
4. ✅ `DATASET_PREFIX_CONVERTED_STATUS.md` - Status da conversão
5. ✅ `DATASET_PREFIX_READY.md` - Este arquivo (instruções de uso)
6. ✅ **HuggingFace Hub**: `augustocsc/sintetico_natural_prefix`

---

## 🚀 Próximos Passos Recomendados

### 1. Testar Treinamento
```bash
# Teste rápido (1 época)
python train_prefix.py --num_train_epochs 1 --save_strategy no
```

### 2. Comparar com Infix
Treinar ambos os modelos e comparar:
- Valid expression rate
- R² scores em Nguyen benchmarks
- Diversidade de expressões
- Complexidade das expressões geradas

### 3. Publicar Resultados
- Documentar diferenças de performance
- Criar model cards para ambos
- Adicionar ao relatório de pesquisa

---

## ✅ Checklist Completo

- [x] Dataset convertido (12,221 exemplos)
- [x] Taxa de sucesso 100%
- [x] Dataset salvo localmente
- [x] Upload para HuggingFace Hub
- [x] Documentação completa criada
- [ ] Treinamento de modelo teste
- [ ] Comparação infix vs prefix
- [ ] Publicação de resultados

---

## 🔗 Links Importantes

- **Dataset no HuggingFace**: https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix
- **Dataset Original**: https://huggingface.co/datasets/augustocsc/sintetico_natural
- **Script de Conversão**: `scripts/data/convert_infix_to_prefix.py`
- **Documentação Técnica**: `1_data/processed/PREFIX_CONVERSION_README.md`

---

## 🤝 Contribuindo

Se encontrar algum problema:
1. Verificar se a conversão está correta (validação script)
2. Reportar issue no GitHub
3. Sugerir melhorias no algoritmo de conversão

---

**Pronto para treinar! 🚀**

```bash
# Comando exemplo para começar
python 2_training/supervised/train.py \
  --dataset_repo_id augustocsc/sintetico_natural_prefix \
  --data_column p_prompt_n_converted \
  --approach prefix \
  --output_dir ./output/gpt2_prefix_converted \
  --num_train_epochs 3
```

---

**Data de Criação**: 2026-02-09
**Status**: ✅ PRONTO PARA USO
**Autor**: Claude Sonnet 4.5 (co-authored)
