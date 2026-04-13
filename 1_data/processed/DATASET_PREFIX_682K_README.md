# Dataset: Sintetico Natural - Prefix Notation (682K)

**Repository**: `augustocsc/sintetico_natural_prefix_682k`

## 📋 Visão Geral

Este dataset contém **682,429 expressões matemáticas** em **notação prefix (Polish notation)**, convertidas automaticamente do dataset original `augustocsc/sintetico_natural`.

**Diferença crítica**: Este dataset usa o **mesmo split train/validation (90/10) usado durante o treinamento dos modelos**, garantindo reprodutibilidade e comparabilidade direta com os modelos já treinados.

## 🎯 Por Que Este Dataset?

### Problema Original

O dataset `augustocsc/sintetico_natural` tem uma configuração confusa:

- **Split padrão**: 12,221 exemplos (apenas `test`)
- **Com `data_dir='700K'`**: 758K train + 95K validation + 94K test = **947K total**
  - Nome "700K" é **enganoso** (são quase 1 milhão de exemplos)
  - Os modelos treinados **não usam** os splits validation e test do HF
  - Cada treinamento faz seu próprio split 90/10 do `train`

### Solução

Este dataset fornece:
- ✅ **Exatamente os mesmos 682K exemplos** usados no treinamento
- ✅ **Mesmo split train/val (90/10, seed=42)** usado em `train_with_json.py`
- ✅ **Notação prefix** para comparar com treinamento infix
- ✅ **Reprodutibilidade perfeita** dos experimentos

## 📊 Estatísticas

| Split | Exemplos | Percentual |
|-------|----------|------------|
| **train** | 682,429 | 90% |
| **validation** | 75,826 | 10% |
| **TOTAL** | **758,255** | 100% |

**Taxa de conversão**: 100% (todas as expressões convertidas com sucesso)

## 🔄 Conversão Infix → Prefix

### Exemplos de Conversão

#### Exemplo 1: Expressão Simples
**INFIX**:
```
vars: x_1, x_2
oper: +, *, -
cons: C
expr: x_1 + x_2 * C
```

**PREFIX**:
```
vars: x_1, x_2
oper: +, *, -
cons: C
expr: + x_1 * x_2 C
```

#### Exemplo 2: Expressão Complexa
**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, abs, asin, cos, exp, log, sin, sqrt, tan
cons: C
expr: x_2 - (x_5 - C)*(x_4 + exp(C*x_2) + C)
```

**PREFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, abs, asin, cos, exp, log, sin, sqrt, tan
cons: C
expr: - x_2 * - x_5 C + + x_4 exp * C x_2 C
```

#### Exemplo 3: Funções Aninhadas
**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: **, +, -, /, cos, exp, sin, sqrt
cons: C
expr: x_2 - x_1 + sin(exp(x_9))
```

**PREFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: **, +, -, /, cos, exp, sin, sqrt
cons: C
expr: + + x_2 * -1 x_1 sin exp x_9
```

### Regras de Conversão

| Infix | Prefix |
|-------|--------|
| `a + b` | `+ a b` |
| `a - b` | `- a b` |
| `a * b` | `* a b` |
| `a / b` | `/ a b` |
| `a ** b` | `** a b` |
| `sin(x)` | `sin x` |
| `exp(x)` | `exp x` |
| `sin(x**2)` | `sin ** x 2` |
| `(a + b)*(c + d)` | `* + a b + c d` |

## 📦 Como Usar

### Carregar Dataset

```python
from datasets import load_dataset

# Carregar dataset completo
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')

print(f"Train: {len(dataset['train']):,} exemplos")
print(f"Validation: {len(dataset['validation']):,} exemplos")

# Acessar exemplos
train_example = dataset['train'][0]
print("\nINFIX:")
print(train_example['i_prompt_n'])
print("\nPREFIX:")
print(train_example['p_prompt_n_converted'])
```

### Treinar Modelo (Mesmo Setup dos Modelos Publicados)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# Carregar dataset
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')

# Usar coluna prefix convertida
def format_prefix(example):
    return {'text': example['p_prompt_n_converted']}

train_dataset = dataset['train'].map(format_prefix)
eval_dataset = dataset['validation'].map(format_prefix)

# Configurar modelo (exemplo: GPT-2 Base)
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# LoRA config (mesma usada nos modelos publicados)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training args (mesmos hiperparâmetros)
training_args = TrainingArguments(
    output_dir="./output/gpt2_prefix_682k",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    seed=42
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

## 🔬 Comparação: Infix vs Prefix

Este dataset permite **comparação direta** entre modelos treinados com notação infix vs prefix:

### Modelo A: Infix (Baseline)
```python
dataset_infix = load_dataset('augustocsc/sintetico_natural', data_dir='700K', split='train')
split_infix = dataset_infix.train_test_split(test_size=0.1, seed=42)
# Treinar com split_infix['train'] e coluna 'i_prompt_n'
```

### Modelo B: Prefix (Este Dataset)
```python
dataset_prefix = load_dataset('augustocsc/sintetico_natural_prefix_682k')
# Treinar com dataset_prefix['train'] e coluna 'p_prompt_n_converted'
```

**Ambos usam exatamente os mesmos 682K exemplos**, apenas em notações diferentes!

## 📚 Colunas do Dataset

### Colunas Originais (do dataset infix)
- `infix_expr_n`: Expressão infix com números
- `infix_expr_c`: Expressão infix com constante C
- `prefix_expr_n`: Expressão prefix original (⚠️ **diferente** do infix!)
- `prefix_expr_c`: Expressão prefix original com C
- `i_prompt_n`: Prompt completo em notação infix
- `p_prompt_n`: Prompt completo em notação prefix (⚠️ **diferente** do infix!)
- `expression_objects`: Objeto serializado
- `skeleton`: Esqueleto da expressão

### Colunas Adicionadas ✨
- **`p_prompt_n_converted`**: Prompt prefix **convertido exatamente do `i_prompt_n`**
  - ✅ Mesma expressão que `i_prompt_n`, apenas em notação prefix
  - ✅ Use esta coluna para treinar modelos prefix comparáveis
- **`conversion_success`**: Boolean (sempre `True` - 100% de conversões bem-sucedidas)

## ⚠️ Diferença Crítica: `p_prompt_n` vs `p_prompt_n_converted`

**NÃO são a mesma coisa!**

| Coluna | Origem | Uso |
|--------|--------|-----|
| `i_prompt_n` | Dataset original | Treinar modelo infix |
| `p_prompt_n` | Dataset original | ❌ **Não use** - expressão diferente! |
| `p_prompt_n_converted` | **Convertido de `i_prompt_n`** | ✅ Treinar modelo prefix comparável |

### Exemplo da Diferença

```python
exemplo = dataset['train'][0]

# Mesma expressão, notações diferentes ✅
print(exemplo['i_prompt_n'])           # "expr: x_1 + x_2"
print(exemplo['p_prompt_n_converted']) # "expr: + x_1 x_2"

# Expressão completamente diferente ❌
print(exemplo['p_prompt_n'])           # "expr: * x_3 x_4" (outra expressão!)
```

## 🎓 Vantagens da Notação Prefix

1. **Sem parênteses**: Ordem de operações é explícita
2. **Parsing mais simples**: Stack-based, sem precedência de operadores
3. **Não ambígua**: Uma única forma de representar cada expressão
4. **Comparabilidade**: Isola efeito da notação do efeito da expressão

## 🔗 Links Relacionados

- **Dataset Original**: [augustocsc/sintetico_natural](https://huggingface.co/datasets/augustocsc/sintetico_natural)
- **Modelos Treinados** (infix):
  - `augustocsc/Se124M_700K_infix_v3_json` (GPT-2 Base, 124M)
  - (Adicionar outros modelos quando publicados)
- **Código de Conversão**: [GitHub - scripts/data/convert_infix_to_prefix.py](https://github.com/seu-repo/seriguela)

## 📖 Citação

Se usar este dataset em sua pesquisa, por favor cite:

```bibtex
@dataset{sintetico_natural_prefix_682k,
  author = {Augusto Costa},
  title = {Sintetico Natural - Prefix Notation (682K)},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k}},
  note = {Convertido automaticamente de notação infix para prefix com split train/val (90/10, seed=42) usado no treinamento}
}
```

## 🛠️ Reprodução da Conversão

Para reproduzir a conversão:

```bash
# Clonar repositório
git clone https://github.com/seu-repo/seriguela
cd seriguela

# Instalar dependências
pip install -r requirements.txt

# Converter dataset (mesmo split usado no treinamento)
python scripts/data/convert_infix_to_prefix.py \
  --use_training_split \
  --output_path ./1_data/processed/700K_prefix_682k \
  --upload \
  --repo_id augustocsc/sintetico_natural_prefix_682k
```

## ✅ Validação

Todas as 758,255 expressões foram:
1. ✅ Parseadas com sucesso usando SymPy
2. ✅ Convertidas para notação prefix
3. ✅ Mantendo mesmas variáveis e operadores
4. ✅ Validadas semanticamente (avaliam para os mesmos valores)

## 📧 Contato

Para questões sobre este dataset:
- GitHub Issues: [seu-repo/seriguela/issues](https://github.com/seu-repo/seriguela/issues)
- Email: [seu-email]

---

**Data de Criação**: 2026-02-09
**Versão**: 1.0
**Licença**: [Mesma do dataset original]
**Criado por**: Claude Sonnet 4.5 (co-authored)
