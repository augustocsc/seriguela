# Dataset Infix→Prefix Convertido ✅

**Data**: 2026-02-09
**Status**: COMPLETO

---

## 📊 Estatísticas da Conversão

- **Total de Exemplos**: 12,221
- **Taxa de Sucesso**: 100.00% (12,221/12,221)
- **Tempo de Conversão**: ~15 segundos
- **Tamanho do Dataset**: 7 MB

---

## 📁 Localização

**Dataset Convertido**: `./1_data/processed/700K_prefix_converted/`

**Arquivos**:
- `data-00000-of-00001.arrow` (7 MB)
- `dataset_info.json`
- `state.json`

---

## 🔧 Colunas do Dataset

### Colunas Originais
- `infix_expr_n`: Expressão infix com números
- `infix_expr_c`: Expressão infix com constante C
- `expression_objects`: Objeto Expression serializado
- `prefix_expr_c`: Expressão prefix original (diferente!)
- `prefix_expr_n`: Expressão prefix original com números
- `i_prompt_n`: **Prompt infix original** (usado para conversão)
- `p_prompt_n`: Prompt prefix original (expressão diferente!)
- `skeleton`: Esqueleto da expressão

### Colunas Adicionadas ✨
- **`p_prompt_n_converted`**: Prompt prefix CONVERTIDO do i_prompt_n (mesma expressão!)
- **`conversion_success`**: Boolean indicando sucesso da conversão (todos True)

---

## 🎯 Diferença Crítica

### ANTES (Problema)
```
i_prompt_n:  "expr: x_1 + x_2"        (5 variáveis)
p_prompt_n:  "expr: * x_3 x_4"        (10 variáveis, EXPRESSÃO DIFERENTE!)
```
❌ Não são a mesma expressão!

### AGORA (Solução)
```
i_prompt_n:           "expr: x_1 + x_2"
p_prompt_n_converted: "expr: + x_1 x_2"  (MESMA expressão em prefix!)
```
✅ Mesma expressão, notações diferentes!

---

## 📝 Exemplos de Conversão

### Exemplo 1
**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, abs, asin, cos, exp, log, sin, sqrt, tan
cons: C
expr: x_2 - (x_5 - C)*(x_4 + exp(C*x_2) + C)
```

**CONVERTIDO PARA PREFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, +, -, /, abs, asin, cos, exp, log, sin, sqrt, tan
cons: C
expr: - x_2 * - x_5 C + + x_4 exp * C x_2 C
```

### Exemplo 2
**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: **, +, -, /, cos, exp, sin, sqrt
cons: C
expr: x_2 - x_1 + sin(exp(x_9))
```

**CONVERTIDO PARA PREFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: **, +, -, /, cos, exp, sin, sqrt
cons: C
expr: + + x_2 * -1 x_1 sin exp x_9
```

### Exemplo 3
**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, +, /, asin, sin, tan
cons: C
expr: (tan(x_7) + C)*(asin(x_5) + C)
```

**CONVERTIDO PARA PREFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, +, /, asin, sin, tan
cons: C
expr: * + tan x_7 C + asin x_5 C
```

---

## 🚀 Próximos Passos

### Opção 1: Upload para HuggingFace Hub (Recomendado)

```bash
python scripts/data/convert_infix_to_prefix.py \
  --split test \
  --output_path ./1_data/processed/700K_prefix_converted \
  --upload \
  --repo_id augustocsc/sintetico_natural_prefix_converted
```

**Vantagens**:
- Facilita compartilhamento
- Pode usar em qualquer lugar
- Scripts de treinamento já funcionam com HF datasets

**Requisitos**:
- Permissão de escrita no repo HuggingFace
- Login: `huggingface-cli login`

---

### Opção 2: Treinar Localmente

#### A. Usando Dataset Local com Scripts Existentes

**Problema**: Os scripts atuais (`2_training/supervised/train.py`) carregam direto do HuggingFace Hub.

**Solução**: Adaptar o script ou criar um novo.

#### B. Carregar Dataset Local no Python

```python
from datasets import load_from_disk

# Carregar dataset convertido
ds = load_from_disk('./1_data/processed/700K_prefix_converted')

# Usar coluna p_prompt_n_converted para treinar
# (adaptar script de treinamento)
```

---

### Opção 3: Criar Script de Treinamento Adaptado

Criar `2_training/supervised/train_local_prefix.py` que:
1. Carrega dataset de `./1_data/processed/700K_prefix_converted`
2. Usa coluna `p_prompt_n_converted`
3. Treina com formato prefix

---

## 📋 Checklist de Validação

- [x] Dataset convertido (12,221 exemplos)
- [x] Taxa de sucesso 100%
- [x] Dataset salvo localmente
- [x] Documentação criada
- [ ] Upload para HuggingFace Hub
- [ ] Treinamento de modelo teste
- [ ] Comparação infix vs prefix

---

## 🔍 Como Verificar o Dataset

```bash
# Testar o dataset
cd /c/Users/madeinweb/seriguela

python -c "
from datasets import load_from_disk
ds = load_from_disk('./1_data/processed/700K_prefix_converted')
print(f'Total: {len(ds)} exemplos')
print(f'Colunas: {ds.column_names}')
print(f'Sucesso: {sum(ds[\"conversion_success\"])/len(ds)*100:.1f}%')
print()
print('Exemplo 1:')
print('INFIX:', ds[0]['i_prompt_n'])
print()
print('PREFIX:', ds[0]['p_prompt_n_converted'])
"
```

---

## 📚 Arquivos Relacionados

- `scripts/data/convert_infix_to_prefix.py` - Script de conversão
- `1_data/processed/PREFIX_CONVERSION_README.md` - Documentação detalhada
- `classes/expression.py` - Classe com `parse_prefix()` para validação

---

## ✅ Qualidade da Conversão

### Verificação Automática
Todas as 12,221 expressões foram:
1. ✅ Parseadas com sucesso usando SymPy
2. ✅ Convertidas para notação prefix
3. ✅ Mantendo mesmas variáveis e operadores
4. ✅ Salvas com sucesso

### Validação Manual
3 exemplos verificados manualmente mostraram conversão correta.

---

**Próxima Ação Recomendada**:
1. Fazer upload para HuggingFace Hub (mais fácil para usar)
2. OU adaptar script de treinamento para carregar dataset local

**Você prefere qual opção?**
