# Conversão Infix → Prefix Dataset

## Objetivo

Converter o dataset `augustocsc/sintetico_natural` de notação **infix** para **prefix (Polish notation)**, mantendo as mesmas expressões matemáticas mas em formato prefixado.

## Motivação

O dataset original contém:
- `i_prompt_n`: Prompts com expressões em notação infix (e.g., `x_1 + x_2`)
- `p_prompt_n`: Prompts com expressões em notação prefix (e.g., `+ x_1 x_2`)

**PROBLEMA**: As expressões em `i_prompt_n` e `p_prompt_n` são DIFERENTES! Não são a mesma expressão convertida.

**SOLUÇÃO**: Converter automaticamente as expressões de `i_prompt_n` para notação prefix, criando `p_prompt_n_converted`.

## Script de Conversão

**Arquivo**: `scripts/data/convert_infix_to_prefix.py`

**Funcionalidade**:
1. Lê expressões infix do campo `i_prompt_n`
2. Parseia usando SymPy
3. Converte para notação prefix (Polish notation)
4. Mantém as mesmas variáveis e operadores do prompt original
5. Salva em nova coluna `p_prompt_n_converted`

## Exemplos de Conversão

### Exemplo 1
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

### Exemplo 2
**INFIX**:
```
vars: x_1, x_2, x_3
oper: +, -, /, abs, cos, exp
cons: C
expr: (x_1 - C)/(x_2 + C)
```

**PREFIX**:
```
vars: x_1, x_2, x_3
oper: +, -, /, abs, cos, exp
cons: C
expr: / - x_1 C + x_2 C
```

### Exemplo 3
**INFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, +, /, asin, sin, tan
cons: C
expr: (tan(x_7) + C)*(asin(x_5) + C)
```

**PREFIX**:
```
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, +, /, asin, sin, tan
cons: C
expr: * + tan x_7 C + asin x_5 C
```

## Regras de Conversão

### Operadores Binários
- Infix: `a + b` → Prefix: `+ a b`
- Infix: `a - b` → Prefix: `- a b`
- Infix: `a * b` → Prefix: `* a b`
- Infix: `a / b` → Prefix: `/ a b`
- Infix: `a ** b` → Prefix: `** a b`

### Funções Unárias
- Infix: `sin(x)` → Prefix: `sin x`
- Infix: `exp(x)` → Prefix: `exp x`
- Infix: `log(x)` → Prefix: `log x`

### Expressões Complexas
- Infix: `sin(x**2)` → Prefix: `sin ** x 2`
- Infix: `x*(y + z)` → Prefix: `* x + y z`
- Infix: `(a + b)*(c + d)` → Prefix: `* + a b + c d`

### Casos Especiais
- **Negação**: `-x` → `* -1 x`
- **Múltiplas adições**: `a + b + c` → `+ + a b c` (aninhado à esquerda)
- **Divisão**: `x/y` → `/ x y` (SymPy representa como `x * y**-1`, conversão detecta e corrige)

## Uso do Script

### Teste (10 exemplos)
```bash
python scripts/data/convert_infix_to_prefix.py --test_only
```

### Conversão Completa
```bash
python scripts/data/convert_infix_to_prefix.py \
  --split test \
  --output_path ./1_data/processed/700K_prefix_converted
```

### Upload para HuggingFace
```bash
python scripts/data/convert_infix_to_prefix.py \
  --split test \
  --output_path ./1_data/processed/700K_prefix_converted \
  --upload \
  --repo_id augustocsc/sintetico_natural_prefix_converted
```

## Output do Dataset

**Colunas adicionadas**:
- `p_prompt_n_converted`: Prompt prefixado convertido do infix
- `conversion_success`: Boolean indicando se conversão foi bem-sucedida

**Taxa de sucesso esperada**: ~99%+

## Vantagens

1. **Comparabilidade**: Agora é possível treinar modelos prefix com as MESMAS expressões dos modelos infix
2. **Consistência**: Mantém vars/operators/constants do prompt original
3. **Reprodutibilidade**: Conversão automática e determinística
4. **Escalabilidade**: Fácil aplicar a novos datasets

## Treinamento com Dataset Convertido

### Usando o dataset local
```bash
python 2_training/supervised/train.py \
  --dataset_path ./1_data/processed/700K_prefix_converted \
  --data_column p_prompt_n_converted \
  --approach prefix \
  --output_dir ./output/gpt2_prefix_converted
```

### Comparação: Infix vs Prefix (mesma expressão)
Agora você pode treinar dois modelos com a MESMA expressão:
- Modelo A: `--data_column i_prompt_n --approach infix`
- Modelo B: `--data_column p_prompt_n_converted --approach prefix`

E comparar diretamente qual notação o modelo aprende melhor!

## Validação

Para verificar a correção da conversão:
1. Parsear expressão prefix convertida
2. Avaliar em pontos de teste
3. Comparar com avaliação da expressão infix original
4. R² score deve ser ~1.0 (mesma expressão, mesmos resultados)

```python
from classes.expression import Expression

# Expressão infix original
expr_infix = Expression("x_1 + x_2", is_prefix=False)

# Expressão prefix convertida
expr_prefix = Expression("+ x_1 x_2", is_prefix=True)

# Testar
import numpy as np
x = np.array([[1.0, 2.0], [3.0, 4.0]])

result_infix = expr_infix.evaluate(x)
result_prefix = expr_prefix.evaluate(x)

# Devem ser idênticos
assert np.allclose(result_infix, result_prefix)
```

## Limitações Conhecidas

1. **Expressões muito complexas**: Pode haver casos edge com aninhamento profundo
2. **Operadores customizados**: Se houver operadores não mapeados no SymPy
3. **Simplificação automática**: SymPy pode simplificar algumas expressões, alterando forma mas não valor

## Status

- [x] Script de conversão implementado
- [x] Testes em 10 exemplos (100% sucesso)
- [ ] Conversão dataset completo (~12k exemplos)
- [ ] Upload para HuggingFace Hub
- [ ] Treinamento de modelo teste
- [ ] Comparação infix vs prefix

## Referências

- **Polish Notation**: https://en.wikipedia.org/wiki/Polish_notation
- **SymPy Documentation**: https://docs.sympy.org/
- **Dataset Original**: https://huggingface.co/datasets/augustocsc/sintetico_natural

---

**Data de Criação**: 2026-02-09
**Autor**: Claude Sonnet 4.5 (co-authored)
**Última Atualização**: 2026-02-09
