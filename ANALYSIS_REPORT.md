# Seriguela - Relatório Consolidado de Análise

**Data:** 2026-02-01
**Status:** ⚠️ BLOCK 2 PRECISA RETREINO

---

## Resumo Executivo

Projeto Seriguela tem 3 blocos:
1. **Block 1 - Dados:** Preparação e análise ✅
2. **Block 2 - Treino Supervisionado:** Treinar LLM para gerar expressões ❌ PROBLEMA
3. **Block 3 - PPO Finetuning:** Otimizar para symbolic regression ⛔ BLOQUEADO

**Conclusão:** Os modelos V1 e V2 no HuggingFace Hub **NÃO funcionam** como documentado. Ambos geram 0% de expressões válidas. Precisa retreinar.

---

## Modelos Testados

| Modelo | HuggingFace Hub | Esperado | Real | Status |
|--------|-----------------|----------|------|--------|
| V1 | augustocsc/Se124M_700K_infix | 83.3% válidas | **0%** | ❌ Falha |
| V2 | augustocsc/Se124M_700K_infix_v2 | 90% válidas | **0%** | ❌ Falha |

---

## Testes Realizados

### Teste 1: Comparação V1 vs V2 (mesmo prompt)

**Prompt:**
```
vars: x_1, x_2
oper: *, +, -, sin, cos
cons: C
expr:
```

**Configurações ótimas usadas:**
- V1: temp=0.5, top_k=40, top_p=0.9, rep_penalty=1.15
- V2: temp=0.7, top_k=0, top_p=0.8, rep_penalty=1.0

**Resultados (20 gerações cada):**

| Métrica | V1 | V2 |
|---------|----|----|
| Expressões Válidas | 0% | 0% |
| Símbolos Corretos | 0% | 45% |

### Teste 2: PPO Evaluation

**Objetivo:** Verificar se modelo pode ser usado para PPO (symbolic regression)

**Resultados:**
- Valid Rate: 6.7% (muito baixo)
- Best R²: N/A (não conseguiu computar)
- **Conclusão:** PPO inviável com modelo atual

---

## Problemas Identificados

### 1. Modelos Não Param Corretamente

**Sintoma:** Expressões continuam além do esperado
```
Esperado: C*x_1 + sin(x_2)<|endofex|>
Gerado:   C*x_1 + sin(x_2) + C Stockholmvars: x_1, x_2, x_3...
```

**Causa:** Modelo não aprendeu a gerar `<|endofex|>`

### 2. Garbage Tokens na Saída

**Exemplos de lixo gerado:**
- "BuyableInstoreAndOnline"
- "Stockholm", "GREEN", "Muslims"
- "intuition", "records", "crash"
- "xstatics", "xid", "sinmod"

**Causa:** Dados de treino contaminados OU modelo não convergiu

### 3. Variáveis Erradas

**Sintoma:** Usa variáveis não permitidas
```
Prompt pede: x_1, x_2
Modelo gera: x_9, x_10, x_3, x_4
```

**Causa:** Modelo não aprendeu a respeitar o prompt

### 4. Discrepância com Documentação

**Documentação dizia:**
- V1: 83.3% válidas com config otimizada
- V2: 90% válidas com nucleus sampling

**Realidade:**
- V1: 0% válidas
- V2: 0% válidas

**Possíveis causas:**
1. Modelos no Hub não são os mesmos testados
2. Testes anteriores tinham bug
3. Forma de carregar modelo está errada

---

## Configurações de Inferência Testadas

### V1 Config Ótima (segundo docs)
```python
{
    "temperature": 0.5,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "max_new_tokens": 100,
    "do_sample": True,
}
```

### V2 Config Ótima (segundo docs)
```python
{
    "temperature": 0.7,
    "top_k": 0,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
    "max_new_tokens": 128,
    "do_sample": True,
}
```

**Resultado:** Mesmo com configs ótimas, 0% válidas.

---

## Forma de Carregar Modelos

```python
# 1. Carregar base GPT-2
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

# 2. Configurar tokenizer com tokens especiais
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
})

# 3. Redimensionar embeddings
model.resize_token_embeddings(len(tokenizer))

# 4. Carregar adapter LoRA
model = PeftModel.from_pretrained(model, "augustocsc/Se124M_700K_infix_v2")

# 5. Merge adapter no modelo base
model = model.merge_and_unload()
model.eval()
```

---

## Conclusões

### Block 2 (Treino) - PRECISA RETREINO

**Problemas no treino:**
1. Modelo não aprendeu `<|endofex|>` marker
2. Dados podem estar contaminados com garbage
3. Modelo não respeita variáveis do prompt

**Ações necessárias:**
1. Validar dados de treino (100% devem ter `<|endofex|>`)
2. Limpar garbage tokens dos dados
3. Monitorar valid rate durante treino
4. Só considerar treino bem-sucedido se valid rate > 80%

### Block 3 (PPO) - BLOQUEADO

**Pré-requisitos para PPO:**
- ✅ Base model gera >80% expressões válidas
- ✅ Expressões podem ser avaliadas (R² computável)
- ✅ Modelo para corretamente em boundaries

**Status atual:** ❌ Nenhum pré-requisito atendido

---

## Próximos Passos

1. **Investigar dados de treino**
   - Verificar se `<|endofex|>` está presente
   - Identificar fonte de garbage tokens

2. **Retreinar modelo (V3)**
   - Usar dados validados
   - Monitorar valid rate durante treino
   - Validar antes de fazer push pro Hub

3. **Só então testar PPO**
   - Após valid rate > 80%
   - Com modelo que para corretamente

---

## Arquivos de Código Relevantes

- `scripts/train.py` - Script de treino
- `scripts/generate.py` - Geração com stopping criteria
- `scripts/evaluate.py` - Avaliação de modelo
- `scripts/compare_v1_v2_simple.py` - Comparação V1 vs V2
- `scripts/evaluate_ppo.py` - Avaliação para PPO
- `scripts/data/prepare_training_data_fixed.py` - Preparação de dados
- `classes/expression.py` - Parsing e validação de expressões

---

## Infraestrutura AWS

- **Instance:** g5.xlarge (NVIDIA A10G, 24GB)
- **Instance ID:** i-0377b6c8de3660a82
- **Custo:** ~$1/hora
- **Status atual:** Stopped (para economizar)

---

**Última atualização:** 2026-02-01
