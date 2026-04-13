# Experimento: Comparacao de Modelos Modernos para Regressao Simbolica

## Objetivo

Avaliar se modelos pre-treinados modernos (2024-2025) oferecem vantagens sobre o GPT-2 (2019) para a tarefa de regressao simbolica, mesmo sendo de tamanho similar.

## Hipotese

Modelos mais recentes, treinados com arquiteturas otimizadas e em datasets maiores/melhores curados, podem apresentar melhor desempenho em regressao simbolica mesmo com numero similar de parametros.

## Modelos Selecionados

### 1. SmolLM2-135M (Prioridade Alta)
- **Repositorio**: `HuggingFaceTB/SmolLM2-135M`
- **Parametros**: 135M
- **Licenca**: Apache 2.0
- **Por que**: Tamanho identico ao GPT-2 base, arquitetura moderna (2024), **2 trilhoes de tokens** de treino
- **LoRA target_modules**: `["q_proj", "v_proj"]`
- **Arquitetura**: Transformer decoder otimizado
- **Dados de treino**: FineWeb-Edu, DCLM, The Stack (2T tokens)
- **Precisao**: bfloat16
- **Paper**: arXiv 2502.02737

### 2. Pythia-160M (Prioridade Alta)
- **Repositorio**: `EleutherAI/pythia-160m`
- **Parametros**: 162M
- **Licenca**: Apache 2.0
- **Por que**: Melhor documentado, 154 checkpoints intermediarios, ideal para pesquisa academica
- **LoRA target_modules**: `["query_key_value"]`
- **Arquitetura**: GPTNeoX (12 layers, 768 dim, 12 heads)
- **Dados de treino**: The Pile (825GiB, 300B tokens)
- **Nota**: Sem atualizacoes desde 2023, mas continua sendo referencia para pesquisa

### 3. Qwen3-0.6B (Prioridade Media)
- **Repositorio**: `Qwen/Qwen3-0.6B`
- **Parametros**: 600M
- **Licenca**: Apache 2.0
- **Por que**: Geracao mais recente (2025), arquitetura estado da arte (SwiGLU, GQA, RoPE)
- **LoRA target_modules**: `["q_proj", "k_proj", "v_proj"]`
- **Arquitetura**: 28 layers, 16 Q heads, 8 KV heads
- **Contexto**: 32K tokens
- **Requisito**: transformers>=4.51.0
- **Base model**: `Qwen/Qwen3-0.6B-Base` (usar este para fine-tuning)

### 4. CodeGen-350M-mono (Prioridade Media)
- **Repositorio**: `Salesforce/codegen-350M-mono`
- **Parametros**: 350M
- **Licenca**: BSD-3
- **Por que**: Treinado em Python (71.7B tokens), pode entender sintaxe matematica melhor
- **LoRA target_modules**: `["qkv_proj"]`
- **Hipotese especifica**: Pre-treino em codigo ajuda em regressao simbolica?
- **Nota**: CodeGen2 existe mas menor versao e 1B (muito grande para comparacao direta)

## Baseline (Modelos Atuais)

| Modelo | Params | Repositorio |
|--------|--------|-------------|
| GPT-2 Base Infix | 124M | augustocsc/gpt2_base_infix_682k |
| GPT-2 Base Prefix | 124M | augustocsc/gpt2_base_prefix_682k |

## Design Experimental

### Variaveis Controladas
- **Dataset**: augustocsc/sintetico_natural_prefix_682k (682K exemplos)
- **Notacao**: Testar ambas (infix e prefix) para cada modelo
- **Configuracao LoRA**: r=8, alpha=32, dropout=0.05
- **Epocas**: 3
- **Early stopping**: patience=3

### Variaveis Independentes
- Modelo base (GPT-2 vs Pythia vs SmolLM vs CodeGen)

### Metricas de Avaliacao
1. **Taxa de expressoes validas** (%)
2. **R² medio** nos benchmarks Nguyen 1-12
3. **Complexidade das expressoes** geradas
4. **Loss de validacao** final

## Adaptacoes Necessarias no Codigo

### Script de Treinamento
Arquivo: `2_training/supervised/train_with_json.py`

```python
# Adicionar mapeamento de target_modules por modelo
TARGET_MODULES = {
    "gpt2": ["c_attn"],
    "gpt2-medium": ["c_attn"],
    "gpt2-large": ["c_attn"],
    "HuggingFaceTB/SmolLM2-135M": ["q_proj", "v_proj"],
    "EleutherAI/pythia-160m": ["query_key_value"],
    "Qwen/Qwen3-0.6B-Base": ["q_proj", "k_proj", "v_proj"],
    "Salesforce/codegen-350M-mono": ["qkv_proj"],
}

# Requisitos de versao do transformers
TRANSFORMERS_MIN_VERSION = {
    "HuggingFaceTB/SmolLM2-135M": "4.37.0",
    "EleutherAI/pythia-160m": "4.20.0",
    "Qwen/Qwen3-0.6B-Base": "4.51.0",  # IMPORTANTE: versao alta
    "Salesforce/codegen-350M-mono": "4.20.0",
}
```

### Consideracoes de Tokenizer
- Cada modelo usa tokenizer diferente
- O formato JSON deve funcionar com todos (tokens ASCII padrao)
- Verificar se pad_token precisa ser configurado

## Experimentos Descartados

### Treinar do Zero
- **Motivo**: Resultado previsivel (pre-treino ajuda com dataset pequeno de 682K)
- **Custo**: Alto (treinamento completo vs LoRA)
- **Valor cientifico**: Baixo (ja bem estabelecido na literatura)

## Recursos Estimados

| Modelo | VRAM | Tempo (estimado) |
|--------|------|------------------|
| SmolLM2-135M | ~4GB | Similar ao GPT-2 base |
| Pythia-160M | ~4GB | Similar ao GPT-2 base |
| Qwen3-0.6B-Base | ~6GB | Ligeiramente maior |
| CodeGen-350M | ~6GB | Similar ao GPT-2 medium |

## Resultados Esperados

### Cenario A: Modelos modernos superam GPT-2
- Indica que arquiteturas otimizadas e dados de treino melhores sao mais importantes que tamanho
- Recomendacao: Migrar para SmolLM ou Pythia como base

### Cenario B: GPT-2 permanece competitivo
- Indica que a tarefa especifica (regressao simbolica) nao se beneficia tanto de pre-treino geral
- O fine-tuning domina o desempenho

### Cenario C: CodeGen supera todos
- Indica que pre-treino em codigo e relevante para regressao simbolica
- Abre linha de pesquisa sobre modelos especializados em codigo/matematica

## Referencias

- SmolLM2: https://arxiv.org/abs/2502.02737
- SmolLM Blog: https://huggingface.co/blog/smollm
- Pythia: https://arxiv.org/abs/2304.01373
- Qwen3: https://qwenlm.github.io/blog/qwen3/
- CodeGen: https://arxiv.org/abs/2203.13474

---

**Status**: Proposto
**Data**: 2026-02-20
**Ultima atualizacao**: 2026-02-20 (verificado versoes mais recentes)
**Prioridade**: Media (apos conclusao da avaliacao atual)
