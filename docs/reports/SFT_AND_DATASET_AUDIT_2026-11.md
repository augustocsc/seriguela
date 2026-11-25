# Auditoria Empírica: Corpus Sintético (682K) e Modelos SFT (HuggingFace)

**Data:** 2026-11-25  
**Objetivo:** Substituir descrições superficiais ou divergentes da dissertação (Cap. 3, §§3.2–3.3 e Cap. 4, §4.2) pelos dados reais medidos diretamente nos artefatos publicados no HuggingFace (`augustocsc/sintetico_natural_prefix_682k` e os 6 repositórios `augustocsc/gpt2_{base,medium,large}_{infix,prefix}_682k`).

---

## 1. Corpus Sintético (`augustocsc/sintetico_natural_prefix_682k`)

### 1.1 Estrutura, Splits e Unicidade
- **Arquivos oficiais:** `data/train-00000-of-00001.parquet` (128,7 MB) e `data/validation-00000-of-00001.parquet` (14,4 MB).
- **Splits exatos (90/10, `seed=42`):**
  - `train`: **682.429** exemplos
  - `validation`: **75.826** exemplos
  - **Total:** **758.255** exemplos
- **10 colunas disponíveis:** `infix_expr_n`, `infix_expr_c`, `expression_objects`, `prefix_expr_c`, `prefix_expr_n`, `i_prompt_n`, `p_prompt_n`, `skeleton`, `p_prompt_n_converted`, `conversion_success` (`100%` `True`).
- **Unicidade e sobreposição:** `671.749` expressões distintas em `infix_expr_n` (`86.506` repetições geradas antes/durante aumentação; `7.615` expressões com intersecção exata entre `train` e `validation` — importante declarar no Cap. 3 como detalhe honesto do split aleatório pós-aumentação).
- **Scripts originais da engenharia de dados:** embora ausentes do Git atual, estão preservados integralmente dentro dos repositórios Prefix no HuggingFace (`scripts/data/{data_processing,data_cleaning,data_augmentation,convert_infix_to_prefix,prepare_training_data_fixed}.py`).

### 1.2 Pipeline Real dos Prompts (Texto → JSON)
- No Parquet do dataset, `i_prompt_n` e `p_prompt_n_converted` estão no formato texto estruturado (`vars: ... / oper: ... / cons: C / expr: ...`), onde:
  - constantes numéricas de `infix_expr_n` (ex.: `1.43`) foram abstraídas para o token `C` em `infix_expr_c`;
  - variáveis foram remapeadas para índices em `x_1 .. x_10` e **sobre-provisionadas** no prompt: média de **7,44 variáveis oferecidas** para **2,41 variáveis efetivamente usadas** por expressão;
  - a lista de operadores (`oper`) está **100% em ordem alfabética** (`0/5.000` amostras fora de ordem), sorteada dentre 13 símbolos: `*`, `**`, `+`, `-`, `/`, `abs`, `asin`, `cos`, `exp`, `log`, `sin`, `sqrt`, `tan`.
- No treinamento SFT (`train_with_json.py`), cada exemplo é convertido *on-the-fly* para JSON:
  ```json
  {"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "sin(x_1 + C*x_2)"}
  ```
  Essa escolha veio do experimento preliminar **EXP-A (JSON) vs. EXP-B (`<|endoftext|>`)**, em que o fechamento explícito `"}` elevou a taxa de expressões válidas de **0,5%** para **80,0%** (`eval_loss` 0,366 → 0,298).

### 1.3 Estatísticas Estruturais do Corpus (`N = 758.255`)
- **Comprimento em tokens** (amostra `50.000`): p5 = `8`, p25 = `11`, **mediana = `14`**, p75 = `16`, p95 = `21`, p99 = `25`, máx = `43`.
- **Variáveis distintas usadas por expressão** (`50.000`): 1 var = `22,6%`, 2 vars = `54,1%`, 3 vars = `21,5%`, 4 vars = `1,8%`.
- **Presença de operadores nas expressões:**
  - Binários: `*` (`80,0%`), `+` (`71,1%`), `-` (`53,3%`), `**` (`39,7%`), `/` (`29,3%`).
  - Unários: `sin` (`20,1%`), `cos` (`20,0%`), `tan` (`20,0%`), `log` (`18,7%`), `exp` (`18,5%`), `sqrt` (`12,3%`), `asin` (`10,2%`), **`abs` (`0,0%` — aparece apenas como distrator no prompt)**.
- **Explicação empírica dos `NaN` em domínios com negativos:** **`223.841` expressões (`29,52%` do corpus)** contêm `log(...)` ou `sqrt(...)` e **nenhuma** usa `abs(...)` para proteção de domínio. Isso explica diretamente a alta fração de `NaN` quando o prior é amostrado em `[-1, 1]` (Nguyen 1–6 canônico).

### 1.4 Auditoria de Contaminação (*Data Leakage*) contra 127 Benchmarks
Comparando todas as `758.255` expressões do corpus contra as equações-alvo de **Nguyen (12)**, **Feynman (101)** e **Strogatz (14)**:
- **Nguyen (12):** **0/12** ocorrências exatas ou com `C=1`! Mesmo por esqueleto abstrato (`V`, `C`), apenas 3 problemas têm casamento raro (`nguyen_1`: 6, `nguyen_9`: 3, `nguyen_10`: 5 em 758 mil = `< 0,001%`). **Nguyen-5 tem 0 ocorrências exatas e 0 por esqueleto.**
- **Strogatz (14):** **13/14** com zero ocorrências (apenas `strogatz_glider2`, `V1 - cos(V2)/V1`, aparece 2 vezes em 758 mil).
- **Feynman (101):** apenas **17** fórmulas têm casamento, **todas** monômios/razões elementares de 2 a 4 variáveis (`x_1*x_2`, `x_1/x_2`, `x_1*x_2*x_3`, `x_1*x_2*sin(x_3)`, `asin(x_1*sin(x_2))`, `sqrt(x_1*x_2/x_3)`, `x_1*x_2**2`). Nenhuma equação composta não-trivial de Feynman consta no corpus.

---

## 2. Treinamento SFT Real dos 6 Modelos (`adapter_config.json` + `trainer_state.json`)

Todos os 6 modelos publicados no HuggingFace compartilham **exatamente o mesmo LoRA** (`r = 8`, `lora_alpha = 32`, `lora_dropout = 0.05`, `target_modules = ["c_attn"]`, `fp16 = True`, vocabulário padrão `50.257`) e usaram `EarlyStoppingCallback(patience=3, threshold=0.001)` com avaliação a cada `500` passos:

| Modelo | Backbone | Batch por GPU × Accum = Efetivo | LR | Melhor Checkpoint (passo) | Fração de Época Vista | Exemplos Vistos até Melhor Ckpt | Melhor `eval_loss` | `train_loss` final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt2_base_infix_682k` | `gpt2` (124M) | `8 × 4 = 32` | `5e-5` | `11.500` | `0,54` | `368.000` (53,9%) | **0,2912** | `0,3047` |
| `gpt2_base_prefix_682k` | `gpt2` (124M) | `8 × 4 = 32` | `5e-5` | `10.000` | `0,47` | `320.000` (46,9%) | **0,3033** | `0,3228` |
| `gpt2_medium_infix_682k` | `gpt2-medium` (355M) | `4 × 4 = 16` | `3e-5` | `6.500` | `0,15` | `104.000` (15,2%) | **0,2913** | `0,3022` |
| `gpt2_medium_prefix_682k` | `gpt2-medium` (355M) | `4 × 4 = 16` | `3e-5` | `13.500` | `0,32` | `216.000` (31,7%) | **0,2944** | `0,3017` |
| `gpt2_large_infix_682k` | `gpt2-large` (774M) | `2 × 4 = 8` | `2e-5` | `9.500` | `0,11` | `76.000` (11,1%) | **0,2884** | `0,2928` |
| `gpt2_large_prefix_682k` | `gpt2-large` (774M) | `2 × 4 = 8` | `2e-5` | `10.000` | `0,12` | `80.000` (11,7%) | **0,2996** | `0,3060` |

### Achado Crítico sobre Escala (Base vs. Large)
Como `eval_steps = 500` foi mantido fixo enquanto o *batch* efetivo caiu de `32` (Base) para `16` (Medium) e `8` (Large) por limite de VRAM, o `EarlyStoppingCallback(threshold=0.001)` avaliou o modelo Large **4× mais frequentemente por amostra** e encerrou seu treino após ver apenas **76 mil exemplos (11,1% de uma época)**, contra **368 mil exemplos (53,9%)** no Base. Isso fornece uma explicação mecanística exata e verificável para o resultado empírico em que o `Base Infix` superou ou empatou com o `Large Infix` como prior de busca a uma fração do custo computacional.

---

## 3. Validação Local na RTX 3050 Pós-Correções do Núcleo v3

1. **Efeito da lista/ordem de operadores no prompt (`N = 256`, `gpt2_base_infix_682k`, sem `top_k`):**
   - Em `nguyen_1` (`[-1, 1]`):
     - `v3_current_10ops` (não ordenado): validade `29,7%` (`T=1.0`) / `48,8%` (`T=0.7`).
     - `sorted_10ops` (mesmos 10 ops em ordem alfabética): `34,8%` (`T=1.0`) / `51,2%` (`T=0.7`).
     - `sft_sorted_13ops` (os 13 ops ordenados do SFT): **`49,2%` (`T=1.0`) / `71,1%` (`T=0.7`)** — reduz `NaN` de `53,1%` para `31,2%` e cobre `asin`/`tan` exigidos em Feynman.
2. **Canário pós-correção em GPU (`nguyen_7`, 6 passos × 64 amostras):**
   - `bon`, `ppo`, `ppo+elite`, `ppo+random`, `rspg`, `rspg+elite`, `rspg+random`, `grpo+elite` passaram todos os *tripwires* (`sample_drift <= 3,2e-4`, `approx_kl >= 0`, replay respeitando teto e sem sequestrar o quantil do `rspg`).
3. **Custo real de `--const_mode fit` (L-BFGS-B):**
   - Em 10.000 linhas (`batch=128`, 2 passos, 254 expressões únicas): `15,23s` (`const_mode="one"`) vs. `17,91s` (`const_mode="fit"`) — acréscimo de apenas **~2,7s por 256 amostras (~10,7s por passo de 1.024)** em CPU local, elevando o melhor $R^2$ zero-shot de `0,513` para `0,909`.
