# Log de Progresso — Seriguela v3

> Registro vivo das execuções, decisões experimentais, consumo de Compute Units (Colab) e próximos passos no branch `claude/parecer-academico-analise-d028ed`.
> Plano mestre: [PLANO_V3_2026-11.md](PLANO_V3_2026-11.md) | Auditoria SFT/Dados: [SFT_AND_DATASET_AUDIT_2026-11.md](SFT_AND_DATASET_AUDIT_2026-11.md)

---

## 1. Linha do tempo e orçamento (Colab CUs)

Saldo inicial: **`600.00 CU`** (L4 medida a **`2.33 CU/h`**).

| Data | Etapa | Commit | Status | Custo (CU) | Saldo final (CU) | Observações |
|---|---|---|---|---|---|---|
| 2026-11-25/26 | Correções v3 + auditoria SFT/dados + G1 timing L4 | `db9ecbb`–`4d4cdd6` | Concluído | `4.10` | `595.90` | 132/132 testes locais + G0/G1 na L4; `update_mb=32`, `srbench_truth` (121 Feynman + 14 Strogatz), `OPERATORS` alinhado ao SFT |
| 2026-11-27 | `g2a_r2` (parte 1: 4/12 antes de queda de websocket) | `4d4cdd6` | Interrompido | `4.77` | `591.13` | Motivou refatoração de `seriguela/colab.py` |
| 2026-11-27 | `g2a_r2` (execução desacoplada 8/12 + retomada 12/12) | `d73c6e2` → `0ce25ca` | **12/12 concluído** | `16.61` | `574.52` | Zero falhas de tripwire; sync incremental funcionou em todas as quedas de websocket |
| 2026-11-27 | `g2a_nrmse` (12/12 numa sessão) | `0ce25ca` | **12/12 concluído** | `11.76` | `562.76` | 5,08h de L4, 0 falhas |
| 2026-11-27 | `g2b_lr3e5` (`lr=3e-5`, `nrmse`, 8 runs) | `c947922` | **Em curso (6/8 salvos)** | teto `14.00` | — | Faltam apenas `nguyen_9/rspg+elite` (seeds `42, 123`), ETA ~1,2h |

---

## 2. Portões e decisões registradas

### Infraestrutura (`seriguela/colab.py`)
- Execução desacoplada em background (`start_new_session=True`), *polling* curto a cada 45s (absorve quedas de websocket após ~2h sem perder o processo da GPU), download incremental de `/content/results_v3.tgz` a cada run concluído e retomada automática (`colab upload` de `seed_*.json`).
- Suite offline atual: **136/136 testes `pytest` passando**.

### G2-A — Comparação `reward=r2` vs `reward=nrmse` (24/24 runs válidos)
Campanhas: [g2a_r2.json](../../campaigns/g2a_r2.json) vs [g2a_nrmse.json](../../campaigns/g2a_nrmse.json) (`steps=100`, `batch=1024`, `lr=5e-6`, `const_mode=one`, seeds `42, 123`).

| Problema | Braço | `test_r2` (`r2` / `nrmse`, mediana) | `zero_reward_frac` fim (`r2` → `nrmse`) | `mean_reward` fim (`nrmse`) | `kl_from_prior` (`nrmse`) | Match simbólico |
|---|---|---|---|---|---|---|
| `nguyen_5` | `bon` | 0.9757 / 0.9757 | 0.998 → 0.192 | `0.0727` | `0.0000` | `0/2` / `0/2` |
| `nguyen_5` | `ppo+elite` | 0.8965 / 0.8965 | 0.997 → **0.128** | **`0.1031`** | `0.0358` | `0/2` / `0/2` |
| `nguyen_5` | `rspg+elite` | 0.9473 / **0.9757** | 0.999 → 0.169 | `0.0794` | `0.0036` | `0/2` / `0/2` |
| `nguyen_9` | `bon` | 0.9844 / 0.9844 | 0.869 → 0.026 | `0.3145` | `0.0000` | `0/2` / `0/2` |
| `nguyen_9` | `ppo+elite` | 0.9828 / **0.9868** | 0.786 → **0.019** | **`0.3806`** | `0.0244` | `0/2` / `0/2` |
| `nguyen_9` | `rspg+elite` | **0.9880** / 0.9832 | 0.839 → 0.026 | `0.3369` | `0.0031` | `0/2` / `0/2` |

- **Decisão 6 fechada:** **`nrmse` adotado**. `r2` zerava 99,6%–100% do batch em `nguyen_5`; `nrmse` reduziu `zero_reward_frac` para `12,8%` e aumentou `mean_reward` em `+21%` a `+42%` sobre `bon`.
- **Gargalo identificado:** `lr=5e-6` moveu pouco a política em 100 passos (`kl ≈ 0.025–0.036` no PPO e `~0.003` no RSPG).

### G2-B — Calibração de `lr=3e-5` com `nrmse` ([g2b_lr3e5.json](../../campaigns/g2b_lr3e5.json), parcial 6/8)

| Problema | Braço | Seed | `test_r2` | `symbolic_match` | `zero_reward_frac` | `mean_reward` | `kl_from_prior` | Tempo | Expressão |
|---|---|---|---|---|---|---|---|---|---|
| `nguyen_5` | `ppo+elite` | `42` | **1.0000** | **True** | 0.546 → 0.001 | 0.0457 → 0.6023 | 0.8227 | 95.3m | `C*sin(x_1*x_1)**C*cos(x_1) - C` |
| `nguyen_5` | `ppo+elite` | `123` | **1.0000** | **True** | 0.530 → 0.008 | 0.0401 → 0.6079 | 1.0323 | 95.0m | `sin(x_1*x_1)*cos(x_1) - C` |
| `nguyen_5` | `rspg+elite` | `42` | 0.9764 | False | 0.546 → 0.039 | 0.0457 → 0.2383 | 0.2616 | 26.3m | `x_1*x_1*cos(x_1) - C` |
| `nguyen_5` | `rspg+elite` | `123` | **0.9923** | False | 0.530 → 0.046 | 0.0401 → 0.2352 | 0.3245 | 26.7m | `x_1*sin(x_1*cos(x_1)) - C` |
| `nguyen_9` | `ppo+elite` | `42` | 0.9747 | False | 0.298 → 0.000 | 0.2003 → 0.8134 | 1.0554 | 96.4m | `x_1**C + C*x_2*sin(sin(x_2))` |
| `nguyen_9` | `ppo+elite` | `123` | **1.0000** | **True** | 0.286 → 0.001 | 0.1963 → 0.8231 | 0.9385 | 97.1m | `sin(x_2**C*x_2) + sin(x_1)` |
| `nguyen_9` | `rspg+elite` | `42, 123` | — | — | — | — | — | — | *em execução na Colab L4* |

- **Leitura parcial de G2-B:** `lr=3e-5` desbloqueou o aprendizado de RL (**3/4 matches simbólicos exatos no `ppo+elite`** vs `0/4` no `bon` e `0/4` em `lr=5e-6`). `kl_from_prior ≈ 0.82–1.06` no PPO sugere avaliar se `lr=1.5e-5` ou `2e-5` reduz convergência prematura (como em `nguyen_9/seed_42`) mantendo a descoberta exata, antes de congelar o protocolo do G3.

---

## 3. Próximos passos
1. Aguardar conclusão dos últimos 2/8 runs de `g2b_lr3e5` (`nguyen_9/rspg+elite`) e atualizar este log com o fechamento 8/8 + saldo exato de CUs.
2. Avaliar necessidade de ponto intermediário (`lr=2e-5` ou `steps=100` vs `150`) / fechar G2 no painel completo de desenvolvimento (Nguyen) com `nrmse` e `lr` calibrado.
3. Congelar commit do protocolo → dimensionar `M` via `seriguela.analysis.required_problems()` → executar **G3 confirmatório** (Feynman/Strogatz estratificado, `>=5` seeds pareadas, `--const_mode fit`) → executar `seriguela.analysis` (`H1`–`H5`) e **G4** (SRBench 130 + PySR CPU).
4. Finalizar Capítulos 3–5 da dissertação e fazer merge de `claude/parecer-academico-analise-d028ed` em `main`.
