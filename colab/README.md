# Colab Pipeline — Guia de Uso

Este diretório contém os notebooks para rodar experimentos do Seriguela no Google Colab de forma autônoma.

## Arquivos

| Arquivo | Função |
|---------|--------|
| `seriguela_runner.ipynb` | **Notebook principal.** Daemon que processa a fila de experimentos automaticamente. Abre 1x/dia, clica "Run All", fecha. |
| `ssh_bootstrap.ipynb` | Debug remoto via SSH. Usar quando algo quebra e precisa inspecionar o ambiente. |

## Setup Inicial (Fazer UMA VEZ)

### 1. Adicionar secrets no Colab

No Google Colab, clique no ícone 🔑 (cadeado) na barra lateral esquerda e adicione:

| Secret | Onde obter |
|--------|-----------|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY` | [wandb.ai/settings](https://wandb.ai/settings) → API keys |
| `GH_PAT` | [github.com/settings/tokens/new](https://github.com/settings/tokens/new) → escopo: `repo`, expiração: 90 dias |

### 2. Verificar Runtime

Para usar o runner:
- **T4** (economiza units): Modelos Base (124M) cabem bem. Fase 2 com Base-Infix.
- **L4** (mais rápido): Modelos Medium/Large, ou se a fila for longa. Custa ~2.5x mais units.
- **T4 Free**: Apenas para Fase 1 piloto zero-shot (inferência pura, zero units Pro).

## Uso Diário

### Fluxo normal (5 min/dia)

1. Abra `seriguela_runner.ipynb` no Colab
2. Clique **"Runtime" → "Run All"** (ou Ctrl+F9)
3. Aguarde as célula 1-5 passarem (~2 min)
4. A célula 6 roda por até 11h automaticamente
5. Feche o browser — o Colab continua rodando

### Como Claude adiciona novos experimentos

Claude edita `experiments/queue.yaml` localmente e faz `git push`. O daemon no Colab faz `git pull` a cada 60 segundos e pega automaticamente os novos jobs.

Você **não precisa fazer nada** para que novos experimentos apareçam na fila. Apenas mantenha o daemon rodando.

### Verificar progresso

Via W&B: [wandb.ai/augustocsc/seriguela](https://wandb.ai/augustocsc/seriguela)

Via heartbeat (verifica se o daemon está vivo):
```python
# No Colab ou localmente:
import json
from pathlib import Path

# Heartbeat no repo
h = json.loads(Path('experiments/heartbeat.json').read_text())
print(f"Último update: {h['timestamp']}")
print(f"Experimento atual: {h['current_experiment']}")
print(f"Done nesta sessão: {h['experiments_done_this_session']}")
```

Via git log:
```bash
git log --oneline -5  # commits de [queue] mostram progresso
```

## Debug Remoto (quando algo quebra)

1. Abra `colab/ssh_bootstrap.ipynb` **em uma nova aba** do Colab (pode ser no mesmo runtime ou num diferente)
2. Execute a célula — ela exibe um endpoint `tcp://xxx.trycloudflare.com:PORTA` e uma senha
3. Envie ao Claude no chat: "O endpoint é tcp://xxx.trycloudflare.com:PORTA com senha SENHA"
4. Claude se conecta e pode inspecionar logs, processos, editar arquivos

## Arquitetura

```
Local (Claude + você)
    │  edita queue.yaml
    │  git push
    ▼
GitHub (fonte de verdade)
    │  git pull (a cada 60s)
    ▼
Colab Runner (seriguela_runner.ipynb)
    │  executa experimentos
    │  salva resultados no Drive
    │  git commit + push resultados
    ▼
Drive + W&B (resultados)
    │  git pull
    ▼
Local (análise dos resultados)
```

## Troubleshooting

| Problema | Solução |
|----------|---------|
| "GH_PAT not found" | Adicionar `GH_PAT` nos Colab Secrets |
| "CUDA out of memory" | O daemon reduz `batch_size` pela metade e tenta de novo. Se persistir, trocar para L4. |
| Daemon parou sem commitar | Checar `experiments/heartbeat.json` — se >2h atrás, reiniciar o notebook |
| "Lock file found" | Deletar `experiments/_running.lock` e reiniciar |
| Drive cheio | O heartbeat avisa. Comprimir resultados antigos: `tar -czf archive_phase1.tar.gz results/phase_1/` |

## Estimativa de Compute Units (Colab Pro)

| Fase | GPU | Runs | Tempo est. | Units |
|------|-----|------|-----------|-------|
| 1 — Piloto | T4 Free | 18 | ~5h | 0 |
| 1.5 — Scout | T4 Pro | 6 | ~3h | ~6 |
| 2 — RL Base | T4 Pro | 100 | ~50h | ~100 |
| 4.1 — Feynman zero-shot | L4 | 226 | ~19h | ~92 |
| 4.2 — RL Categoria A | L4 | ~120 | ~60-135h | ~290-650 |

Colab Pro: 100 units/mês ≈ $10/mês.
