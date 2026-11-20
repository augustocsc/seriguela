# CLAUDE.md

Orientação para agentes (Claude Code e outros) que trabalham neste repositório.

## O projeto

**Seriguela** é uma dissertação de mestrado. Um GPT-2 Base (124M) ajustado com LoRA
em 682 mil expressões sintéticas serve de *prior* para uma busca de expressões por
problema: a cada passo o modelo amostra expressões, elas são avaliadas nos dados de
treino, e um algoritmo de RL atualiza a política. O prompt não contém os dados; eles
entram só pela recompensa.

**Fonte de verdade do desenho experimental:** [docs/reports/PLANO_V3_2026-11.md](docs/reports/PLANO_V3_2026-11.md).
O `THESIS_PLAN.md` descreve a v2 e fica como histórico.

## Estado (novembro de 2026)

- **v2 está encerrada e não é evidência para a defesa.** Um parecer externo e a
  conferência do código acharam falhas que invalidam as comparações (seção
  "Por que a v2 foi refeita" abaixo). Código em `legacy/v2/`, resultados em
  `results/v2/`, tag git `v2-final` para rodar o código antigo.
- **v3 é o núcleo novo**, em `seriguela/`, com testes em `tests/`. Ainda não há
  resultados v3: a campanha segue os portões G0–G4 do plano.
- Modelos do SFT no HuggingFace (`augustocsc/gpt2_{base,medium,large}_{infix,prefix}_682k`)
  continuam válidos. Padrão da tese: `augustocsc/gpt2_base_infix_682k`.

## Estrutura

```
seriguela/            núcleo v3 (pacote Python)
  expr.py             parse seguro, avaliação numpy, constantes C, avaliador de referência
  data.py             problemas: Nguyen (canônico) e SRBench/PMLB; treino/teste
  reward.py           Evaluator: expressão -> R², recompensa (com cache)
  policy.py           Policy: amostrar e pontuar por um único caminho (float32)
  advantages.py       z-score, postos, busca de risco
  buffer.py           replay de elite e replay aleatório (controle)
  trainer.py          laço único para todos os braços + tripwires + esquema do resultado
  symbolic.py         equivalência simbólica no critério do SRBench
  provenance.py       commit, versões, hashes
  run.py              linha de comando (um run por chamada)
  campaign.py         fila de runs de um arquivo, vários por GPU, teto de horas
  pod.py              RunPod: subir, acompanhar, buscar resultados, desligar
  canary.py, collect.py  portão G1 e tabela da campanha
campaigns/            arquivos de campanha (JSON)
tests/                pytest; modelo minúsculo aleatório, sem GPU e sem download
data/benchmarks/      Nguyen, Strogatz, metadados do Feynman (CSVs do Feynman sob demanda)
results/v2/           resultados antigos (legado)   results/v3/  resultados novos
dissertation/         LaTeX da dissertação (texto ainda da v2: será reescrito)
docs/reports/         plano v3, post-mortems, consolidações
legacy/               eras encerradas (v2, phase_a, colab, aws, dissertation_v1)
.github/workflows/    CI: pytest em CPU a cada push
```

## Comandos

```bash
pytest                                   # obrigatório antes de qualquer commit; ~1 min, CPU
python -m seriguela.run --problem nguyen_5 --algo ppo --replay elite --seed 42
python -m seriguela.run --help           # todo campo de RunConfig vira uma opção
```

Braços (`--algo`): `bon` (amostragem, sem atualização), `ppo`, `grpo`, `rank_pg`,
`rspg`. Replay (`--replay`): `none`, `elite`, `random`. Todos consomem exatamente
`steps * batch` amostras novas.

Teste toy na GPU local (RTX 3050 6GB), só para verificar que o código roda:

```bash
python -m seriguela.run --problem nguyen_1 --algo ppo --replay elite --steps 12 --batch 128 --update_mb 32 --allow_dirty --out /tmp/toy
```

Campanha (arquivo em `campaigns/`, commitado e com push antes de subir o pod):

```bash
python -m seriguela.campaign plan campaigns/g1_timing.json      # lista os runs
python -m seriguela.pod up campaigns/g1_timing.json --budget 2  # teto em US$
python -m seriguela.pod watch                                   # acompanha, busca, desliga
python -m seriguela.pod down                                    # apaga o pod na mão
```

No pod, o job roda G0 (pytest) e G1 (canário) antes da campanha, tudo sob `timeout`,
e o pod se desliga sozinho ao fim; se ninguém buscar os resultados em 45 min, ele
para em vez de apagar e o volume `/workspace` fica guardado.

## Regras do experimento (não negociáveis)

1. **Nada de GPU paga sem o portão anterior verde.** G0 = `pytest` verde no commit.
2. **Um run de campanha exige árvore git limpa.** `--allow_dirty` é só para testes.
3. **Não desligue um tripwire para "fazer rodar".** Se dispara, há um bug: conserte
   e acrescente o teste que o prova.
4. **O teste só é lido em `trainer._finalize`**, uma vez, para a expressão escolhida
   pelo treino. Nenhuma seleção, ajuste ou parada antecipada pode olhar para o teste.
5. **Mesmo orçamento e mesma política de amostragem em todos os braços**, baseline incluído.
6. **≥ 5 seeds por célula**; a unidade de generalização é o problema, não a seed.
7. **Ajuste de hiperparâmetros só no painel de desenvolvimento** (Nguyen), com o
   mesmo número de tentativas por braço.
8. Resultado é o JSON validado por `trainer.validate` (esquema v3). Ausência é
   ausência: nunca preencher com outro arquivo nem maximizar entre arquivos.

## Por que a v2 foi refeita

Achados conferidos no código (detalhes e evidências no plano v3):

- **Treino em FP16 puro**: pesos, gradientes e estado do Adam em meia precisão. O
  segundo momento do Adam zerava por underflow; o otimizador não era Adam.
- **Amostragem ≠ atualização**: `generate` aplicava top-k=50, as elites eram
  pontuadas sem temperatura, o dropout do LoRA ficava ligado no update.
- **Baseline diferente dos braços RL**: outro prompt, top-p=0,9, temperatura fixa,
  100 tokens contra 50, e metade do orçamento em um problema.
- **Recompensa com zona morta**: com a penalidade "gradient", R² em [0; 0,5) valia 0.
- **"GRPO" não era GRPO**: postos em grupos arbitrários de 8, sem razão nem recorte.
- **Bug `x**C` -> `x**2`** no avaliador, corrigido só depois de treinar.
- **Comparação com o SRBench com critérios trocados** e sem os 19 problemas mais difíceis.

## Credenciais

`~/.tokens.txt` (nunca imprimir nem commitar): `huggingface`, `wandb`, `runpod`.

## Máquina local e computação

- Local: i5-12450HX, 16 GB, RTX 3050 6 GB. Serve para `pytest` e testes toy.
  O Python do sistema (3.13, torch 2.11+cu128, transformers 5.8) roda o núcleo v3.
- Campanha: RunPod, RTX 3090 da comunidade (≈ US$ 0,22/h em junho de 2026), vários
  runs por placa. Orquestrador: `seriguela/pod.py` + `seriguela/campaign.py`. AWS foi
  descartada (sem créditos; mais cara por hora).
- Versões fixadas para CI e pod: `requirements.txt` (transformers 4.51.3, peft 0.15.1).

**Última atualização:** 2026-11-19
