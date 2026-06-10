# RunPod — compute oficial do projeto (a partir de 2026-06)

Substitui Colab (créditos esgotados) e AWS (custo). Decisão de 2026-06-09:
**community cloud, A40 48GB** (~US$0,40–0,47/h), fallback A6000/3090/4090/L40S/A5000/L4.

## Pré-requisitos (uma vez)

1. Conta RunPod com créditos.
2. Chave de API com escopo **Read & Write** (Settings → API Keys).
3. Adicionar a linha abaixo em `~/.tokens.txt` (nunca commitar):
   ```
   runpod = <sua-chave>
   ```
4. `ssh` e `scp` disponíveis no PATH (Windows: OpenSSH embutido serve).

## Fluxo 1 — Piloto de timing (Fase 1.5: plateau scout + custo real)

```bash
python runpod/launch_pilot.py            # cria/reusa pod, bootstrap, smoke test, inicia piloto
python runpod/launch_pilot.py --status   # acompanha (markers, logs, timing_report.md)
# ao final:
scp -i ~/.ssh/runpod_pilot -P <port> root@<ip>:seriguela/results/pilot_timing/timing_report.* results/pilot_timing/
python runpod/launch_pilot.py --terminate   # SEMPRE terminar o pod
```

O piloto (`experiments/run_pilot_timing.py`) mede s/step solo e com 2x/3x runs
concorrentes, throughput do best_of_n e um run Feynman — e projeta horas/custo
das Fases 2 e 4.2. Os runs de 500 steps dobram como **scout de plateau**: o
`MAX_STEPS` da Fase 2 sai do `timing_report.md`.

- O pod **não** se auto-termina: `--terminate` é manual e obrigatório.
- Idempotente: guard por nome `augusto-seriguela-pilot` — rodar de novo reusa o pod.
- O bootstrap roda `experiments/test_smoke.py` **no pod** antes de qualquer GPU-hora útil.

## Fluxo 2 — Fase 2 (RL principal, 100 seed-runs)

1. Gerar a fila (depois de escolher `MAX_STEPS` com o piloto):
   ```bash
   python experiments/build_phase2_queue.py --max-steps 200 --write
   python experiments/test_smoke.py        # valida schema da fila
   git add experiments/queue.yaml && git commit -m "queue: phase 2" && git push
   ```
2. Subir um pod (reusa o launcher; o bootstrap clona o repo já com a fila):
   ```bash
   python runpod/launch_pilot.py --status   # se não houver pod, rode sem flags p/ criar
   ```
3. No pod, rodar o daemon da fila em modo VM (sem git push do pod):
   ```bash
   ssh -i ~/.ssh/runpod_pilot -p <port> root@<ip>
   cd seriguela && source .venv/bin/activate
   NO_GIT=1 nohup python -c "from experiments.queue_processor import run_queue_loop; run_queue_loop(max_hours=24)" > queue.log 2>&1 &
   ```
4. Coletar resultados periodicamente (do laptop):
   ```bash
   scp -r -i ~/.ssh/runpod_pilot -P <port> root@<ip>:seriguela/results/phase_2 results/
   ```
5. Terminar o pod ao final. Commitar os resultados localmente.

## Orçamento (estimativas pré-piloto — refinar com timing_report)

| Item | Estimativa |
|---|---|
| Piloto completo | ~3–5h A40 ≈ US$1,5–2,5 |
| Fase 2 (100 seed-runs × 200 steps) | ~30–50h A40 ≈ US$15–25 |
| Fase 4.2 (RL na Categoria A) | medido pelo probe Feynman do piloto |

## Segurança

- A chave nunca é impressa nem commitada; `runpod/pod_info.json` (IP/porta) é gitignored.
- Convenção de nomes: pods do projeto começam com `augusto-` (mesma regra da era AWS).
