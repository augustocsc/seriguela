#!/bin/bash
# Fecha as 2 celulas bon_grpo inviaveis: roda os 8 seeds restantes com os dois
# guards (complexity cap + anti-hang) ja no codigo. Um seed por vez, timeout
# de 2h cada (nenhum hang/lentidao segura o pod), seed-skip (nao re-roda o que
# ja existe). Grava direto no caminho real de results/phase_2.
set -u
cd /root/seriguela

run_seed () {
  local prob=$1 seed=$2
  local dst="results/phase_2/base_infix/${prob}/bon_grpo/bon_grpo_gpt2_base_infix_682k/${prob}/cosine_annealing/seed_${seed}"
  if [ -f "${dst}/results_latest.json" ]; then
    echo "[skip] ${prob} seed ${seed} ja existe"; return 0
  fi
  echo "[run ] ${prob} seed ${seed} (timeout 2h)..."
  NO_GIT=1 timeout 7200 .venv/bin/python 2_training/reinforcement/run_experiment.py \
    --algorithm bon_grpo --buffer_ratio 0.2 --problem "${prob}" --max_steps 200 \
    --batch_size 1024 --reward r2_clipped --penalty gradient \
    --temperature cosine_annealing --prompt_type standard --patience 100000 \
    --seeds "${seed}" --no_wandb \
    --output_dir "results/phase_2/base_infix/${prob}/bon_grpo"
  echo "[done] ${prob} seed ${seed} exit=$?"
}

# duas passadas: a 2a recupera qualquer seed que tenha caido por timeout transitorio
for pass in 1 2; do
  echo "===== passada ${pass} ====="
  for s in 123 456 789 1011; do run_seed nguyen_3 "$s"; done
  for s in 123 456 789 1011; do run_seed nguyen_7 "$s"; done
done

echo "===== agregando as 2 celulas ====="
for prob in nguyen_3 nguyen_7; do
  NO_GIT=1 .venv/bin/python 2_training/reinforcement/run_experiment.py \
    --algorithm bon_grpo --buffer_ratio 0.2 --problem "${prob}" --max_steps 200 \
    --batch_size 1024 --reward r2_clipped --penalty gradient \
    --temperature cosine_annealing --prompt_type standard --patience 100000 \
    --seeds 42 123 456 789 1011 --no_wandb \
    --output_dir "results/phase_2/base_infix/${prob}/bon_grpo" 2>&1 | tail -2
done
echo "BON_GRPO_8_FIM"
