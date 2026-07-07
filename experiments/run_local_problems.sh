#!/bin/bash
# Runner local genérico (RTX 3050, venv C:) — roda o campeão bon_ppo numa LISTA
# de problemas passada por argumento, output em results/phase_4_local/ (que a
# tabela-mãe já lê). Resumível (seed-skip), expandable_segments p/ a VRAM de 6GB.
# Coordena com o RunPod marcando estes problemas como 'skipped' na fila do pod.
#
# Uso:  bash experiments/run_local_problems.sh feynman_I_12_1 feynman_I_13_4 ...
set -u
cd "C:/Users/Rudá/seriguela"
export HF_HOME=D:\\seriguela_local2\\hf_cache
export TMP=D:\\seriguela_local2\\tmp TEMP=D:\\seriguela_local2\\tmp
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="C:/Users/Rudá/sgtest_venv/Scripts/python.exe"
OUT="results/phase_4_local"
LOG="C:/Users/Rudá/phase4_local_feynman.log"

PROBLEMS="$@"
echo "=== LOCAL PROBLEMS START $(date) — $PROBLEMS ===" >> "$LOG"
for prob in $PROBLEMS; do
  echo "[$(date '+%H:%M')] >>> START $prob" >> "$LOG"
  "$PY" -u 2_training/reinforcement/run_experiment.py \
    --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k \
    --problem "$prob" --reward r2_clipped --penalty gradient \
    --temperature cosine_annealing --prompt_type standard \
    --batch_size 1024 --max_steps 200 --seeds 42 \
    --buffer_ratio 0.2 --learning_rate 1e-6 --ppo_epochs 2 --max_kl 0.02 \
    --patience 100000 --no_wandb \
    --output_dir "$OUT/$prob/bon_ppo/" >> "$LOG" 2>&1
  echo "[$(date '+%H:%M')] <<< END $prob rc=$?" >> "$LOG"
done
echo "=== LOCAL PROBLEMS DONE $(date) ===" >> "$LOG"
