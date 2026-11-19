#!/bin/bash
# Runner overnight da Fase 4 na RTX 3050 6GB local (venv C:, SAC permite C:).
# Campeão bon_ppo, batch 1024 (protocolo VÁLIDO), seed 42, problemas nguyen
# ainda não cobertos pela Fase 2. Sequencial + resumível (seed-skip do
# run_experiment) + expandable_segments pra aliviar o limite de VRAM (~97%).
# Saída em results/phase_4_local/ (dados; SAC não bloqueia dados).
set -u
cd "C:/Users/Rudá/seriguela"
export HF_HOME=D:\\seriguela_local2\\hf_cache
export TMP=D:\\seriguela_local2\\tmp TEMP=D:\\seriguela_local2\\tmp
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="C:/Users/Rudá/sgtest_venv/Scripts/python.exe"
OUT="results/phase_4_local"
LOG="C:/Users/Rudá/phase4_local.log"

# nguyen fora da Fase 2 (3,5,7,9 já feitos); 1 excluído (sem sinal)
PROBLEMS="nguyen_2 nguyen_4 nguyen_6 nguyen_8 nguyen_10 nguyen_11 nguyen_12"

echo "=== OVERNIGHT LOCAL START $(date) ===" >> "$LOG"
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
  rc=$?
  echo "[$(date '+%H:%M')] <<< END $prob rc=$rc" >> "$LOG"
done
echo "=== OVERNIGHT LOCAL DONE $(date) ===" >> "$LOG"
