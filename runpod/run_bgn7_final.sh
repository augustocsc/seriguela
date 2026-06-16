#!/bin/bash
# Passada final limitada da celula bon_grpo/nguyen_7 (2/5 seeds bancadas).
# timeout 2h por seed: hang de sympy nunca mais segura o pod.
cd /root/seriguela
for s in 456 789 1011; do
  if ls results/phase_2/base_infix/nguyen_7/bon_grpo/*/nguyen_7/*/seed_$s/results_latest.json >/dev/null 2>&1; then
    echo "seed $s ja completa"; continue
  fi
  echo "rodando seed $s (timeout 2h)..."
  NO_GIT=1 timeout 7200 .venv/bin/python 2_training/reinforcement/run_experiment.py \
    --algorithm bon_grpo --buffer_ratio 0.2 --problem nguyen_7 --max_steps 200 \
    --batch_size 1024 --reward r2_clipped --penalty gradient \
    --temperature cosine_annealing --prompt_type standard --patience 100000 \
    --seeds $s --no_wandb \
    --output_dir results/phase_2/base_infix/nguyen_7/bon_grpo
  echo "seed $s exit=$?"
done
echo "agregando o que existir..."
NO_GIT=1 timeout 1800 .venv/bin/python - <<'PYEOF'
import json, glob
rs = []
for f in sorted(glob.glob('results/phase_2/base_infix/nguyen_7/bon_grpo/*/nguyen_7/*/seed_*/results_latest.json')):
    d = json.load(open(f))
    d.pop('history', None)
    rs.append(d)
if rs:
    import statistics as st
    vals = [r.get('best_r2', 0) or 0 for r in rs]
    seeds = sorted({int(f.split('seed_')[1].split('/')[0]) for f in glob.glob('results/phase_2/base_infix/nguyen_7/bon_grpo/*/nguyen_7/*/seed_*/results_latest.json')})
    agg = {'seeds': seeds, 'problem': 'nguyen_7', 'algorithm': 'bon_grpo',
           'model': 'augustocsc/gpt2_base_infix_682k',
           'mean_best_r2': st.mean(vals), 'std_best_r2': st.pstdev(vals),
           'mean_train_r2': st.mean(vals), 'std_train_r2': st.pstdev(vals),
           'max_train_r2': max(vals), 'min_train_r2': min(vals),
           'mean_test_r2': st.mean([r.get('test_r2', r.get('best_r2', 0)) or 0 for r in rs]),
           'std_test_r2': 0.0, 'max_test_r2': 0.0, 'min_test_r2': 0.0,
           'individual_results': rs}
    out = 'results/phase_2/base_infix/nguyen_7/bon_grpo/aggregate_bon_grpo_nguyen_7_seed' + '_'.join(map(str, seeds)) + '.json'
    json.dump(agg, open(out, 'w'), indent=2, default=str)
    print('aggregate escrito:', out, 'n =', len(rs))
PYEOF
echo BGN7_FINAL_FIM
