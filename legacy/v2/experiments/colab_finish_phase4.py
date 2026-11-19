# ============================================================================
# CÉLULA COLAB — fecha a Fase 4: roda os 11 problemas que faltam, salva no Drive.
# Cole isto numa célula do Colab (Runtime > Change runtime type > T4 GPU).
# Resumível: se a sessão cair, re-rode a célula — ela pula o que já está no Drive.
# ============================================================================

# 1) Clona o repo (codigo com os guards de OOM ja esta no origin/main)
!git clone -q https://github.com/augustocsc/seriguela.git
%cd /content/seriguela

# 2) Deps (usa o torch que ja vem no Colab; instala o stack pinado do projeto)
!pip install -q "transformers==4.51.3" "trl==0.16.1" "peft==0.15.1" "datasets==3.5.0" "sympy==1.13.1" pmlb

# 3) Monta o Drive — resultados PERSISTEM entre sessoes
from google.colab import drive
drive.mount('/content/drive')
import os
OUT = '/content/drive/MyDrive/seriguela_phase4'
os.makedirs(OUT, exist_ok=True)

# 4) Confirma a GPU
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else '*** SEM GPU: Runtime > Change runtime type > T4 GPU, e rode de novo ***')

# 5) Os 11 que faltam (nguyen_1 primeiro = ~5 min, vitoria rapida; depois 10 feynman)
PROBLEMS = ['nguyen_1',
    'feynman_I_12_4', 'feynman_I_18_4', 'feynman_I_29_4', 'feynman_I_32_17',
    'feynman_I_34_10', 'feynman_I_39_1', 'feynman_I_40_1', 'feynman_I_43_31',
    'feynman_I_47_23', 'feynman_I_8_14']

# 6) Roda cada um (campeao bon_ppo, C=1, 1 seed — mesma config da Fase 4).
import subprocess, sys, glob
for p in PROBLEMS:
    if glob.glob(f'{OUT}/{p}/bon_ppo/**/seed_42/results_latest.json', recursive=True):
        print(f'== {p}: ja feito, pulando =='); continue
    print(f'\n===== rodando {p} =====', flush=True)
    subprocess.run([sys.executable, '2_training/reinforcement/run_experiment.py',
        '--algorithm', 'bon_ppo', '--model', 'augustocsc/gpt2_base_infix_682k',
        '--problem', p, '--reward', 'r2_clipped', '--penalty', 'gradient',
        '--temperature', 'cosine_annealing', '--prompt_type', 'standard',
        '--batch_size', '1024', '--max_steps', '200', '--seeds', '42',
        '--buffer_ratio', '0.2', '--learning_rate', '1e-6', '--ppo_epochs', '2',
        '--max_kl', '0.02', '--patience', '100000', '--no_wandb',
        '--output_dir', f'{OUT}/{p}/bon_ppo/'])
print('\n=== LOTE CONCLUIDO — resultados em', OUT, '===')
