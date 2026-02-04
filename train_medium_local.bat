@echo off
REM Script para treinar GPT-2 Medium localmente (Windows)

echo Treinando GPT-2 Medium (355M parametros)...
echo AVISO: Requer ~16GB VRAM (GPU)
echo.

python scripts\train_medium.py ^
  --model_size gpt2-medium ^
  --dataset_repo augustocsc/sintetico_natural ^
  --data_dir 700K ^
  --data_column i_prompt_n ^
  --output_dir ./output/gpt2_medium_700K_json ^
  --num_train_epochs 3 ^
  --per_device_train_batch_size 2 ^
  --learning_rate 5e-5

echo.
echo Treinamento concluido!
pause
