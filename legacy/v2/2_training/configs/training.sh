CUDA_VISIBLE_DEVICES=0 python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
  --dataset_repo_id augustocsc/sintetico_natural \
  --data_dir 500k \
  --output_dir ./output \
  --push_to_hub \
  --hub_model_id augustocsc/Se124M500KInfPrompt_EOS \
  --source_data_column i_prompt \
  --report_to wandb \
  --run_name Se124M500KInfPrompt_EOS \
  --model_name_or_path gpt2 \
  --bf16 \
  --eval_strategy steps \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 8 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --logging_steps 20 \
  --eval_steps 500 \
  --save_steps 1000 \
  --save_total_limit 3 \


# CUDA_VISIBLE_DEVICES=1 python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#   --dataset_repo_id augustocsc/sintetico_final \
#   --data_dir 100k \
#   --output_dir ./output \
#   --push_to_hub \
#   --hub_model_id augustocsc/Se124M100KInfPrompt_NT \
#   --source_data_column i_prompt \
#   --report_to wandb \
#   --run_name Se124M100KInfPrompt_NT \
#   --bf16 \
#   --eval_strategy steps \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --gradient_accumulation_steps 2 \
#   --dataloader_num_workers 8 \
#   --learning_rate 2e-5 \
#   --warmup_ratio 0.03 \
#   --weight_decay 0.01 \
#   --max_grad_norm 1.0 \
#   --lr_scheduler_type cosine \
#   --optim adamw_torch_fused \
#   --logging_steps 20 \
#   --eval_steps 500 \
#   --save_steps 1000 \
#   --save_total_limit 3

# CUDA_VISIBLE_DEVICES=0 python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#   --dataset_repo_id augustocsc/sintetico_final \
#   --data_dir 100k \
#   --output_dir ./output \
#   --push_to_hub \
#   --hub_model_id augustocsc/Se124M100KInfPrompt_WT \
#   --source_data_column i_prompt \
#   --report_to wandb \
#   --run_name Se124M100KInfPrompt_WT \
#   --bf16 \
#   --eval_strategy steps \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --gradient_accumulation_steps 2 \
#   --dataloader_num_workers 8 \
#   --learning_rate 2e-5 \
#   --warmup_ratio 0.03 \
#   --weight_decay 0.01 \
#   --max_grad_norm 1.0 \
#   --lr_scheduler_type cosine \
#   --optim adamw_torch_fused \
#   --logging_steps 20 \
#   --eval_steps 500 \
#   --save_steps 1000 \
#   --save_total_limit 3
