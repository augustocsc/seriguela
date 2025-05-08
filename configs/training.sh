CUDA_VISIBLE_DEVICES=0 python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
  --dataset_repo_id augustocsc/sintetico_final \
  --data_dir 100k \
  --output_dir ./output \
  --push_to_hub \
  --hub_model_id augustocsc/Se124M100KInfPrompt_endtoken_2 \
  --source_data_column i_prompt \
  --report_to wandb \
  --run_name Se124M100KInfPrompt_endtoken_2 \
  --fp16 \
  --load_best_model_at_end \
  --learning_rate 5e-4 \
  --lr_scheduler_type cosine \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules c_attn \
  --lora_bias none \
  --overwrite_output_dir \


# CUDA_VISIBLE_DEVICES=0 python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#   --dataset_repo_id augustocsc/sintetico_final \
#   --data_dir 100k \
#   --output_dir ./output \
#   --push_to_hub \
#   --hub_model_id augustocsc/Se124M100KInfPrompt_endtoken \
#   --early_stopping_patience 5 \
#   --source_data_column i_prompt \
#   --report_to wandb \
#   --run_name Se124M100KInfPrompt_endtoken \
#   --fp16 \
#   --load_best_model_at_end \
#   --eval_strategy epoch \
#   --num_train_epochs 50 \
#   --learning_rate 5e-4 \
#   --lr_scheduler_type cosine \
#   --warmup_steps 200 \





















# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#   --dataset_repo_id augustocsc/sintetico_final \
#   --data_dir 100k \
#   --output_dir ./output \
#   --push_to_hub \
#   --hub_model_id augustocsc/Se124M100KInfPrompt_endtoken \
#   --early_stopping_patience 2 \
#   --source_data_column i_prompt \
#   --report_to wandb \
#   --run_name Se124M100KInfPrompt_endtoken \
#   --fp16 \
#   --load_best_model_at_end \
#   --eval_strategy epoch \
#   --num_train_epochs 50

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#   --dataset_repo_id augustocsc/sintetico_final \
#   --data_dir 500k \
#   --output_dir ./output \
#   --push_to_hub \
#   --hub_model_id augustocsc/Se124M500KInfPrompt_endtoken \
#   --early_stopping_patience 2 \
#   --source_data_column i_prompt \
#   --report_to wandb \
#   --run_name Se124M500KInfPrompt_endtoken \
#   --fp16 \
#   --load_best_model_at_end \
#   --eval_strategy epoch \
#   --num_train_epochs 50

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#   --dataset_repo_id augustocsc/sintetico_final \
#   --data_dir 1M \
#   --output_dir ./output \
#   --push_to_hub \
#   --hub_model_id augustocsc/Se124M1MInfPrompt_endtoken \
#   --early_stopping_patience 2 \
#   --source_data_column i_prompt \
#   --report_to wandb \
#   --run_name Se124M1MInfPrompt_endtoken \
#   --fp16 \
#   --load_best_model_at_end \
#   --eval_strategy epoch \
#   --num_train_epochs 50

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 10k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M10KInfDelimiter \
#     --early_stopping_patience 2 \
#     --source_data_column i_delimiter \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M10KInfDelimiter \

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 10k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M10KInfMinimalist \
#     --early_stopping_patience 2 \
#     --source_data_column i_minimalist \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M10KInfMinimalist \


##################################

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 100k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M100KInfSimple \
#     --early_stopping_patience 2 \
#     --source_data_column i_simple \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M100KInfSimple \

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 100k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M100KInfKeyValue \
#     --early_stopping_patience 2 \
#     --source_data_column i_key_value \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M100KInfKeyValue \

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 100k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M100KInfDelimiter \
#     --early_stopping_patience 2 \
#     --source_data_column i_delimiter \
#     --load_best_model_at_end
#     --report_to wandb \
#     --run_name Se124M100KInfDelimiter \

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 100k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M100KInfMinimalist \
#     --early_stopping_patience 2 \
#     --source_data_column i_minimalist \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M100KInfMinimalist \


##################################


# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 500k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M500KInfSimple \
#     --early_stopping_patience 2 \
#     --source_data_column i_simple \
#     --report_to wandb \
#     --run_name Se124M500KInfSimple \
#     --fp16 \
#     --eval_strategy epoch \
#     --num_train_epochs 50

    
# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 500k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M500KInfKeyValue \
#     --early_stopping_patience 2 \
#     --source_data_column i_key_value \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M500KInfKeyValue \
#     --fp16 \
#     --eval_strategy epoch \
#     --num_train_epochs 50

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 500k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M500KInfDelimiter \
#     --early_stopping_patience 2 \
#     --source_data_column i_delimiter \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M500KInfDelimiter \
#     --fp16 \
#     --eval_strategy epoch \
#     --num_train_epochs 50

# python /home/augusto/symbo_repos/seringuela/scripts/train_test.py \
#     --dataset_repo_id augustocsc/sintetico \
#     --data_dir 500k \
#     --output_dir ./output \
#     --push_to_hub \
#     --load_best_model_at_end \
#     --hub_model_id augustocsc/Se124M500KInfMinimalist \
#     --early_stopping_patience 2 \
#     --source_data_column i_minimalist \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --run_name Se124M500KInfMinimalist \
#         --fp16 \
#     --eval_strategy epoch \
#     --num_train_epochs 50


# ##################################







