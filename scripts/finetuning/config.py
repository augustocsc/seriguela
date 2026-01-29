# finetuning/config.py

# --- Constants ---
SPECIAL_TOKENS_DICT = {
    "eos_token": "<endofex>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<startofex>"]
}
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<endofex>"
START_OF_EX_TOKEN = "<startofex>" # Explicit constant for clarity if needed elsewhere

DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LOGGING_STEPS = 100
DEFAULT_SAVE_EVAL_STEPS = 500
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_SEED = 42
DEFAULT_EVAL_STRATEGY = "epoch"
DEFAULT_SAVE_STRATEGY = "epoch"
DEFAULT_DATA_COLUMN = "text" # Default target column after processing
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ["c_attn"]
DEFAULT_LORA_BIAS = "none"
DEFAULT_WARMUP_STEPS = 0
DEFAULT_LR_SCHEDULER_TYPE = "linear"
DEFAULT_EARLY_STOPPING_PATIENCE = 2 # Consistent naming
DEFAULT_REPORT_TO = "wandb"
DEFAULT_RUN_NAME = "train_gpt2_equations"

# Source data column default from arguments
DEFAULT_SOURCE_DATA_COLUMN = "i_prompt_n"
DEFAULT_DATA_DIR = "700K"

# Wandb defaults
DEFAULT_WANDB_PROJECT = "seriguela"
DEFAULT_WANDB_ENTITY = None

# Dataset defaults
DEFAULT_DATASET_REPO_ID = "augustocsc/sintetico_natural"