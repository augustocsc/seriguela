#!/usr/bin/env python3
"""
Launch RL experiments on AWS EC2.

Usage:
    python launch_rl_experiment.py --experiment nguyen_5_test
    python launch_rl_experiment.py --experiment full_ablation --instance_type g5.xlarge
"""

import argparse
import boto3
import json
import time
from datetime import datetime
from pathlib import Path

# AWS Configuration
AWS_REGION = "us-east-1"
AMI_ID = "ami-0f3d7b789119ccbfa"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04)
KEY_NAME = "chave-gpu-nova"
SECURITY_GROUP_ID = "sg-0deaa73e23482e3f6"

# Instance types and their GPUs
INSTANCE_TYPES = {
    "g5.xlarge": {"gpu": "A10G (24GB)", "vcpus": 4, "ram": "16GB", "cost": "$1.01/hr"},
    "g5.2xlarge": {"gpu": "A10G (24GB)", "vcpus": 8, "ram": "32GB", "cost": "$1.21/hr"},
    "g5.4xlarge": {"gpu": "A10G (24GB)", "vcpus": 16, "ram": "64GB", "cost": "$1.62/hr"},
}

# Pre-defined experiments (single seed for stress/exploration experiments)
EXPERIMENTS = {
    # Quick factorial verification test (19 configs covering all dimensions)
    "quick_factorial_test": {
        "description": "Quick test of all config dimensions (19 runs, 10 steps each)",
        "commands": [
            "python quick_factorial_test.py --max_steps 10"
        ],
    },

    # Quick test
    "nguyen_5_test": {
        "description": "Quick test on Nguyen-5",
        "commands": [
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward r2_clipped --penalty binary --temperature fixed_0.7 "
            "--max_steps 1000 --seeds 42 --use_wandb"
        ],
    },

    # ========== ABLATION STUDIES (single seed) ==========
    "reward_ablation": {
        "description": "Compare all 3 reward functions",
        "commands": [
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward r2_clipped --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward sr_ic --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },
    "penalty_ablation": {
        "description": "Compare binary vs gradient penalty",
        "commands": [
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty binary --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },
    "temperature_ablation": {
        "description": "Compare temperature strategies",
        "commands": [
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.9 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature linear_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },

    # ========== ALGORITHM COMPARISON (single seed) ==========
    "algorithm_comparison": {
        "description": "Compare all algorithms: BoN-PPO, BoN-GRPO, Pure-PPO, Pure-GRPO, Best-of-N",
        "commands": [
            # BoN-PPO (hybrid with elite buffer)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # BoN-GRPO (hybrid with elite buffer)
            "python run_experiment.py --algorithm bon_grpo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Pure PPO (no buffer)
            "python run_experiment.py --algorithm pure_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Pure GRPO (no buffer)
            "python run_experiment.py --algorithm pure_grpo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Best-of-N baseline (no RL training)
            "python run_experiment.py --algorithm best_of_n --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },

    # ========== PROMPT ROBUSTNESS (single seed) ==========
    "prompt_robustness": {
        "description": "Compare standard vs oracle vs distractor prompts",
        "commands": [
            # Standard prompt
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--prompt_type standard --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Oracle prompt (helpful hints)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--prompt_type oracle --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Distractor prompt (misleading hints)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--prompt_type distractor --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },

    # ========== NOISE ROBUSTNESS (single seed) ==========
    "noise_robustness": {
        "description": "Test robustness to different noise levels",
        "commands": [
            # No noise (baseline)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type none --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Low noise (1%)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type gaussian --noise_level 0.01 --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # Medium noise (5%)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type gaussian --noise_level 0.05 --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # High noise (10%)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type gaussian --noise_level 0.1 --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },

    # ========== FULL ABLATION SUITE (single seed) ==========
    # Complete ablation covering EVERYTHING that was missing from previous experiment
    # Total: 21 runs on Nguyen-5
    "full_ablation_suite": {
        "description": "Complete ablation: algorithm, reward, penalty, temperature, prompt, noise",
        "commands": [
            # ============================================================
            # 1. ALGORITHM COMPARISON (5 runs) - Critical for understanding RL contribution
            # ============================================================
            # 1.1 BoN-PPO (hybrid: PPO + elite buffer)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 1.2 BoN-GRPO (hybrid: GRPO + elite buffer)
            "python run_experiment.py --algorithm bon_grpo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 1.3 Pure PPO (NO buffer - tests if RL alone helps)
            "python run_experiment.py --algorithm pure_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 1.4 Pure GRPO (NO buffer - tests if RL alone helps)
            "python run_experiment.py --algorithm pure_grpo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 1.5 Best-of-N (NO RL - pure sampling baseline)
            "python run_experiment.py --algorithm best_of_n --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",

            # ============================================================
            # 2. REWARD FUNCTION ABLATION (3 runs) - WAS NOT LOGGED PROPERLY
            # ============================================================
            # 2.1 R² Clipped (pure fitness)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward r2_clipped --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 2.2 Length-Penalized R² (simplicity bias)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 2.3 SR-IC (information-theoretic complexity)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward sr_ic --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",

            # ============================================================
            # 3. PENALTY STRATEGY ABLATION (2 runs) - WAS NOT LOGGED PROPERLY
            # ============================================================
            # 3.1 Binary penalty (-1.0 for all invalid)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty binary --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 3.2 Gradient penalty (differentiated by error type)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",

            # ============================================================
            # 4. TEMPERATURE SCHEDULE ABLATION (4 runs) - WAS NOT LOGGED PROPERLY
            # ============================================================
            # 4.1 Fixed 0.7 (low, exploitative)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 4.2 Fixed 0.9 (high, explorative)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.9 "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 4.3 Linear annealing (1.0 → 0.5)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature linear_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 4.4 Cosine annealing (smooth decay)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 --use_wandb --upload_hf",

            # ============================================================
            # 5. PROMPT ROBUSTNESS (3 runs) - WAS MISSING ENTIRELY
            # ============================================================
            # 5.1 Standard prompt (all operators)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--prompt_type standard --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 5.2 Oracle prompt (true operators - helpful hint)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--prompt_type oracle --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 5.3 Distractor prompt (wrong operators - misleading)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--prompt_type distractor --max_steps 5000 --seeds 42 --use_wandb --upload_hf",

            # ============================================================
            # 6. NOISE ROBUSTNESS (4 runs) - WAS MISSING ENTIRELY
            # ============================================================
            # 6.1 Clean data (0% noise - baseline)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type none --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 6.2 Low noise (1%)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type gaussian --noise_level 0.01 --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 6.3 Medium noise (5%)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type gaussian --noise_level 0.05 --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
            # 6.4 High noise (10%)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--noise_type gaussian --noise_level 0.1 --max_steps 5000 --seeds 42 --use_wandb --upload_hf",
        ],
    },

    # ========== SCALING EXPERIMENTS (single seed) ==========
    "scaling_base_infix": {
        "description": "Base model (infix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    "scaling_base_prefix": {
        "description": "Base model (prefix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_prefix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    "scaling_medium_infix": {
        "description": "Medium model (infix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_medium_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    "scaling_medium_prefix": {
        "description": "Medium model (prefix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_medium_prefix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    "scaling_large_infix": {
        "description": "Large model (infix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_large_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    "scaling_large_prefix": {
        "description": "Large model (prefix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_large_prefix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },

    # ========== ALL NGUYEN BENCHMARKS (single seed) ==========
    "all_nguyen": {
        "description": "Run best config on all Nguyen problems",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },

    # ========== FULL FACTORIAL EXPERIMENTS ==========
    # 6 models × 3 problems × 1,440 configs = 25,920 total runs
    # Each factorial_* experiment runs one model on all problems with all 1,440 configs
    # = 4,320 runs per model

    "factorial_base_infix": {
        "description": "Full factorial: Base Infix (4,320 runs)",
        "commands": [
            "python run_factorial_experiment.py --model_idx 0 --max_steps 5000 --batch_size 32"
        ],
    },
    "factorial_base_prefix": {
        "description": "Full factorial: Base Prefix (4,320 runs)",
        "commands": [
            "python run_factorial_experiment.py --model_idx 1 --max_steps 5000 --batch_size 32"
        ],
    },
    "factorial_medium_infix": {
        "description": "Full factorial: Medium Infix (4,320 runs)",
        "commands": [
            "python run_factorial_experiment.py --model_idx 2 --max_steps 5000 --batch_size 32"
        ],
    },
    "factorial_medium_prefix": {
        "description": "Full factorial: Medium Prefix (4,320 runs)",
        "commands": [
            "python run_factorial_experiment.py --model_idx 3 --max_steps 5000 --batch_size 32"
        ],
    },
    "factorial_large_infix": {
        "description": "Full factorial: Large Infix (4,320 runs)",
        "commands": [
            "python run_factorial_experiment.py --model_idx 4 --max_steps 5000 --batch_size 16"
        ],
    },
    "factorial_large_prefix": {
        "description": "Full factorial: Large Prefix (4,320 runs)",
        "commands": [
            "python run_factorial_experiment.py --model_idx 5 --max_steps 5000 --batch_size 16"
        ],
    },
}


def read_local_tokens() -> tuple[str, str]:
    """Read tokens from local file."""
    tokens_file = Path.home() / ".tokens.txt"
    hf_token = ""
    wandb_token = ""

    if tokens_file.exists():
        with open(tokens_file) as f:
            for line in f:
                if "huggingface" in line.lower():
                    hf_token = line.split("=")[1].strip()
                elif "wandb" in line.lower():
                    wandb_token = line.split("=")[1].strip()

    return hf_token, wandb_token


def generate_userdata(experiment_name: str, commands: list, hf_token: str = "", wandb_token: str = "") -> str:
    """Generate EC2 userdata script."""
    commands_str = "\n".join([f"    {cmd}" for cmd in commands])

    # Token setup for ubuntu user
    if hf_token or wandb_token:
        token_setup = f"""
# Setup credentials for ubuntu user
sudo -u ubuntu mkdir -p /home/ubuntu/.cache/huggingface

# Create .netrc for WandB
cat > /home/ubuntu/.netrc << 'NETRC'
machine api.wandb.ai
  login user
  password {wandb_token}
NETRC
chmod 600 /home/ubuntu/.netrc
chown ubuntu:ubuntu /home/ubuntu/.netrc

# Create HuggingFace token file
echo "{hf_token}" > /home/ubuntu/.cache/huggingface/token
chmod 600 /home/ubuntu/.cache/huggingface/token
chown -R ubuntu:ubuntu /home/ubuntu/.cache/huggingface

# Export tokens for current session
export HF_TOKEN="{hf_token}"
export WANDB_API_KEY="{wandb_token}"
"""
    else:
        token_setup = ""

    return f"""#!/bin/bash
set -e

# Set HOME explicitly (required for userdata scripts)
export HOME=/root

# Log everything
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting experiment: {experiment_name}"
echo "Date: $(date)"

# Update system
apt-get update -y

{token_setup}

# Clone repository and checkout RL experiment branch
cd /home/ubuntu
if [ ! -d "seriguela" ]; then
    sudo -u ubuntu git clone https://github.com/augustocsc/seriguela.git
fi

cd seriguela
sudo -u ubuntu git config --global --add safe.directory /home/ubuntu/seriguela
sudo -u ubuntu git fetch origin
sudo -u ubuntu git checkout experiment/ppo-symbolic-regression
sudo -u ubuntu git pull origin experiment/ppo-symbolic-regression

# Setup Python environment as ubuntu user
sudo -u ubuntu python3 -m venv .venv

# Install dependencies
sudo -u ubuntu .venv/bin/pip install --upgrade pip
sudo -u ubuntu .venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
sudo -u ubuntu .venv/bin/pip install -r requirements.txt
sudo -u ubuntu .venv/bin/pip install wandb huggingface_hub

# Create experiment script
cat > /home/ubuntu/seriguela/run_all_experiments.sh << 'EXPERIMENT_SCRIPT'
#!/bin/bash
set -e
cd /home/ubuntu/seriguela
source .venv/bin/activate
cd 2_training/reinforcement
{commands_str}
EXPERIMENT_SCRIPT
chmod +x /home/ubuntu/seriguela/run_all_experiments.sh
chown ubuntu:ubuntu /home/ubuntu/seriguela/run_all_experiments.sh

# Run experiments as ubuntu user with proper environment
echo "Running experiments..."
sudo -E -u ubuntu bash -c "export HF_TOKEN='{hf_token}' && export WANDB_API_KEY='{wandb_token}' && export HF_HOME='/home/ubuntu/.cache/huggingface' && export HOME='/home/ubuntu' && /home/ubuntu/seriguela/run_all_experiments.sh"

echo "Experiments completed!"
echo "Stopping instance..."

# Results are uploaded to HuggingFace via --upload_hf flag in experiments

# Signal completion
touch /tmp/experiment_complete

# Optional: Stop instance after completion
# sudo shutdown -h now
"""


def launch_instance(
    experiment_name: str,
    instance_type: str = "g5.xlarge",
    dry_run: bool = False,
) -> dict:
    """Launch EC2 instance with experiment."""

    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENTS.keys())}")

    experiment = EXPERIMENTS[experiment_name]

    # Read local tokens and embed in userdata
    hf_token, wandb_token = read_local_tokens()
    if hf_token:
        print("Found HuggingFace token locally - will embed in instance")
    if wandb_token:
        print("Found W&B token locally - will embed in instance")

    userdata = generate_userdata(experiment_name, experiment["commands"], hf_token, wandb_token)

    print(f"Launching experiment: {experiment_name}")
    print(f"Description: {experiment['description']}")
    print(f"Instance type: {instance_type} ({INSTANCE_TYPES.get(instance_type, {})})")
    print(f"Commands to run: {len(experiment['commands'])}")

    if dry_run:
        print("\nDry run - would execute:")
        print(userdata)
        return {"dry_run": True}

    # Create EC2 client
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Launch instance
    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=instance_type,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SECURITY_GROUP_ID],
        MinCount=1,
        MaxCount=1,
        UserData=userdata,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"augusto-seriguela-{experiment_name}"},
                    {"Key": "Project", "Value": "seriguela"},
                    {"Key": "Experiment", "Value": experiment_name},
                    {"Key": "Owner", "Value": "augusto"},
                ],
            }
        ],
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 100,  # 100GB storage
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print(f"\nInstance launched: {instance_id}")

    # Wait for instance to be running
    print("Waiting for instance to start...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    # Get public IP
    instance_info = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = instance_info["Reservations"][0]["Instances"][0].get("PublicIpAddress")

    print(f"Instance running!")
    print(f"Public IP: {public_ip}")
    print(f"\nTo connect:")
    print(f"  ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip}")
    print(f"\nTo monitor:")
    print(f"  ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip} 'tail -f /var/log/user-data.log'")

    return {
        "instance_id": instance_id,
        "public_ip": public_ip,
        "experiment": experiment_name,
        "instance_type": instance_type,
    }


def list_experiments():
    """List available experiments."""
    print("Available experiments:")
    print("=" * 60)
    for name, exp in EXPERIMENTS.items():
        print(f"\n{name}:")
        print(f"  Description: {exp['description']}")
        print(f"  Commands: {len(exp['commands'])}")


def main():
    parser = argparse.ArgumentParser(description="Launch RL experiments on AWS")
    parser.add_argument("--experiment", type=str, help="Experiment name to run")
    parser.add_argument("--instance_type", type=str, default="g5.xlarge",
                        choices=list(INSTANCE_TYPES.keys()))
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be executed")

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        parser.print_help()
        print("\nUse --list to see available experiments")
        return

    result = launch_instance(
        experiment_name=args.experiment,
        instance_type=args.instance_type,
        dry_run=args.dry_run,
    )

    # Save launch info
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        info_file = Path(f"aws_launch_{args.experiment}_{timestamp}.json")
        with open(info_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nLaunch info saved to: {info_file}")


if __name__ == "__main__":
    main()
