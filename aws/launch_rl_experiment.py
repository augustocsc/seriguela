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

# Pre-defined experiments
EXPERIMENTS = {
    "nguyen_5_test": {
        "description": "Quick test on Nguyen-5",
        "commands": [
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward r2_clipped --penalty binary --temperature fixed_0.7 "
            "--max_steps 1000 --seeds 42 --use_wandb"
        ],
    },
    "reward_ablation": {
        "description": "Compare all 3 reward functions",
        "commands": [
            # R2 Clipped
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward r2_clipped --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
            # Length Penalized
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
            # SR-IC
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward sr_ic --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
        ],
    },
    "penalty_ablation": {
        "description": "Compare binary vs gradient penalty",
        "commands": [
            # Binary penalty
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty binary --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb",
            # Gradient penalty
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb",
        ],
    },
    "temperature_ablation": {
        "description": "Compare temperature strategies",
        "commands": [
            # Fixed 0.7
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
            # Fixed 0.9
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.9 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
            # Linear annealing
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature linear_annealing "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
            # Cosine annealing
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb",
        ],
    },
    "algorithm_comparison": {
        "description": "Compare BoN-PPO vs BoN-GRPO",
        "commands": [
            # BoN-PPO
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb",
            # BoN-GRPO
            "python run_experiment.py --algorithm bon_grpo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb",
        ],
    },
    "all_nguyen": {
        "description": "Run best configuration on all Nguyen problems",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    # Full ablation suite - runs all ablation experiments sequentially (~26h)
    "full_ablation_suite": {
        "description": "Complete ablation study: reward, penalty, temperature, algorithm",
        "commands": [
            # Reward ablation (3 configs x 3 seeds x 5000 steps)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward r2_clipped --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward sr_ic --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            # Penalty ablation (2 configs x 5 seeds x 5000 steps)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty binary --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb --upload_hf",
            # Temperature ablation (4 configs x 3 seeds x 5000 steps)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.7 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature fixed_0.9 "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature linear_annealing "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 123 456 --use_wandb --upload_hf",
            # Algorithm comparison (2 configs x 5 seeds x 5000 steps)
            "python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb --upload_hf",
            "python run_experiment.py --algorithm bon_grpo --model augustocsc/gpt2_base_infix_682k "
            "--problem nguyen_5 --reward length_penalized --penalty gradient --temperature cosine_annealing "
            "--max_steps 5000 --seeds 42 123 456 789 1337 --use_wandb --upload_hf",
        ],
    },
    # Model scaling experiments - Base Infix (~12h)
    "scaling_base_infix": {
        "description": "Base model (infix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    # Model scaling experiments - Base Prefix (~12h)
    "scaling_base_prefix": {
        "description": "Base model (prefix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_base_prefix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    # Model scaling experiments - Medium Infix (~14h)
    "scaling_medium_infix": {
        "description": "Medium model (infix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_medium_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    # Model scaling experiments - Medium Prefix (~14h)
    "scaling_medium_prefix": {
        "description": "Medium model (prefix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_medium_prefix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    # Model scaling experiments - Large Infix (~16h)
    "scaling_large_infix": {
        "description": "Large model (infix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_large_infix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
        ],
    },
    # Model scaling experiments - Large Prefix (~16h)
    "scaling_large_prefix": {
        "description": "Large model (prefix) on all Nguyen benchmarks",
        "commands": [
            f"python run_experiment.py --algorithm bon_ppo --model augustocsc/gpt2_large_prefix_682k "
            f"--problem nguyen_{i} --reward length_penalized --penalty gradient --temperature cosine_annealing "
            f"--max_steps 10000 --seeds 42 123 456 --use_wandb --upload_hf"
            for i in range(1, 13)
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
sudo -E -u ubuntu bash -c "export HF_TOKEN='{hf_token}' && export WANDB_API_KEY='{wandb_token}' && /home/ubuntu/seriguela/run_all_experiments.sh"

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
