#!/usr/bin/env python3
"""
Minimal GRPO test - tests if GRPO works with custom reward functions.
GRPO (Group Relative Policy Optimization) supports reward_funcs parameter.
"""

import os
os.environ['TRL_EXPERIMENTAL_SILENCE'] = '1'

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

import torch
import numpy as np

print("=" * 60)
print("MINIMAL GRPO TEST")
print("=" * 60)

# Test 1: Import GRPO
print("\n[1] Testing GRPO imports...")
import trl
print(f"    TRL version: {trl.__version__}")

try:
    from trl import GRPOConfig, GRPOTrainer
    print("    [OK] GRPO modules imported")
except ImportError as e:
    print(f"    [FAIL] Could not import GRPO: {e}")
    sys.exit(1)

# Test 2: Load tokenizer and model
print("\n[2] Loading base GPT-2...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("    [OK] Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained("gpt2")
print("    [OK] Model loaded")

# Test 3: Define custom reward function
print("\n[3] Defining custom reward function...")

def custom_reward_func(completions, **kwargs):
    """
    Custom reward function for symbolic regression.
    For testing, just returns length-based rewards.
    """
    rewards = []
    for completion in completions:
        # Simple reward based on completion content
        text = completion if isinstance(completion, str) else str(completion)
        # Reward shorter completions (for testing)
        reward = max(0.0, 1.0 - len(text) / 100)
        rewards.append(reward)
    return rewards

print("    [OK] Reward function defined")

# Test 4: Create dataset
print("\n[4] Creating training dataset...")
from datasets import Dataset

prompts = [
    '{"vars": ["x_1"], "ops": ["+", "sin"], "cons": null, "expr": "',
    '{"vars": ["x_1", "x_2"], "ops": ["+", "-"], "cons": null, "expr": "',
] * 2  # 4 samples

train_dataset = Dataset.from_dict({"prompt": prompts})
print(f"    [OK] Dataset with {len(train_dataset)} samples")

# Test 5: Create GRPOConfig
print("\n[5] Creating GRPOConfig...")
try:
    grpo_config = GRPOConfig(
        output_dir="./output/grpo_test",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=30,
        num_train_epochs=1,
        report_to=[],  # Empty list to disable reporting
        use_cpu=True,
        bf16=False,
    )
    print("    [OK] GRPOConfig created")
except Exception as e:
    print(f"    [FAIL] GRPOConfig: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Create GRPOTrainer
print("\n[6] Creating GRPOTrainer...")
try:
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=custom_reward_func,
    )
    print("    [OK] GRPOTrainer created!")

    # Test 7: Try a training step
    print("\n[7] Testing training (1 epoch)...")
    try:
        grpo_trainer.train()
        print("    [OK] GRPO Training completed!")
    except Exception as e:
        print(f"    [FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"    [FAIL] GRPOTrainer creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
