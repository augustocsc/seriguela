#!/usr/bin/env python3
"""
Minimal PPO test - tests if TRL PPO works with custom reward model.
Uses base GPT-2 (no LoRA) for simplicity.
"""

import os
os.environ['TRL_EXPERIMENTAL_SILENCE'] = '1'

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("MINIMAL PPO TEST")
print("=" * 60)

# Test 1: TRL imports
print("\n[1] Testing TRL imports...")
import trl
print(f"    TRL version: {trl.__version__}")

from trl.experimental.ppo import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
print("    [OK] PPO modules imported")

# Test 2: Load base tokenizer and model
print("\n[2] Loading base GPT-2...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("    [OK] Tokenizer loaded")

base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
print("    [OK] Base model loaded")

# Test 3: Create custom reward model
print("\n[3] Creating custom reward model...")

class SequenceClassifierOutput:
    def __init__(self, logits):
        self.logits = logits

class SimpleRewardModel(nn.Module):
    """Simple reward model that returns random scores for testing."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = type('Config', (), {'pad_token_id': tokenizer.pad_token_id})()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        # Return random rewards for testing
        rewards = torch.rand(batch_size, 1)
        return SequenceClassifierOutput(logits=rewards)

device = torch.device("cpu")  # Use CPU for testing
reward_model = SimpleRewardModel(tokenizer).to(device)
print("    [OK] Custom reward model created")

# Test forward pass
test_ids = tokenizer("test input", return_tensors="pt")["input_ids"]
output = reward_model(test_ids)
print(f"    [OK] Forward pass works, logits shape: {output.logits.shape}")

# Test 4: PPOConfig
print("\n[4] Creating PPOConfig...")
try:
    ppo_config = PPOConfig(
        output_dir="./output/ppo_test",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        total_episodes=4,
        num_ppo_epochs=1,
        response_length=20,
        report_to=None,
        use_cpu=True,  # Required for CPU-only systems
        bf16=False,
    )
    print("    [OK] PPOConfig created")
except Exception as e:
    print(f"    [FAIL] PPOConfig: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Create models with value head
print("\n[5] Creating models with value heads...")
try:
    from transformers import GenerationConfig

    # Load models directly from pretrained string (not from model object)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

    # Add generation_config (required by PPOTrainer)
    gen_config = GenerationConfig.from_pretrained("gpt2")
    policy_model.generation_config = gen_config
    ref_model.generation_config = gen_config
    value_model.generation_config = gen_config

    # Add base_model_prefix if missing
    if not hasattr(policy_model, 'base_model_prefix'):
        policy_model.base_model_prefix = 'transformer'
        ref_model.base_model_prefix = 'transformer'
        value_model.base_model_prefix = 'transformer'

    print("    [OK] Models with value heads created")
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Create dataset
print("\n[6] Creating training dataset...")
from datasets import Dataset

prompt = '{"vars": ["x_1"], "ops": ["+", "sin"], "cons": null, "expr": "'
train_dataset = Dataset.from_dict({"query": [prompt] * 4})
print(f"    [OK] Dataset with {len(train_dataset)} samples")

# Test 7: Create PPOTrainer
print("\n[7] Creating PPOTrainer...")
try:
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
    )
    print("    [OK] PPOTrainer created!")

    # Test 8: Try a training step
    print("\n[8] Testing training step...")
    try:
        ppo_trainer.train()
        print("    [OK] Training completed!")
    except Exception as e:
        print(f"    [FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"    [FAIL] PPOTrainer creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
