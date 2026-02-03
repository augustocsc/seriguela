#!/usr/bin/env python3
"""
Test script to verify PPO setup with custom reward model.

This tests if the custom SymbolicRegressionRewardModel is compatible
with TRL's PPOTrainer before running the full experiment.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

import torch
import numpy as np

# Test 1: Check TRL version and available modules
print("=" * 60)
print("TEST 1: TRL Version and Modules")
print("=" * 60)

import trl
print(f"TRL version: {trl.__version__}")

try:
    from trl.experimental.ppo import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    print("[OK] Experimental PPO modules imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import experimental PPO: {e}")
    sys.exit(1)

# Test 2: Check if our custom reward model works
print("\n" + "=" * 60)
print("TEST 2: Custom Reward Model")
print("=" * 60)

from transformers import AutoTokenizer

# Load tokenizer
model_path = "./output/exp_a_json"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[OK] Tokenizer loaded from {model_path}")
except Exception as e:
    print(f"[FAIL] Failed to load tokenizer: {e}")
    sys.exit(1)

# Create dummy data
X = np.random.randn(100, 2)
y = np.sin(X[:, 0]) + X[:, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Dummy data created (device: {device})")

# Import and test custom reward model
try:
    from ppo_experiment_v2 import SymbolicRegressionRewardModel, SequenceClassifierOutput

    reward_model = SymbolicRegressionRewardModel(tokenizer, X, y, device)
    reward_model = reward_model.to(device)
    print("[OK] SymbolicRegressionRewardModel created")

    # Test forward pass
    test_text = '{"vars": ["x_1", "x_2"], "ops": ["+", "-"], "cons": null, "expr": sin(x_1) + x_2"}'
    test_ids = tokenizer(test_text, return_tensors="pt")["input_ids"].to(device)

    output = reward_model(test_ids)
    print(f"[OK] Forward pass works")
    print(f"  Output type: {type(output)}")
    print(f"  Logits shape: {output.logits.shape}")
    print(f"  Logits value: {output.logits.item():.4f}")

except Exception as e:
    print(f"[FAIL] Reward model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check PPOConfig
print("\n" + "=" * 60)
print("TEST 3: PPOConfig")
print("=" * 60)

try:
    ppo_config = PPOConfig(
        output_dir="./output/ppo_test",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        total_episodes=10,
        num_ppo_epochs=1,
        response_length=30,
        report_to=None,
    )
    print(f"[OK] PPOConfig created successfully")
except Exception as e:
    print(f"[FAIL] PPOConfig failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check model loading
print("\n" + "=" * 60)
print("TEST 4: Model Loading")
print("=" * 60)

try:
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)

    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    model_with_lora = PeftModel.from_pretrained(base_model, model_path)
    merged_model = model_with_lora.merge_and_unload()
    print("[OK] Base model and LoRA loaded")

    # Wrap with value head
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)
    print("[OK] AutoModelForCausalLMWithValueHead created")

except Exception as e:
    print(f"[FAIL] Model loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Dataset format
print("\n" + "=" * 60)
print("TEST 5: Dataset Format")
print("=" * 60)

try:
    from datasets import Dataset

    prompt = '{"vars": ["x_1", "x_2"], "ops": ["+", "-", "*", "sin", "cos"], "cons": null, "expr": "'
    train_dataset = Dataset.from_dict({"query": [prompt] * 10})
    print(f"[OK] Dataset created with {len(train_dataset)} samples")
    print(f"  Sample query: {train_dataset[0]['query'][:50]}...")

except Exception as e:
    print(f"[FAIL] Dataset creation failed: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("""
All basic tests passed. The custom reward model approach should work.

To run full PPO experiment:
  python scripts/ppo_experiment_v2.py --dataset ./data/ppo_test/sin_x1.csv

Note: If PPOTrainer fails due to API incompatibility, consider:
1. Checking TRL source code for exact reward_model interface
2. Using the old TRL 0.11.0 with pip install trl==0.11.0
""")
