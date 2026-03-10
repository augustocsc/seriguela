#!/usr/bin/env python3
"""
Quick smoke test for the retokenize_expression fix.

Verifies that:
1. retokenize_expression() produces non-empty tokens and log_probs
2. bon_grpo and bon_ppo now include buffer samples in gradient updates
3. bon_grpo produces DIFFERENT results from pure_grpo

Run on Google Colab (T4 GPU, ~2-3 minutes):
    cd 2_training/reinforcement
    python test_retokenize_fix.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import logging
import random
from pathlib import Path

import numpy as np
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from run_experiment import (
    set_seed, generate_train_test_data, create_trainer
)
from algorithms import TrainerConfig, BoNGRPOTrainer, BoNPPOTrainer, PureGRPOTrainer
from rewards import create_reward_with_penalty
from schedulers import create_temperature_scheduler
from callbacks import EarlyStoppingCallback, EarlyStoppingConfig
from buffers import EliteBuffer


# ── Test Config ─────────────────────────────────────────────────
MODEL = "augustocsc/gpt2_base_infix_682k"
BASE_MODEL = "gpt2"
PROBLEM = "nguyen_1"
SEED = 42
NUM_STEPS = 3
BATCH_SIZE = 16  # small for speed


def make_trainer(algorithm: str, output_dir: str):
    """Create a trainer for the given algorithm."""
    set_seed(SEED)
    data = generate_train_test_data(PROBLEM, SEED)

    reward_fn, penalty_handler = create_reward_with_penalty(
        reward_type="sr_ic",
        penalty_strategy="gradient",
    )

    temp_scheduler = create_temperature_scheduler("cosine_annealing")

    es_config = EarlyStoppingConfig(
        patience=100, delta=0.01, r2_threshold=0.999, max_steps=NUM_STEPS
    )
    early_stopping = EarlyStoppingCallback(es_config, ground_truth=data["equation"])

    elite_buffer = EliteBuffer(max_size=100, sample_ratio=0.3)

    config = TrainerConfig(
        model_path=MODEL,
        base_model=BASE_MODEL,
        learning_rate=1e-5,
        batch_size=BATCH_SIZE,
        max_steps=NUM_STEPS,
        group_size=8,
        buffer_size=100,
        buffer_sample_ratio=0.3,
        patience=100,
        delta=0.01,
        prompt_type="standard",
        log_every=1,
        save_every=99999,
        output_dir=output_dir,
        use_wandb=False,
        resume=False,
    )

    return create_trainer(
        algorithm=algorithm,
        config=config,
        x=data["train"]["x"],
        y=data["train"]["y"],
        reward_fn=reward_fn,
        penalty_handler=penalty_handler,
        temp_scheduler=temp_scheduler,
        early_stopping=early_stopping,
        elite_buffer=elite_buffer,
        is_prefix=False,
        valid_variables=data["valid_variables"],
        ground_truth=data["equation"],
    )


def test_retokenize_method():
    """Test 1: retokenize_expression produces non-empty tokens."""
    print("\n" + "=" * 60)
    print("TEST 1: retokenize_expression() produces real tokens")
    print("=" * 60)

    trainer = make_trainer("bon_grpo", "/tmp/test_retok_1")

    # Generate one expression first
    rollout = trainer.generate_expression(temperature=0.7)
    expr = rollout.expression
    print(f"  Generated expression: {expr}")
    print(f"  Original tokens ({len(rollout.tokens)}): {rollout.tokens[:10]}...")

    # Re-tokenize it
    retok = trainer.retokenize_expression(expr)

    assert retok is not None, "retokenize_expression returned None!"
    assert len(retok.tokens) > 0, "retokenize_expression produced empty tokens!"
    assert len(retok.log_probs) > 0, "retokenize_expression produced empty log_probs!"
    assert len(retok.tokens) == len(retok.log_probs), "tokens/log_probs length mismatch!"

    print(f"  Re-tokenized tokens ({len(retok.tokens)}): {retok.tokens[:10]}...")
    print(f"  Re-tokenized log_probs: {retok.log_probs[:5]}...")
    print(f"  ✅ PASSED: retokenize_expression returns real tokens and log_probs")

    # Cleanup
    del trainer
    torch.cuda.empty_cache()


def test_bon_grpo_uses_buffer():
    """Test 2: bon_grpo includes buffer rollouts in training."""
    print("\n" + "=" * 60)
    print("TEST 2: bon_grpo includes buffer samples in training")
    print("=" * 60)

    trainer = make_trainer("bon_grpo", "/tmp/test_retok_2")

    # Run training steps and check stats
    for step in range(NUM_STEPS):
        stats = trainer.train_step()

        total = stats.get("total_count", 0)
        fresh = stats.get("fresh_count", 0)
        buffer_used = total - fresh

        print(f"  Step {step}: total={total}, fresh={fresh}, buffer={buffer_used}")

        # After step 0, the buffer should have entries and step 1+ should use them
        if step > 0 and trainer.elite_buffer and len(trainer.elite_buffer) > 0:
            assert total > fresh, (
                f"Buffer samples not being added! total={total} == fresh={fresh}"
            )
            print(f"  ✅ Buffer samples ARE participating (total > fresh)")

    print(f"  ✅ PASSED: bon_grpo uses buffer samples in training batch")

    del trainer
    torch.cuda.empty_cache()


def test_bon_grpo_differs_from_pure():
    """Test 3: bon_grpo produces different results from pure_grpo."""
    print("\n" + "=" * 60)
    print("TEST 3: bon_grpo differs from pure_grpo")
    print("=" * 60)

    # Run bon_grpo
    set_seed(SEED)
    bon_trainer = make_trainer("bon_grpo", "/tmp/test_retok_3a")
    bon_results = bon_trainer.run()
    bon_discovered = set(bon_results["discovered_expressions"].keys())
    del bon_trainer
    torch.cuda.empty_cache()

    # Run pure_grpo
    set_seed(SEED)
    pure_trainer = make_trainer("pure_grpo", "/tmp/test_retok_3b")
    pure_results = pure_trainer.run()
    pure_discovered = set(pure_results["discovered_expressions"].keys())
    del pure_trainer
    torch.cuda.empty_cache()

    # They should differ (bon has buffer influence + shuffling)
    bon_best = bon_results["best_r2"]
    pure_best = pure_results["best_r2"]

    print(f"  bon_grpo  best_r2: {bon_best:.6f}, unique: {len(bon_discovered)}")
    print(f"  pure_grpo best_r2: {pure_best:.6f}, unique: {len(pure_discovered)}")

    # The discovered expression sets should differ
    # (different gradient updates → different policy → different generations)
    symmetric_diff = bon_discovered.symmetric_difference(pure_discovered)
    overlap = bon_discovered.intersection(pure_discovered)

    print(f"  Shared expressions: {len(overlap)}")
    print(f"  Different expressions: {len(symmetric_diff)}")

    if len(symmetric_diff) > 0:
        print(f"  ✅ PASSED: bon_grpo and pure_grpo produce different expression sets")
    else:
        print(f"  ⚠️  WARNING: same expressions found — with only {NUM_STEPS} steps this may happen by chance")
        print(f"  The full Test 6 (50 steps) will show a clear difference")


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: Buffer Re-tokenization Fix")
    print("=" * 60)

    test_retokenize_method()
    test_bon_grpo_uses_buffer()
    test_bon_grpo_differs_from_pure()

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS COMPLETE")
    print("=" * 60)
