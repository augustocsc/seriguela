#!/usr/bin/env python3
"""
Quick test script to verify all components work correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "classes"))

print("Testing Seriguela RL Components")
print("=" * 50)

# Test 1: Reward functions
print("\n1. Testing Reward Functions...")
from rewards import R2ClippedReward, LengthPenalizedReward, SRICReward, PenaltyHandler, PenaltyStrategy

# Create simple test data
x = np.linspace(0, 2, 50).reshape(-1, 1)
y = x.flatten() ** 2 + 1  # x^2 + 1

# Test R2 Clipped
r2_reward = R2ClippedReward()
result = r2_reward.compute("x_1**2 + 1", x, y, is_prefix=False)
print(f"   R2 Clipped: expression='x_1**2 + 1', R²={result.r2:.4f}, reward={result.reward:.4f}")

# Test Length Penalized
lp_reward = LengthPenalizedReward(alpha=0.01)
result = lp_reward.compute("x_1**2 + 1", x, y, is_prefix=False)
print(f"   Length Penalized: R²={result.r2:.4f}, complexity={result.complexity}, reward={result.reward:.4f}")

# Test SR-IC
sric_reward = SRICReward(lambda_complexity=0.1)
result = sric_reward.compute("x_1**2 + 1", x, y, is_prefix=False)
print(f"   SR-IC: R²={result.r2:.4f}, complexity={result.complexity}, reward={result.reward:.4f}")

# Test penalty handler
print("\n2. Testing Penalty Handler...")
from rewards.base import ErrorType

binary_handler = PenaltyHandler(PenaltyStrategy.BINARY)
gradient_handler = PenaltyHandler(PenaltyStrategy.GRADIENT)

print(f"   Binary penalty for PARSING: {binary_handler.get_penalty(ErrorType.PARSING)}")
print(f"   Gradient penalty for PARSING: {gradient_handler.get_penalty(ErrorType.PARSING)}")
print(f"   Gradient penalty for NAN_INF: {gradient_handler.get_penalty(ErrorType.NAN_INF)}")
print(f"   Gradient penalty for NEGATIVE_R2: {gradient_handler.get_penalty(ErrorType.NEGATIVE_R2)}")

# Test 2: Temperature Schedulers
print("\n3. Testing Temperature Schedulers...")
from schedulers import FixedTemperature, LinearAnnealing, CosineAnnealing

fixed = FixedTemperature(0.7)
linear = LinearAnnealing(t_max=1.0, t_min=0.5)
cosine = CosineAnnealing(t_max=1.0, t_min=0.5)

total_steps = 100
for step in [0, 25, 50, 75, 100]:
    print(f"   Step {step:3d}: Fixed={fixed.get_temperature(step, total_steps):.2f}, "
          f"Linear={linear.get_temperature(step, total_steps):.2f}, "
          f"Cosine={cosine.get_temperature(step, total_steps):.2f}")

# Test 3: Early Stopping
print("\n4. Testing Early Stopping...")
from callbacks import EarlyStoppingCallback, EarlyStoppingConfig, StopReason

config = EarlyStoppingConfig(
    patience=3,
    delta=0.01,
    r2_threshold=0.999,
    max_steps=100,
)
early_stop = EarlyStoppingCallback(config, ground_truth="x_1**2 + 1")

# Simulate training
for step in range(10):
    mean_reward = 0.5 + step * 0.05
    best_r2 = 0.8 + step * 0.02
    policy_entropy = 2.0 - step * 0.1

    reason = early_stop.check(
        step=step,
        mean_reward=mean_reward,
        best_r2=best_r2,
        best_expr="x_1**2 + C",
        policy_entropy=policy_entropy
    )

    if reason != StopReason.NONE:
        print(f"   Stopped at step {step}: {reason.value}")
        break
else:
    print(f"   No early stop triggered (simulated 10 steps)")

# Test 4: Elite Buffer
print("\n5. Testing Elite Buffer...")
from buffers import EliteBuffer, BufferEntry

buffer = EliteBuffer(max_size=10, sample_ratio=0.2)

# Add some expressions
for i in range(15):
    entry = BufferEntry(
        expression=f"x_1**{i+1}",
        r2=0.5 + i * 0.03,
        reward=0.5 + i * 0.03,
        log_prob=-2.0 - i * 0.1,
        complexity=i + 3,
        step_added=i
    )
    buffer.add(entry)

stats = buffer.stats()
print(f"   Buffer size: {stats['size']}/{buffer.max_size}")
print(f"   Mean R²: {stats['mean_r2']:.4f}, Max R²: {stats['max_r2']:.4f}")

best = buffer.get_best(3)
print(f"   Top 3 expressions:")
for b in best:
    print(f"      {b.expression}: R²={b.r2:.4f}")

# Test 5: Sample from buffer
samples = buffer.sample(10)
print(f"   Sampled {len(samples)} expressions from buffer")

print("\n" + "=" * 50)
print("All component tests passed!")
print("=" * 50)
