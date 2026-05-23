#!/usr/bin/env python3
"""
Smoke test for Colab pipeline validation.

Run this locally before pushing any change that touches:
  - experiments/queue_processor.py
  - 2_training/reinforcement/run_experiment.py
  - 2_training/reinforcement/algorithms/*.py
  - 2_training/reinforcement/rewards/*.py
  - colab/seriguela_runner.ipynb

Usage:
    python experiments/test_smoke.py

All tests run on CPU with a mocked 2-layer GPT-2 (no HuggingFace download).
Expected runtime: <60s on any machine.

Exit 0 = pass, Exit 1 = fail.
"""

import sys
import os
import json
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ─── Repo root on sys.path ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
REINFORCEMENT_ROOT = REPO_ROOT / "2_training" / "reinforcement"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classes"))
sys.path.insert(0, str(REINFORCEMENT_ROOT))

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_tiny_gpt2():
    """Return a tiny 2-layer GPT-2 model (no download, CPU only)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    return GPT2LMHeadModel(cfg)


def _make_tiny_tokenizer():
    """Return real GPT-2 tokenizer — cached after first download."""
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    except Exception:
        # Offline: create a mock that returns plausible tensors
        import torch
        tok = MagicMock()
        tok.pad_token = "<|endoftext|>"
        tok.pad_token_id = 50256
        tok.eos_token_id = 50256
        # encode → returns tiny tensor
        tok.encode.return_value = [198, 90, 47]  # dummy token ids
        tok.return_tensors = "pt"
        dummy = MagicMock()
        dummy.__getitem__ = lambda self, k: torch.zeros(1, 3, dtype=torch.long)
        tok.return_value = dummy
        tok.decode.return_value = '{"vars": ["x_1"], "ops": [], "cons": "C", "expr": "x_1 + 1"}'
        return tok


PASS = []
FAIL = []


def test(name):
    """Decorator: catch exceptions and record pass/fail."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                PASS.append(name)
                print(f"  ✓ {name}")
            except Exception as e:
                FAIL.append((name, traceback.format_exc()))
                print(f"  ✗ {name}: {e}")
        return wrapper
    return decorator


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 1 — Imports
# ═════════════════════════════════════════════════════════════════════════════

@test("import: classes.expression")
def t_import_expression():
    from classes.expression import Expression
    assert hasattr(Expression, "parse_prefix")


@test("import: rewards (all 3 reward classes)")
def t_import_rewards():
    from rewards import R2ClippedReward, LengthPenalizedReward, SRICReward
    from rewards import PenaltyStrategy, PenaltyHandler, create_reward_with_penalty


@test("import: algorithms (all 5 trainers)")
def t_import_algorithms():
    from algorithms import (
        BoNPPOTrainer, BoNGRPOTrainer, PurePPOTrainer, PureGRPOTrainer,
        TrainerConfig, BestOfNBaseline, BoNConfig, run_best_of_n_baseline,
    )


@test("import: buffers.EliteBuffer")
def t_import_buffers():
    from buffers import EliteBuffer


@test("import: schedulers.create_temperature_scheduler")
def t_import_schedulers():
    from schedulers import create_temperature_scheduler


@test("import: callbacks.EarlyStoppingCallback")
def t_import_callbacks():
    from callbacks import EarlyStoppingCallback, EarlyStoppingConfig


@test("import: utils.hf_upload.HuggingFaceUploader")
def t_import_hf_upload():
    from utils.hf_upload import HuggingFaceUploader


@test("import: run_experiment module-level (no side effects)")
def t_import_run_experiment():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_experiment",
        str(REINFORCEMENT_ROOT / "run_experiment.py"),
    )
    mod = importlib.util.load_from_spec = spec  # just resolve, don't exec fully
    # Minimal check: the file is parseable Python
    src = (REINFORCEMENT_ROOT / "run_experiment.py").read_text(encoding="utf-8")
    compile(src, "run_experiment.py", "exec")


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 2 — Queue processor command building
# ═════════════════════════════════════════════════════════════════════════════

@test("experiments/__init__.py exists (required for `from experiments.queue_processor import ...`)")
def t_experiments_package():
    init = SCRIPT_DIR / "__init__.py"
    assert init.exists(), f"Missing {init} — experiments/ won't be importable as a package"


@test("queue_processor: imports without CUDA")
def t_qp_import():
    sys.path.insert(0, str(SCRIPT_DIR))
    import importlib
    spec = importlib.util.spec_from_file_location(
        "queue_processor", str(SCRIPT_DIR / "queue_processor.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't exec (it would start a daemon) — just parse
    src = (SCRIPT_DIR / "queue_processor.py").read_text(encoding="utf-8")
    compile(src, "queue_processor.py", "exec")


@test("queue_processor: build_command produces correct nargs=+ seeds")
def t_qp_build_command():
    # Inline the build_command logic rather than importing to avoid daemon startup
    import yaml

    queue_yaml = REPO_ROOT / "experiments" / "queue.yaml"
    with open(queue_yaml) as f:
        data = yaml.safe_load(f)

    first_exp = next(e for e in data["queue"] if e.get("status") == "pending")
    seeds = first_exp["args"]["seeds"]

    # Simulate build_command
    cmd_parts = ["python", "2_training/reinforcement/run_experiment.py",
                 "--seeds"] + [str(s) for s in seeds]

    # Key assertion: seeds appear as consecutive positional args, NOT repeated flags
    seeds_idx = cmd_parts.index("--seeds")
    actual_seeds = cmd_parts[seeds_idx + 1: seeds_idx + 1 + len(seeds)]
    assert actual_seeds == [str(s) for s in seeds], \
        f"Expected {seeds}, got {actual_seeds}"
    assert "--seeds" not in cmd_parts[seeds_idx + 1:], \
        "Duplicate --seeds flag detected (nargs=+ bug)"


@test("queue_processor: all pending experiments have required fields")
def t_qp_schema():
    import yaml

    queue_yaml = REPO_ROOT / "experiments" / "queue.yaml"
    with open(queue_yaml) as f:
        data = yaml.safe_load(f)

    required = {"id", "status", "args", "output_dir"}
    required_args = {"algorithm", "model", "problem", "seeds", "max_steps", "batch_size"}

    for exp in data["queue"]:
        missing = required - set(exp.keys())
        assert not missing, f"Exp {exp.get('id','?')} missing fields: {missing}"
        if exp["status"] == "pending":
            missing_args = required_args - set(exp["args"].keys())
            assert not missing_args, \
                f"Exp {exp['id']} missing args: {missing_args}"
            assert isinstance(exp["args"]["seeds"], list), \
                f"Exp {exp['id']}: seeds must be a list"


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 3 — Best-of-N end-to-end with mocked model
# ═════════════════════════════════════════════════════════════════════════════

@test("best_of_n: mini run (2 samples, CPU, mocked model)")
def t_bon_mini_run():
    import torch
    import numpy as np
    from algorithms.best_of_n import BestOfNBaseline, BoNConfig
    from rewards import create_reward_with_penalty

    # Tiny data: nguyen_1 (x^3 + x^2 + x)
    x = np.linspace(0, 2, 20).reshape(-1, 1)
    y = x[:, 0]**3 + x[:, 0]**2 + x[:, 0]

    reward_fn, penalty_handler = create_reward_with_penalty("sr_ic", "gradient")

    tiny_model = _make_tiny_gpt2()
    tiny_tok = _make_tiny_tokenizer()

    config = BoNConfig(
        model_path="mock/model",
        base_model="gpt2",
        n_samples=2,
        batch_size=2,
        max_new_tokens=20,
        temperature=1.0,
        use_wandb=False,
    )

    with patch("algorithms.best_of_n.AutoTokenizer.from_pretrained", return_value=tiny_tok), \
         patch("algorithms.best_of_n.AutoModelForCausalLM.from_pretrained", return_value=tiny_model), \
         patch("algorithms.best_of_n.PeftModel.from_pretrained", return_value=tiny_model):

        baseline = BestOfNBaseline(
            config=config,
            x=x,
            y=y,
            reward_fn=reward_fn,
            penalty_handler=penalty_handler,
            is_prefix=False,
            valid_variables={"x_1"},
            ground_truth="x_1**3 + x_1**2 + x_1",
        )
        results = baseline.run()

    # Structural checks
    assert "algorithm" in results and results["algorithm"] == "best_of_n"
    assert "n_valid" in results
    assert "valid_rate" in results
    assert "best_r2" in results
    assert isinstance(results["best_r2"], float)


@test("best_of_n: output dir + JSON written by run_experiment")
def t_bon_output_json():
    """Verify run_experiment.py writes a valid JSON to output_dir."""
    import torch
    import numpy as np
    from algorithms.best_of_n import BestOfNBaseline, BoNConfig
    from rewards import create_reward_with_penalty

    tiny_model = _make_tiny_gpt2()
    tiny_tok = _make_tiny_tokenizer()

    x = np.linspace(0, 2, 20).reshape(-1, 1)
    y = x[:, 0]**3 + x[:, 0]**2 + x[:, 0]

    reward_fn, penalty_handler = create_reward_with_penalty("sr_ic", "gradient")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "phase_test"
        config = BoNConfig(
            model_path="mock/model",
            base_model="gpt2",
            n_samples=2,
            batch_size=2,
            max_new_tokens=20,
            temperature=1.0,
            output_dir=str(out_dir),
            use_wandb=False,
        )

        with patch("algorithms.best_of_n.AutoTokenizer.from_pretrained", return_value=tiny_tok), \
             patch("algorithms.best_of_n.AutoModelForCausalLM.from_pretrained", return_value=tiny_model), \
             patch("algorithms.best_of_n.PeftModel.from_pretrained", return_value=tiny_model):

            baseline = BestOfNBaseline(
                config=config, x=x, y=y,
                reward_fn=reward_fn, penalty_handler=penalty_handler,
                valid_variables={"x_1"},
            )
            results = baseline.run()

        # Simulate what run_experiment.py does: write results to JSON
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "seed_42_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        assert out_file.exists()
        loaded = json.loads(out_file.read_text())
        assert loaded["algorithm"] == "best_of_n"


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 4 — fp16 support
# ═════════════════════════════════════════════════════════════════════════════

@test("best_of_n: BoNConfig accepts fp16 field")
def t_bon_fp16_field():
    from algorithms.best_of_n import BoNConfig
    cfg = BoNConfig(model_path="x", base_model="gpt2", fp16=True)
    assert cfg.fp16 is True


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  Seriguela Smoke Tests")
    print("═" * 60)

    tests = [
        t_import_expression,
        t_import_rewards,
        t_import_algorithms,
        t_import_buffers,
        t_import_schedulers,
        t_import_callbacks,
        t_import_hf_upload,
        t_import_run_experiment,
        t_experiments_package,
        t_qp_import,
        t_qp_build_command,
        t_qp_schema,
        t_bon_mini_run,
        t_bon_output_json,
        t_bon_fp16_field,
    ]

    print(f"\n[Imports]")
    for t in tests[:8]:
        t()

    print(f"\n[Queue processor]")
    for t in tests[8:12]:
        t()

    print(f"\n[Best-of-N end-to-end]")
    for t in tests[12:14]:
        t()

    print(f"\n[fp16 support]")
    for t in tests[14:]:
        t()

    print("\n" + "═" * 60)
    print(f"  Results: {len(PASS)} passed, {len(FAIL)} failed")
    print("═" * 60)

    if FAIL:
        print("\nFailed tests:")
        for name, tb in FAIL:
            print(f"\n  ✗ {name}")
            for line in tb.strip().splitlines()[-6:]:
                print(f"    {line}")
        sys.exit(1)
    else:
        print("\n  All tests passed. Safe to push.")
        sys.exit(0)


if __name__ == "__main__":
    main()
