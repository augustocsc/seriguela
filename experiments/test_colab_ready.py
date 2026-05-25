#!/usr/bin/env python3
"""
Colab readiness test for Phase 1b.

Run this BEFORE starting queue_processor.py to confirm the environment
is correctly set up. Works locally (CPU) and on Colab (GPU).

Usage:
    python experiments/test_colab_ready.py

Checks:
  1. GPU availability + VRAM
  2. Required packages installed
  3. Feynman CSV data is real (not LFS pointers)
  4. Model loads from HuggingFace (uses cached weights if available)
  5. Mini BoN run: 4 samples, 2 seeds, nguyen_1 on CPU/GPU

Exit 0 = ready to run. Exit 1 = fix issues before starting.
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
REINFORCEMENT_ROOT = REPO_ROOT / "2_training" / "reinforcement"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classes"))
sys.path.insert(0, str(REINFORCEMENT_ROOT))

PASS = []
FAIL = []
WARN = []


def check(name):
    def decorator(fn):
        def wrapper():
            try:
                msg = fn()
                PASS.append(name)
                suffix = f" ({msg})" if msg else ""
                print(f"  ✓ {name}{suffix}")
            except Warning as w:
                WARN.append((name, str(w)))
                print(f"  ⚠ {name}: {w}")
            except Exception as e:
                FAIL.append((name, traceback.format_exc()))
                print(f"  ✗ {name}: {e}")
        return wrapper
    return decorator


# ── 1. GPU ────────────────────────────────────────────────────────────────────

@check("GPU available")
def c_gpu():
    import torch
    if not torch.cuda.is_available():
        raise Warning("No GPU detected — running on CPU (will be ~10x slower)")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name}, {vram:.1f} GB VRAM"


@check("fp16 supported")
def c_fp16():
    import torch
    if not torch.cuda.is_available():
        raise Warning("CPU only — fp16 disabled, expect slow generation")
    props = torch.cuda.get_device_properties(0)
    if props.major < 7:
        raise Warning(f"GPU compute {props.major}.{props.minor} — fp16 may be slow")
    return f"compute {props.major}.{props.minor}"


# ── 2. Packages ───────────────────────────────────────────────────────────────

@check("transformers + peft installed")
def c_packages():
    import transformers
    import peft
    return f"transformers {transformers.__version__}, peft {peft.__version__}"


@check("project imports (rewards, expression, algorithms)")
def c_project_imports():
    from rewards import create_reward_with_penalty
    from classes.expression import Expression
    from algorithms.best_of_n import BestOfNBaseline, BoNConfig, run_best_of_n_baseline
    return "ok"


# ── 3. Data files ─────────────────────────────────────────────────────────────

@check("Feynman CSV is real data (not LFS pointer)")
def c_feynman_data():
    from utils.feynman_loader import load_benchmark_data
    import numpy as np

    t0 = time.time()
    data = load_benchmark_data("feynman_I_14_3", seed=42, test_fraction=0.25)
    x_train = data["train"]["x"]
    y_train = data["train"]["y"]
    elapsed = time.time() - t0

    assert x_train.dtype in (np.float32, np.float64), f"Bad dtype: {x_train.dtype}"
    assert not str(x_train.flat[0]).startswith("oid"), "LFS pointer detected in CSV!"
    assert x_train.shape[0] > 10, "Too few rows"

    return f"{x_train.shape[0]} train rows, {elapsed:.1f}s"


@check("Nguyen data generation works")
def c_nguyen_data():
    import sys
    sys.path.insert(0, str(REINFORCEMENT_ROOT))
    from run_experiment import generate_nguyen_data
    import numpy as np

    x, y, equation, valid_vars = generate_nguyen_data("nguyen_1", seed=42)
    assert np.isfinite(y).all(), "Non-finite values in nguyen_1"
    assert "x_1" in valid_vars
    return f"shape {x.shape}, eq={equation}"


# ── 4. Model loading ──────────────────────────────────────────────────────────

@check("HuggingFace model loads (gpt2-base infix)")
def c_model_load():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    MODEL_REPO = "augustocsc/gpt2_base_infix_682k"
    BASE_MODEL = "gpt2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, MODEL_REPO, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    elapsed = time.time() - t0

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{param_count:.0f}M params on {device}, {elapsed:.1f}s"


# ── 5. Mini BoN run ───────────────────────────────────────────────────────────

@check("Mini BoN run: 4 samples × 2 seeds, nguyen_1")
def c_bon_mini_run():
    import torch
    import numpy as np
    import tempfile
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from algorithms.best_of_n import run_best_of_n_baseline
    from rewards import create_reward_with_penalty
    from run_experiment import generate_nguyen_data

    MODEL_REPO = "augustocsc/gpt2_base_infix_682k"
    BASE_MODEL = "gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load model once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, MODEL_REPO, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    reward_fn, penalty_handler = create_reward_with_penalty("sr_ic", "gradient")

    best_r2 = -float("inf")
    t0 = time.time()

    for seed in [42, 123]:
        x_train, y_train, equation, valid_vars = generate_nguyen_data("nguyen_1", seed=seed)
        result = run_best_of_n_baseline(
            model_path=MODEL_REPO,
            base_model=BASE_MODEL,
            x=x_train,
            y=y_train,
            reward_fn=reward_fn,
            penalty_handler=penalty_handler,
            n_samples=4,
            batch_size=4,
            valid_variables=valid_vars,
            ground_truth=equation,
            use_wandb=False,
            model=model,
            tokenizer=tokenizer,
        )
        best_r2 = max(best_r2, result["best_r2"])

    elapsed = time.time() - t0
    throughput = 8 / elapsed  # 4 samples × 2 seeds

    return f"best_r2={best_r2:.3f}, {throughput:.1f} samples/s on {device}"


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  Colab Readiness Test — Phase 1b")
    print("═" * 60)

    checks = [
        c_gpu, c_fp16,
        c_packages, c_project_imports,
        c_feynman_data, c_nguyen_data,
        c_model_load,
        c_bon_mini_run,
    ]

    sections = [
        ("GPU", checks[0:2]),
        ("Packages", checks[2:4]),
        ("Data", checks[4:6]),
        ("Model", checks[6:7]),
        ("End-to-end", checks[7:]),
    ]

    for section, fns in sections:
        print(f"\n[{section}]")
        for fn in fns:
            fn()

    print("\n" + "═" * 60)
    print(f"  {len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed")
    print("═" * 60)

    if WARN:
        print("\nWarnings (non-fatal):")
        for name, msg in WARN:
            print(f"  ⚠ {name}: {msg}")

    if FAIL:
        print("\nFailed checks (fix before running):")
        for name, tb in FAIL:
            print(f"\n  ✗ {name}")
            for line in tb.strip().splitlines()[-5:]:
                print(f"    {line}")
        print()
        sys.exit(1)
    else:
        gpu_ok = not any("No GPU" in w for _, w in WARN)
        if gpu_ok:
            print("\n  ✅ Ready to run Phase 1b on GPU.")
        else:
            print("\n  ⚠  Ready to run, but NO GPU — reconnect Colab GPU runtime first.")
        sys.exit(0)


if __name__ == "__main__":
    main()
