"""Dedicated runner for remaining pre-phase B tests."""
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

EXPERIMENT_RUNNER = "2_training/reinforcement/run_experiment.py"

BASE_CONFIG = {
    "model": "augustocsc/gpt2_base_infix_682k",
    "reward": "r2_clipped",
    "penalty": "gradient",
    "algorithm": "bon_ppo"
}

def build_command(job: dict, output_dir: str, batch_size: int) -> list:
    """Build run_experiment.py command for a single job."""
    return [
        sys.executable, EXPERIMENT_RUNNER,
        "--algorithm", job.get("algorithm", BASE_CONFIG["algorithm"]),
        "--model", BASE_CONFIG["model"],
        "--reward", BASE_CONFIG["reward"],
        "--penalty", BASE_CONFIG["penalty"],
        "--temperature", job["temperature"],
        "--problem", job["problem"],
        "--max_steps", str(job.get("max_steps", 50)),
        "--batch_size", str(batch_size),
        "--patience", "999",
        "--seeds", str(job["seed"]),
        "--output_dir", output_dir,
        "--no_wandb",
    ]

def get_jobs():
    jobs = []
    
    # --- TEST 4 REMAINING (Runs 11 to 18) ---
    # Runs 1-10 were completed. Here is what is missing to finish test4:
    # 11: nguyen_9 | linear_annealing | seed=123
    jobs.append({"label": "test4_resume | nguyen_9 | linear_annealing | seed=123", "problem": "nguyen_9", "temperature": "linear_annealing", "seed": 123})
    # 12: nguyen_9 | linear_annealing | seed=456
    jobs.append({"label": "test4_resume | nguyen_9 | linear_annealing | seed=456", "problem": "nguyen_9", "temperature": "linear_annealing", "seed": 456})
    # 13-18: fixed_0.9 for both problems across 3 seeds
    for problem in ["nguyen_5", "nguyen_9"]:
        for seed in [42, 123, 456]:
            jobs.append({
                "label": f"test4_resume | {problem} | fixed_0.9 | seed={seed}",
                "problem": problem,
                "temperature": "fixed_0.9",
                "seed": seed
            })

    # --- TEST 5 (All 36 algorithm comparison runs) ---
    for algo in ["bon_ppo", "bon_grpo", "pure_ppo", "pure_grpo"]:
        for problem in ["nguyen_1", "nguyen_5", "nguyen_9"]:
            for seed in [42, 123, 456]:
                jobs.append({
                    "label": f"test5 | {algo} | {problem} | seed={seed}",
                    "algorithm": algo,
                    "problem": problem,
                    "temperature": "cosine_annealing",
                    "seed": seed
                })
    
    return jobs

def main():
    output_dir = "/content/drive/MyDrive/seriguela_results/pre_phase_b"
    batch_size = 2048
    jobs = get_jobs()
    total = len(jobs)
    
    print("=" * 65)
    print("  RESUME TEST4 + RUN TEST5 ALGORITHM COMPARISON")
    print(f"  Total Jobs: {total} | Batch: {batch_size}")
    print("=" * 65)

    start = datetime.now()
    
    for i, job in enumerate(jobs):
        cmd = build_command(job, output_dir, batch_size)
        prefix = f"[{i+1}/{total}]"
        print(f"\n{prefix} START  {job['label']}", flush=True)
        
        t0 = time.time()
        try:
            # Create isolated log file
            log_file = Path(output_dir) / f"log_{job.get('algorithm', 'bon_ppo')}_{job['problem']}_{job['temperature']}_seed{job['seed']}.txt"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, "w") as lf:
                process = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            elapsed = (time.time() - t0) / 60
            print(f"{prefix} DONE   {job['label']}  ({elapsed:.1f} min)  [ok]", flush=True)
            
        except subprocess.CalledProcessError as e:
            elapsed = (time.time() - t0) / 60
            print(f"{prefix} ERROR  {job['label']}  ({elapsed:.1f} min)  Exit={e.returncode}", flush=True)
            print(f"         Check log: {log_file}", flush=True)
            # Continue running despite errors
        except KeyboardInterrupt:
            print(f"\n{prefix} INTERRUPTED by user", flush=True)
            break

    dur = datetime.now() - start
    print("\n" + "=" * 65)
    print(f"  FINISHED in {dur}")
    print("=" * 65)

if __name__ == "__main__":
    main()
