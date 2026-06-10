# Pre-Phase B Validation Tests

Targeted tests to validate the Fast Phase A winner config before launching Phase B.

## Winner Config (from Fast Phase A)
| Parameter | Value |
|:---|:---|
| Algorithm | `bon_ppo` |
| Reward | `r2_clipped` |
| Temperature | `cosine_annealing` (1.0→0.5) |
| Penalty | `gradient` |
| Model | `augustocsc/gpt2_base_infix_682k` |

## Test Suite

| Test | Script | What it answers | Runs | Est. Time |
|:---|:---|:---|:---:|:---:|
| 1. Multi-Seed | `test1_multi_seed.py` | Is the winner robust across seeds? | 9 | ~3h |
| 2. Nguyen-5 Debug | `test2_nguyen5_debug.py` | Why did Nguyen-5 fail at 50 steps? | 6 | ~2h |
| 3. Convergence 200 | `test3_convergence_200.py` | What does the learning curve look like? | 3 | ~3h |
| 4. Temperature Comparison | `test4_temp_compare.py` | Does cosine beat other schedules at 50 steps? | 9 | ~3h |

**Total: 27 runs, ~11 hours across 2 machines**

## Machine Assignment

- **Notebook (WSL):** Tests 1 + 3 (~6h)
- **Google Colab:** Tests 2 + 4 (~5h)

## Setup

### On Notebook (WSL)
```bash
# From project root
cd /mnt/c/Users/Rudá/seriguela
source ~/venv_seriguela/bin/activate

# Run tests
python scripts/pre_phase_b/test1_multi_seed.py
python scripts/pre_phase_b/test3_convergence_200.py
```

### On Google Colab
```python
# Cell 1: Mount Google Drive (results persist even if session disconnects)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone and install
!git clone https://github.com/augustocsc/seriguela.git
%cd seriguela
!pip install -r requirements.txt

# Cell 3: Run in PARALLEL (4 experiments at once, batch=2048, saved to Drive)
!python scripts/pre_phase_b/run_parallel.py \\
    --test test2 \\
    --output_dir /content/drive/MyDrive/seriguela_results/pre_phase_b \\
    --max_parallel 4 \\
    --batch_size 2048

# Cell 4: Or run sequentially if parallel causes OOM
!python scripts/pre_phase_b/test2_nguyen5_debug.py \\
    --output_dir /content/drive/MyDrive/seriguela_results/pre_phase_b \\
    --batch_size 2048
```

> **Why Drive matters:** Each completed experiment writes its JSON to Drive immediately.
> If the Colab session disconnects, only the currently-running experiment is lost.
> On reconnect, `run_parallel.py` skips already-completed experiments automatically.

### Dry Run (verify commands without executing)
```bash
python scripts/pre_phase_b/test1_multi_seed.py --dry_run
```

## After All Tests Complete

```bash
python scripts/pre_phase_b/analyze_results.py
```

This reads all JSONs from `results/pre_phase_b/` and produces a go/no-go summary.
