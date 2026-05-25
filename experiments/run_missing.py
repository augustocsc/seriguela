"""
Roda os 2 experimentos faltantes da Phase 1b e salva em ZIP.
Cole e execute numa célula do Colab.
"""

import subprocess, shutil, sys
from pathlib import Path

REPO = Path("/content/seriguela")
PYTHON = sys.executable

MISSING = [
    {
        "problem": "feynman_I_34_10",
        "output_dir": REPO / "results/phase_1b/feynman_I_34_10",
    },
    {
        "problem": "feynman_II_11_17",
        "output_dir": REPO / "results/phase_1b/feynman_II_11_17",
    },
]

SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
COMMON = dict(
    algorithm="best_of_n",
    model="augustocsc/gpt2_base_infix_682k",
    reward="sr_ic",
    penalty="gradient",
    max_steps=1,
    batch_size=512,
)

for exp in MISSING:
    problem = exp["problem"]
    out = exp["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    cmd = [PYTHON, str(REPO / "2_training/reinforcement/run_experiment.py"),
           "--algorithm", COMMON["algorithm"],
           "--model",     COMMON["model"],
           "--problem",   problem,
           "--reward",    COMMON["reward"],
           "--penalty",   COMMON["penalty"],
           "--max_steps", str(COMMON["max_steps"]),
           "--batch_size",str(COMMON["batch_size"]),
           "--seeds",     *[str(s) for s in SEEDS],
           "--no_wandb",
           "--output_dir", str(out)]

    print(f"\n{'='*60}")
    print(f"Rodando: {problem}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode != 0:
        print(f"FALHOU: {problem} (exit {result.returncode})")
    else:
        print(f"OK: {problem}")

# Zip só as 2 pastas de resultado
zip_src = REPO / "results/phase_1b"
zip_out = Path("/content/missing_phase1b")
shutil.make_archive(str(zip_out), "zip", str(zip_src.parent),
                    base_dir="phase_1b/" + MISSING[0]["problem"].replace("feynman_", "feynman_") )

# Zip com as duas pastas juntas
import zipfile
zip_path = Path("/content/missing_phase1b.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for exp in MISSING:
        folder = exp["output_dir"]
        for f in folder.rglob("*.json"):
            zf.write(f, f.relative_to(REPO / "results"))

print(f"\nZIP salvo em: {zip_path}")
print("Conteúdo:")
with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
        print(f"  {name}")
print("\nBaixe o ZIP e extraia dentro da sua pasta results/")
