"""
Coleta os resultados da Phase 1c e gera ZIP para download no Colab.
Cole e execute numa célula APÓS o daemon terminar (ou durante, para ver progresso parcial).

Uso:
    exec(open("/content/seriguela/experiments/download_phase1c.py").read())

O ZIP é salvo em /content/phase1c_results.zip
Em seguida o Colab abre o diálogo de download automaticamente.
"""

import json, zipfile, yaml
from pathlib import Path
from datetime import datetime

REPO      = Path("/content/seriguela")
RESULTS   = REPO / "results" / "phase_1c"

# ── 1. Status das filas (A + B separadas, com fallback para queue.yaml antigo) ─
from collections import Counter

def load_queue_status(queue_file):
    if not queue_file.exists():
        return []
    with open(queue_file) as f:
        return yaml.safe_load(f).get("queue", [])

exps_a = load_queue_status(REPO / "experiments" / "queue_1c_a.yaml")
exps_b = load_queue_status(REPO / "experiments" / "queue_1c_b.yaml")
all_exps = exps_a + exps_b

# Fallback: se as filas A/B não existirem, usa queue.yaml original
if not all_exps:
    exps_old = load_queue_status(REPO / "experiments" / "queue.yaml")
    all_exps = [e for e in exps_old if e.get("phase") == "1c"]

counts = Counter(e.get("status", "pending") for e in all_exps)
total   = len(all_exps)
done    = counts.get("done", 0)
failed  = counts.get("failed", 0)
pending = counts.get("pending", 0)

if exps_a or exps_b:
    ca = Counter(e.get("status", "pending") for e in exps_a)
    cb = Counter(e.get("status", "pending") for e in exps_b)
    print(f"Fila A: {dict(ca)}")
    print(f"Fila B: {dict(cb)}")
print(f"Phase 1c — {done}/{total} concluídos  |  {failed} falhas  |  {pending} pendentes")

# ── 2. Coletar JSONs ──────────────────────────────────────────────────────────
json_files = sorted(RESULTS.rglob("aggregate_*.json")) if RESULTS.exists() else []
print(f"JSONs encontrados: {len(json_files)}")

if not json_files:
    print("\n⚠️  Nenhum resultado encontrado ainda. Rode o daemon primeiro.")
    raise SystemExit(0)

# ── 3. Gerar ZIP ──────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M")
ZIP_PATH = Path(f"/content/phase1c_results_{ts}.zip")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in json_files:
        # Armazena como  phase_1c/<problem>/aggregate_*.json
        arcname = f.relative_to(REPO / "results")
        zf.write(f, arcname)

size_mb = ZIP_PATH.stat().st_size / 1e6
print(f"\nZIP criado: {ZIP_PATH}  ({size_mb:.1f} MB)")
print(f"Contém {len(json_files)} JSONs:\n")
for f in json_files[:8]:
    print(f"  {f.relative_to(RESULTS)}")
if len(json_files) > 8:
    print(f"  ... +{len(json_files)-8} mais")

# ── 4. Resumo rápido dos resultados ───────────────────────────────────────────
solved, r2_vals = 0, []
for jf in json_files:
    try:
        d = json.loads(jf.read_text())
        r2 = d.get("best_r2")
        if r2 is not None:
            r2_vals.append(r2)
            if r2 >= 0.999:
                solved += 1
    except Exception:
        pass

if r2_vals:
    import statistics
    print(f"\nResumo parcial ({len(r2_vals)} problemas com R²):")
    print(f"  Resolvidos (R²≥0.999): {solved}/{len(r2_vals)}  ({100*solved/len(r2_vals):.1f}%)")
    print(f"  Mediana R²: {statistics.median(r2_vals):.4f}")
    print(f"  Média   R²: {statistics.mean(r2_vals):.4f}")
    print(f"  Máx     R²: {max(r2_vals):.4f}")

# ── 5. Download automático ────────────────────────────────────────────────────
print(f"\nIniciando download de {ZIP_PATH.name} ...")
try:
    from google.colab import files
    files.download(str(ZIP_PATH))
    print("Download iniciado pelo navegador.")
except ImportError:
    print("(Não estamos no Colab — baixe manualmente o arquivo acima)")
except Exception as e:
    print(f"Download automático falhou: {e}")
    print(f"Baixe manualmente: {ZIP_PATH}")
