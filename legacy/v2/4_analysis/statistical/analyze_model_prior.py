#!/usr/bin/env python3
"""Fase 3 §4 — liga o prior do modelo SFT e o dataset 682K ao comportamento no RL.

Perguntas (ver docs/reports/PHASE3_ANALYSIS_PLAN.md):
  - O modelo SFT reproduz a distribuição do dataset (operadores, comprimento)?
  - O que o RL converge amplifica o prior ou luta contra ele?
  - A dificuldade por problema (nguyen_7 fácil, nguyen_9 difícil) é explicada
    pela frequência dos motivos no dataset (log frequente? 2-var raro?)

Roda LOCAL em CPU (notebook: 16GB RAM, sem GPU livre — ver CLAUDE.md):
  - modelo Base 124M + LoRA ≈ 550MB de download (cache HF), geração de N
    amostras em CPU ≈ minutos
  - dataset via HF streaming (não baixa os 682K, lê as primeiras --rows linhas)

Uso:
    python 4_analysis/statistical/analyze_model_prior.py --samples 256 --rows 20000
Saída: results/phase_3/model_prior.json + .md
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "phase_3"
MODEL_REPO = "augustocsc/gpt2_base_infix_682k"
DATASET = "augustocsc/sintetico_natural_prefix_682k"
PROMPT = '{"vars": ["x_1"], "ops": ["sin", "cos", "tan", "log", "sqrt", "exp", "+", "-", "*", "/", "**"], "cons": "C", "expr": "'
OPS = ["sin", "cos", "tan", "log", "sqrt", "exp", "**", "*", "/", "+", "-"]


def op_profile(exprs):
    """Frequência relativa de operadores + estatísticas de comprimento/aridade."""
    c = Counter()
    lens, nvars = [], []
    for e in exprs:
        if not e:
            continue
        for op in OPS:
            c[op] += e.count(op) if len(op) > 1 else len(re.findall(re.escape(op), e))
        lens.append(len(re.findall(r"[a-zA-Z_]+\d*|\*\*|[-+*/()]|\d+\.?\d*", e)))
        nvars.append(len(set(re.findall(r"x_\d+", e))))
    tot = sum(c.values()) or 1
    return {
        "op_freq": {k: round(v / tot, 4) for k, v in c.most_common()},
        "mean_len": round(sum(lens) / max(len(lens), 1), 1),
        "nvars_dist": dict(Counter(nvars)),
        "n": len(lens),
    }


def extract_expr(text: str):
    m = re.search(r'"expr":\s*"([^"]*)', text)
    return m.group(1) if m else None


def sample_model(n: int, batch: int = 32):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print(f"Carregando {MODEL_REPO} (CPU)...")
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained("gpt2"), MODEL_REPO).eval()
    exprs, valid = [], 0
    from classes.expression import Expression  # validade com o validador do projeto
    with torch.no_grad():
        ids = tok(PROMPT, return_tensors="pt")
        for i in range(0, n, batch):
            b = min(batch, n - i)
            out = model.generate(
                ids.input_ids.repeat(b, 1), attention_mask=ids.attention_mask.repeat(b, 1),
                do_sample=True, temperature=1.0, max_new_tokens=50,
                pad_token_id=tok.eos_token_id)
            for row in out:
                e = extract_expr(tok.decode(row, skip_special_tokens=True))
                if e:
                    exprs.append(e)
                    try:
                        Expression.parse_infix(e.replace("C", "1")).validate()
                        valid += 1
                    except Exception:  # noqa: BLE001
                        pass
            print(f"  {min(i+b, n)}/{n} amostras...", flush=True)
    return exprs, valid


def dataset_profile(rows: int):
    from datasets import load_dataset
    print(f"Streaming {DATASET} (primeiras {rows} linhas)...")
    ds = load_dataset(DATASET, split="train", streaming=True)
    exprs = []
    for i, row in enumerate(ds):
        if i >= rows:
            break
        e = extract_expr(row.get("i_prompt_n") or "")
        if e:
            exprs.append(e)
    return exprs


def main():
    import sys
    sys.path.insert(0, str(REPO))
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=256, help="amostras zero-shot do modelo")
    ap.add_argument("--rows", type=int, default=20000, help="linhas do dataset (streaming)")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    if not args.skip_dataset:
        ds_exprs = dataset_profile(args.rows)
        report["dataset"] = op_profile(ds_exprs)
        print(f"[dataset] {report['dataset']['n']} expressões; "
              f"len médio {report['dataset']['mean_len']}; "
              f"top ops: {list(report['dataset']['op_freq'])[:5]}")

    if not args.skip_model:
        exprs, valid = sample_model(args.samples)
        report["model_prior"] = op_profile(exprs)
        report["model_prior"]["valid_rate"] = round(valid / max(len(exprs), 1), 3)
        print(f"[modelo] {len(exprs)} expressões; validade {report['model_prior']['valid_rate']}; "
              f"top ops: {list(report['model_prior']['op_freq'])[:5]}")

    # Vencedores do RL (das tabelas da Fase 2, se existirem localmente)
    winners = []
    p2 = REPO / "results" / "phase_2" / "base_infix"
    for f in p2.rglob("results_latest.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
            if (d.get("best_r2") or 0) > 0.9:
                winners.append(d.get("best_expression", ""))
        except Exception:  # noqa: BLE001
            pass
    if winners:
        report["rl_winners"] = op_profile(winners)
        print(f"[RL winners >0.9] {len(winners)} expressões; "
              f"top ops: {list(report['rl_winners']['op_freq'])[:5]}")

    (OUT / "model_prior.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Prior do modelo × dataset × vencedores do RL", ""]
    for k, v in report.items():
        lines += [f"## {k}", "```json", json.dumps(v, indent=2), "```", ""]
    (OUT / "model_prior.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ok] {OUT/'model_prior.json'} e .md")


if __name__ == "__main__":
    main()
