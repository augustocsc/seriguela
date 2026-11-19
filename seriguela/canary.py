"""Portão G1: canário. Roda todos os braços por poucos passos e decide PASSA/FALHA.

    python -m seriguela.canary --problem nguyen_1 --steps 10 --batch 128

Passa se, em todos os braços: nenhum tripwire dispara, o orçamento fecha, o registro
é válido e a mesma seed reproduz o primeiro passo. Sai com código 0 (passa) ou 1.
Custa segundos por braço; é o que deve rodar no pod antes de qualquer campanha.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple

from .trainer import PRESETS, RunConfig, TripwireError, run, validate

ARMS: List[Tuple[str, str]] = [("bon", "none")] + [
    (algo, replay) for algo in PRESETS if algo != "bon" for replay in ("none", "elite", "random")]


def check_arm(algo: str, replay: str, args, policy_factory=None, problem=None) -> Tuple[bool, str]:
    def once(steps: int):
        cfg = RunConfig(problem=args.problem, seed=args.seed, algo=algo, replay=replay, steps=steps,
                        batch=args.batch, update_mb=args.update_mb, model=args.model, ref_check_every=2)
        return run(cfg, policy_factory() if policy_factory else None, problem, allow_dirty=args.allow_dirty)

    started = time.time()
    try:
        result = once(args.steps)
        validate(result)
        again = once(1)
    except TripwireError as error:
        return False, f"tripwire: {error}"
    first, repeat = result["history"][0], again["history"][0]
    for key in ("mean_reward", "valid_rate", "unique_frac"):
        if first[key] != repeat[key]:
            return False, f"não reproduz o 1º passo ({key}: {first[key]} != {repeat[key]})"
    if result["status"] != "ok":
        return False, f"status {result['status']}"
    learned = any(h.get("updated") for h in result["history"])
    if (algo != "bon") != learned:
        return False, "braço que aprende não atualizou" if algo != "bon" else "bon atualizou pesos"
    drift = max(h["sample_drift"] for h in result["history"])
    per_step = (time.time() - started) / (args.steps + 1)
    best = result["best"] or {}
    return True, (f"{per_step:5.1f} s/passo  desvio amostragem {drift:.1e}  "
                  f"KL do prior {result['final_policy']['kl_from_prior_per_token']:+.4f}  "
                  f"melhor R² {best.get('train_r2', float('nan')):.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", default="nguyen_1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--update_mb", type=int, default=32)
    parser.add_argument("--model", default=RunConfig.model)
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args(argv)
    failures = 0
    for algo, replay in ARMS:
        ok, detail = check_arm(algo, replay, args)
        failures += not ok
        name = algo if replay == "none" else f"{algo}+{replay}"
        print(f"{'PASSA' if ok else 'FALHA'}  {name:<16} {detail}", flush=True)
    print(f"\nG1: {'PASSOU' if not failures else f'FALHOU ({failures} braço(s))'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
