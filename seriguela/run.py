"""Linha de comando: um run por chamada.

    python -m seriguela.run --problem nguyen_5 --algo ppo --replay elite --seed 42

Grava ``<out>/<problema>/<braço>/seed_<s>.json``. Um run já gravado é pulado (use
``--force`` para refazer). Se um tripwire dispara, grava ``seed_<s>.failed.json`` e
sai com código 2: quem orquestra a campanha deve parar ali.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import typing
from pathlib import Path

from .provenance import DirtyTreeError
from .trainer import RunConfig, TripwireError, run, save


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hints = typing.get_type_hints(RunConfig)
    for f in dataclasses.fields(RunConfig):
        kind = hints[f.name]
        if typing.get_origin(kind) is typing.Union:          # Optional[T]
            kind = next(a for a in typing.get_args(kind) if a is not type(None))
        parser.add_argument(f"--{f.name}", type=kind, default=f.default)
    parser.add_argument("--out", default="results/v3", help="diretório de saída")
    parser.add_argument("--force", action="store_true", help="refaz mesmo se o resultado existe")
    parser.add_argument("--allow_dirty", action="store_true", help="aceita árvore git suja (só para testes)")
    return parser


def main(argv=None) -> int:
    args = vars(_parser().parse_args(argv))
    out, force, allow_dirty = Path(args.pop("out")), args.pop("force"), args.pop("allow_dirty")
    config = RunConfig(**args)
    path = out / config.problem / config.arm / f"seed_{config.seed}.json"
    if path.exists() and not force:
        print(f"já existe, pulando: {path}")
        return 0
    try:
        result = run(config, allow_dirty=allow_dirty, verbose=True)
    except (TripwireError, DirtyTreeError) as error:
        path.parent.mkdir(parents=True, exist_ok=True)
        failed = path.with_name(f"seed_{config.seed}.failed.json")
        failed.write_text(json.dumps({"error": type(error).__name__, "message": str(error),
                                      "config": dataclasses.asdict(config)}, indent=1), encoding="utf-8")
        print(f"FALHOU ({type(error).__name__}): {error}", file=sys.stderr)
        return 2
    save(result, path)
    best = result["best"] or {}
    print(f"ok: {path}\n  melhor: {best.get('expression')}  R² treino={best.get('train_r2')} "
          f"teste={best.get('test_r2')} simbólico={best.get('symbolic_match')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
