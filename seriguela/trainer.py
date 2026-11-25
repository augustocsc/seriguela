"""Um laço de busca para todos os braços.

Cada passo: amostra ``batch`` expressões novas, avalia no treino e (se o braço
aprende) atualiza a política. Os braços diferem só na configuração:

==========  ==========  ===========  ======================================
braço       vantagem    objetivo     regularizador
==========  ==========  ===========  ======================================
``bon``     --          --           (não atualiza: amostragem pura)
``ppo``     z-score     recortado    bônus de entropia
``grpo``    z-score     recortado    KL contra o prior do SFT (DeepSeekMath)
``rank_pg`` postos      REINFORCE    bônus de entropia (o "GRPO" da v2, com o nome certo)
``rspg``    quantil     REINFORCE    bônus de entropia (busca de risco, DSR)
==========  ==========  ===========  ======================================

e em ``replay``: ``none``, ``elite`` (as melhores) ou ``random`` (controle).
O orçamento é sempre ``steps * batch`` amostras novas, em todos os braços.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import advantages, provenance
from .buffer import EliteBuffer, ReservoirBuffer
from .data import Problem, load_problem
from .expr import evaluators_agree
from .policy import Policy, load_policy
from .reward import Eval, Evaluator
from .symbolic import symbolic_match

SCHEMA_VERSION = 3
SOLVED_R2 = 0.999

PRESETS: Dict[str, Dict] = {
    "bon":     dict(advantage=None,     objective=None,        epochs=0, entropy_coef=0.0,   kl_ref_coef=0.0),
    "ppo":     dict(advantage="zscore", objective="clip",      epochs=2, entropy_coef=0.01,  kl_ref_coef=0.0),
    "grpo":    dict(advantage="zscore", objective="clip",      epochs=2, entropy_coef=0.0,   kl_ref_coef=0.04),
    "rank_pg": dict(advantage="rank",   objective="reinforce", epochs=1, entropy_coef=0.01,  kl_ref_coef=0.0),
    "rspg":    dict(advantage="risk",   objective="reinforce", epochs=1, entropy_coef=0.005, kl_ref_coef=0.0),
}


class TripwireError(RuntimeError):
    """Uma invariante do experimento foi violada: o run para e não produz resultado."""


@dataclass
class RunConfig:
    problem: str = "nguyen_5"
    seed: int = 0
    algo: str = "ppo"
    replay: str = "none"
    steps: int = 200
    batch: int = 1024
    model: str = "augustocsc/gpt2_base_infix_682k"
    base_model: str = "gpt2"
    # temperatura: cosseno de t_max a t_min ao longo dos passos (iguais = fixa)
    t_max: float = 1.0
    t_min: float = 0.5
    # recompensa e dados
    reward: str = "r2"
    const_mode: str = "one"
    invalid_reward: float = 0.0
    nguyen_variant: str = "canonical"
    test_scale: float = 1.0
    max_train: int = 10000
    noise: float = 0.0
    # otimização (None = valor do preset do braço)
    lr: float = 5e-6
    epochs: Optional[int] = None
    entropy_coef: Optional[float] = None
    kl_ref_coef: Optional[float] = None
    clip_eps: float = 0.2
    max_kl: float = 0.02
    max_grad_norm: float = 0.5
    risk_epsilon: float = 0.05
    update_mb: int = 64
    # replay
    replay_ratio: float = 0.2
    buffer_size: int = 1000
    # política
    lora_r: int = 8
    lora_alpha: int = 16
    max_new_tokens: int = 64
    gen_batch: int = 256
    # guardas
    sample_tol: float = 1e-3
    validity_floor: float = 0.02
    validity_patience: int = 20
    ref_check_every: int = 10
    ref_check_n: int = 5
    # relatório
    final_samples: int = 0          # 0 = igual a batch
    top_k: int = 50

    def resolved(self) -> "RunConfig":
        if self.algo not in PRESETS:
            raise ValueError(f"braço desconhecido: {self.algo} (opções: {sorted(PRESETS)})")
        if self.replay not in ("none", "elite", "random"):
            raise ValueError(f"replay desconhecido: {self.replay}")
        preset = PRESETS[self.algo]
        fill = {k: preset[k] for k in ("epochs", "entropy_coef", "kl_ref_coef") if getattr(self, k) is None}
        return dataclasses.replace(self, **fill)

    @property
    def arm(self) -> str:
        return self.algo if self.replay == "none" else f"{self.algo}+{self.replay}"


def temperature(step: int, cfg: RunConfig) -> float:
    progress = min(step / max(cfg.steps, 1), 1.0)
    return cfg.t_min + 0.5 * (cfg.t_max - cfg.t_min) * (1 + math.cos(math.pi * progress))


def _better(a: Eval, b: Optional[Eval]) -> bool:
    """Seleção SEMPRE pelo treino: maior recompensa; no empate, a mais simples."""
    return b is None or (a.reward, -a.complexity) > (b.reward, -b.complexity)


def _clean(value):
    """Torna o resultado serializável em JSON estrito (sem NaN/inf)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _update(policy: Policy, optimizer, cfg: RunConfig, sequences, rewards, n_fresh: int, temp: float) -> Dict:
    """Uma atualização da política sobre o lote (novas + replay).

    As estatísticas da vantagem (média/desvio, postos ou quantil de risco) são
    calculadas APENAS sobre as ``n_fresh`` amostras novas. Na loss, cada amostra
    nova tem peso ``1 / n_fresh_kept`` e cada amostra de replay tem peso
    ``replay_ratio / n_rep_kept`` (quando há replay mantido), de modo que o
    buffer nunca sequestre o gradiente on-policy (em especial no ``rspg``).
    """
    preset = PRESETS[cfg.algo]
    adv, keep = advantages.compute_with_replay(preset["advantage"], np.asarray(rewards), n_fresh, cfg.risk_epsilon)
    if not np.all(np.isfinite(adv)):
        raise TripwireError("vantagem não finita")
    idx = np.flatnonzero(keep)
    stats = {"loss": 0.0, "approx_kl": 0.0, "kl_ref": 0.0, "entropy": 0.0, "grad_norm": 0.0,
             "epochs_used": 0, "update_tokens": 0, "updated": False, "signal": False}
    if len(idx) == 0 or not np.any(adv[idx] != 0):
        return stats                                 # lote sem sinal: nada a aprender
    stats["signal"] = True
    n_fresh_kept = int(np.sum(idx < n_fresh))
    n_rep_kept = int(np.sum(idx >= n_fresh))
    w_fresh = 1.0 / max(n_fresh_kept, 1)
    w_rep = (cfg.replay_ratio / n_rep_kept) if n_rep_kept > 0 else 0.0
    total_weight = (1.0 if n_fresh_kept > 0 else 0.0) + (cfg.replay_ratio if n_rep_kept > 0 else 0.0)
    device = policy.device
    params = policy.trainable_parameters()
    clip = preset["objective"] == "clip"
    ref = policy.score([sequences[i] for i in idx], temp, ref=True, batch=cfg.update_mb) \
        if cfg.kl_ref_coef > 0 else None
    old: Dict[int, torch.Tensor] = {}
    for epoch in range(cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "kl": 0.0, "kl_ref": 0.0, "entropy": 0.0, "tokens": 0.0}
        for start in range(0, len(idx), cfg.update_mb):
            mb = idx[start:start + cfg.update_mb]
            logp, mask, ent = policy.log_probs([sequences[i] for i in mb], temp, grad=True,
                                               entropy=cfg.entropy_coef > 0)
            m = mask.float()
            lengths = m.sum(1)
            a = torch.tensor(adv[mb], dtype=torch.float32, device=device)
            is_fresh = torch.tensor(mb < n_fresh, dtype=torch.bool, device=device)
            weights = torch.where(is_fresh, torch.full_like(a, w_fresh), torch.full_like(a, w_rep))
            if epoch == 0:
                old[start] = logp.detach()
            if clip:
                ratio = torch.exp(logp - old[start])
                surrogate = torch.minimum(ratio * a[:, None],
                                          ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a[:, None])
                per_seq = -((surrogate * m).sum(1) / lengths)
            else:
                per_seq = -(a * (logp * m).sum(1) / lengths)
            loss = (per_seq * weights).sum()
            if cfg.entropy_coef > 0:
                entropy = (((ent * m).sum(1) / lengths) * weights).sum()
                loss = loss - cfg.entropy_coef * entropy
                totals["entropy"] += float(entropy.detach())
            if ref is not None:
                fresh_f = is_fresh.float()
                ref_mb = torch.zeros_like(logp)
                for row, i in enumerate(range(start, start + len(mb))):
                    ref_mb[row, :len(ref[i])] = torch.tensor(ref[i], device=device)
                delta = (ref_mb - logp) * m           # estimador do DeepSeekMath, eq. 4
                kl_ref = (((torch.exp(delta) - delta - 1) * m).sum(1) / lengths * fresh_f * w_fresh).sum()
                loss = loss + cfg.kl_ref_coef * kl_ref
                totals["kl_ref"] += float(kl_ref.detach())
            (loss / total_weight).backward()
            totals["loss"] += float(loss.detach())
            if clip:
                d = (logp.detach() - old[start]) * m  # estimador k3 não-negativo de Schulman
                totals["kl"] += float(((torch.exp(d) - d - 1) * m).sum())
            totals["tokens"] += float(m.sum())
        approx_kl = totals["kl"] / max(totals["tokens"], 1.0)
        if epoch > 0 and approx_kl > cfg.max_kl:
            break                                    # região de confiança: não aplica esta época
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        if not math.isfinite(grad_norm) or not math.isfinite(totals["loss"]):
            raise TripwireError("loss ou gradiente não finito")
        optimizer.step()
        stats.update(loss=totals["loss"] / total_weight, approx_kl=approx_kl,
                     kl_ref=totals["kl_ref"] / total_weight, entropy=totals["entropy"] / total_weight,
                     grad_norm=grad_norm, epochs_used=epoch + 1, updated=True)
        stats["update_tokens"] += int(totals["tokens"])
    return stats


def run(config: RunConfig, policy: Optional[Policy] = None, problem: Optional[Problem] = None,
        allow_dirty: bool = False, verbose: bool = False) -> Dict:
    """Executa um run e devolve o registro de resultado (esquema v3)."""
    cfg = config.resolved()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 3])
    problem = problem or load_problem(cfg.problem, cfg.seed, nguyen_variant=cfg.nguyen_variant,
                                      test_scale=cfg.test_scale, max_train=cfg.max_train, noise=cfg.noise)
    prov = provenance.collect(dataclasses.asdict(cfg), problem.data_hash(), allow_dirty)
    evaluator = Evaluator(problem, cfg.reward, cfg.const_mode, cfg.invalid_reward)
    policy = policy or load_policy(cfg.model, problem.n_vars, cfg.base_model, lora_r=cfg.lora_r,
                                   lora_alpha=cfg.lora_alpha, max_new_tokens=cfg.max_new_tokens,
                                   gen_batch=cfg.gen_batch)
    learn = cfg.algo != "bon"
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.lr, weight_decay=0.0) if learn else None
    buffer = {"none": None, "elite": EliteBuffer(cfg.buffer_size),
              "random": ReservoirBuffer(cfg.buffer_size)}[cfg.replay] if learn else None

    best: Optional[Eval] = None
    best_step = -1
    history: List[Dict] = []
    samples_total = update_tokens = low_validity = 0
    status = "ok"
    started = time.time()

    for step in range(cfg.steps):
        tick = time.time()
        temp = temperature(step, cfg)
        samples = policy.sample(cfg.batch, temp)
        samples_total += len(samples)
        if samples_total != (step + 1) * cfg.batch:
            raise TripwireError(f"orçamento: {samples_total} amostras, esperado {(step + 1) * cfg.batch}")

        # A distribuição amostrada é a mesma que a atualização vai diferenciar?
        probe = samples[:min(64, len(samples))]
        rescored = policy.score([s.tokens for s in probe], temp)
        drift = max(float(np.max(np.abs(a - s.logp))) for a, s in zip(rescored, probe))
        if drift > cfg.sample_tol:
            raise TripwireError(f"log-prob da amostragem difere da pontuação em {drift:.2e}")

        evals = [evaluator(s.expression) for s in samples]
        for ev in evals:
            if ev.valid and _better(ev, best):
                best, best_step = ev, step
        if step % cfg.ref_check_every == 0:          # o avaliador rápido está certo?
            top = sorted({e.expression: e for e in evals if e.valid}.values(), key=lambda e: -e.reward)
            for ev in top[:cfg.ref_check_n]:
                if not evaluators_agree(ev.expression, problem.x_train, ev.constants):
                    raise TripwireError(f"avaliador rápido e de referência divergem em: {ev.expression}")

        update = {"updated": False, "signal": False}
        n_replay = 0
        if learn:
            sequences = [s.tokens for s in samples]
            rewards = [e.reward for e in evals]
            if buffer is not None:                   # replay vem de passos ANTERIORES
                entries = buffer.sample(int(cfg.batch * cfg.replay_ratio), rng)
                n_replay = len(entries)
                sequences += [e.tokens for e in entries]
                rewards += [e.reward for e in entries]
                for s, ev in zip(samples, evals):
                    if ev.valid and s.expression is not None:
                        if buffer.kind == "elite":
                            buffer.add(ev.expression, s.tokens, ev.reward, ev.r2)
                        else:
                            buffer.add(ev.expression, s.tokens, ev.reward, ev.r2, rng)
            update = _update(policy, optimizer, cfg, sequences, rewards, len(samples), temp)
            update_tokens += update.pop("update_tokens")

        rewards_fresh = np.array([e.reward for e in evals])
        valid_rate = float(np.mean([e.valid for e in evals]))
        history.append({
            "step": step, "temperature": temp, "valid_rate": valid_rate,
            "mean_reward": float(rewards_fresh.mean()), "max_reward": float(rewards_fresh.max()),
            "zero_reward_frac": float(np.mean(rewards_fresh == 0)),
            "unique_frac": len({s.tokens for s in samples}) / len(samples),
            "best_r2": best.r2 if best else None, "best_reward": best.reward if best else None,
            "n_replay": n_replay, "samples_total": samples_total, "unique_evals": evaluator.unique,
            "update_tokens": update_tokens, "sample_drift": drift,
            "seconds": time.time() - tick, **update,
        })
        if verbose and (step % 10 == 0 or step == cfg.steps - 1):
            print(f"[{cfg.problem} {cfg.arm} s{cfg.seed}] passo {step:4d} T={temp:.2f} "
                  f"válidas={valid_rate:.0%} recompensa={rewards_fresh.mean():.3f} "
                  f"melhor R²={best.r2 if best else float('nan'):.4f} ({time.time() - tick:.1f}s)", flush=True)
        low_validity = low_validity + 1 if valid_rate < cfg.validity_floor else 0
        if low_validity >= cfg.validity_patience:
            status = "collapsed"                     # registrado, não escondido
            break

    return _finalize(cfg, prov, policy, problem, evaluator, best, best_step, history, status,
                     samples_total, update_tokens, time.time() - started)


def _finalize(cfg, prov, policy, problem, evaluator, best, best_step, history, status,
              samples_total, update_tokens, seconds) -> Dict:
    unique_evals = evaluator.unique
    found = [e for e in evaluator._cache.values() if e.valid]
    found.sort(key=lambda e: (-e.reward, e.complexity))
    # Política final: o que ela produz sozinha, sem buffer (separa "achou" de "aprendeu").
    temp = temperature(cfg.steps, cfg)
    final = policy.sample(cfg.final_samples or cfg.batch, temp)
    final_evals = [evaluator(s.expression) for s in final]
    now = policy.score([s.tokens for s in final], temp)
    prior = policy.score([s.tokens for s in final], temp, ref=True)
    tokens = sum(len(s.tokens) for s in final)
    kl_from_prior = float(sum(float(np.sum(a - b)) for a, b in zip(now, prior)) / max(tokens, 1))
    final_policy = {
        "samples": len(final), "temperature": temp,
        "valid_rate": float(np.mean([e.valid for e in final_evals])),
        "mean_reward": float(np.mean([e.reward for e in final_evals])),
        "solved_frac": float(np.mean([e.valid and e.r2 >= SOLVED_R2 for e in final_evals])),
        "unique_frac": len({s.tokens for s in final}) / len(final),
        "kl_from_prior_per_token": kl_from_prior,
    }
    best_record = None
    if best is not None:                             # o teste é lido aqui, uma única vez
        match = symbolic_match(best.expression, problem.truth, best.constants, problem.x_train) \
            if problem.truth else None
        best_record = {"expression": best.expression, "constants": list(best.constants),
                       "train_r2": best.r2, "train_reward": best.reward, "complexity": best.complexity,
                       "step_found": best_step, "test_r2": evaluator.test_r2(best),
                       "symbolic_match": match}
    prov["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return _clean({
        "schema_version": SCHEMA_VERSION, "status": status, "arm": cfg.arm,
        "problem": cfg.problem, "seed": cfg.seed, "truth": problem.truth,
        "provenance": prov, "config": dataclasses.asdict(cfg), "problem_meta": problem.meta,
        "counters": {"steps": len(history), "samples": samples_total, "unique_evals": unique_evals,
                     "update_tokens": update_tokens, "seconds": seconds},
        "best": best_record, "final_policy": final_policy,
        "top": [{"expression": e.expression, "train_r2": e.r2, "train_reward": e.reward,
                 "complexity": e.complexity} for e in found[:cfg.top_k]],
        "history": history,
    })


REQUIRED = {
    "schema_version": int, "status": str, "arm": str, "problem": str, "seed": int,
    "provenance": dict, "config": dict, "counters": dict, "final_policy": dict, "top": list, "history": list,
}
REQUIRED_PROVENANCE = ("commit", "dirty", "config_hash", "data_hash", "versions", "gpu", "started", "finished")
REQUIRED_COUNTERS = ("steps", "samples", "unique_evals", "update_tokens", "seconds")


def validate(result: Dict) -> None:
    """Levanta ``ValueError`` se o registro não segue o esquema v3."""
    for key, kind in REQUIRED.items():
        if not isinstance(result.get(key), kind):
            raise ValueError(f"campo ausente ou de tipo errado: {key}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version diferente de 3")
    for key in REQUIRED_PROVENANCE:
        if key not in result["provenance"]:
            raise ValueError(f"proveniência sem '{key}'")
    for key in REQUIRED_COUNTERS:
        if key not in result["counters"]:
            raise ValueError(f"contadores sem '{key}'")
    cfg, counters = result["config"], result["counters"]
    if counters["samples"] != counters["steps"] * cfg["batch"]:
        raise ValueError("orçamento inconsistente: samples != steps * batch")
    if result["status"] == "ok" and counters["steps"] != cfg["steps"]:
        raise ValueError("run 'ok' com menos passos que o configurado")
    if "best" not in result:
        raise ValueError("campo ausente: best")


def save(result: Dict, path: Path) -> None:
    validate(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
