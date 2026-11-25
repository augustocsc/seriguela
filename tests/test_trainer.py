import dataclasses
import json

import numpy as np
import pytest
import torch

from conftest import make_policy
from seriguela import provenance, trainer
from seriguela.trainer import PRESETS, RunConfig, TripwireError, run, validate

ARMS = [(algo, replay) for algo in PRESETS for replay in (("none",) if algo == "bon" else ("none", "elite", "random"))]


def config(**kwargs):
    base = dict(problem="toy", seed=0, steps=4, batch=64, update_mb=32, lr=1e-3, ref_check_every=2)
    return RunConfig(**{**base, **kwargs})


def lora_state(policy):
    return {n: p.detach().clone() for n, p in policy.model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("algo,replay", ARMS)
def test_every_arm_runs_with_exact_budget_and_valid_record(problem, algo, replay):
    result = run(config(algo=algo, replay=replay), make_policy(), problem, allow_dirty=True)
    validate(result)
    json.dumps(result, allow_nan=False)                           # JSON estrito
    assert result["counters"]["samples"] == 4 * 64                # mesmo orçamento em todo braço
    assert [h["samples_total"] for h in result["history"]] == [64, 128, 192, 256]
    assert result["arm"] == (algo if replay == "none" else f"{algo}+{replay}")
    assert len(result["provenance"]["commit"]) >= 7 and result["provenance"]["data_hash"]


def test_bon_never_touches_the_weights_and_ppo_does(problem):
    frozen = make_policy()
    before = lora_state(frozen)
    run(config(algo="bon"), frozen, problem, allow_dirty=True)
    assert all(torch.equal(before[n], p) for n, p in lora_state(frozen).items())
    learner = make_policy()
    before = lora_state(learner)
    result = run(config(algo="ppo"), learner, problem, allow_dirty=True)
    assert any(h["updated"] for h in result["history"])
    assert any(not torch.equal(before[n], p) for n, p in lora_state(learner).items())
    assert result["final_policy"]["kl_from_prior_per_token"] != 0.0
    assert all(h["approx_kl"] >= 0.0 for h in result["history"] if h.get("updated"))


def test_same_seed_same_run(problem):
    a = run(config(algo="ppo", replay="elite"), make_policy(), problem, allow_dirty=True)
    b = run(config(algo="ppo", replay="elite"), make_policy(), problem, allow_dirty=True)
    strip = lambda r: [{k: v for k, v in h.items() if k != "seconds"} for h in r["history"]]
    assert strip(a) == strip(b) and a["best"] == b["best"]


def test_search_never_reads_the_test_set(problem):
    """Trocar o teste não pode mudar nada na busca nem na expressão escolhida."""
    other = dataclasses.replace(problem, y_test=problem.y_test * -3 + 1)
    a = run(config(algo="ppo", replay="elite"), make_policy(), problem, allow_dirty=True)
    b = run(config(algo="ppo", replay="elite"), make_policy(), other, allow_dirty=True)
    for key in ("mean_reward", "best_r2", "loss", "valid_rate"):
        assert [h[key] for h in a["history"]] == [h[key] for h in b["history"]]
    assert a["best"]["expression"] == b["best"]["expression"]
    assert a["best"]["train_r2"] == b["best"]["train_r2"]
    assert a["best"]["test_r2"] != b["best"]["test_r2"]


def test_replay_comes_from_earlier_steps_and_respects_the_ratio(problem):
    result = run(config(algo="ppo", replay="elite", steps=6), make_policy(), problem, allow_dirty=True)
    replays = [h["n_replay"] for h in result["history"]]
    assert replays[0] == 0 and max(replays) > 0 and max(replays) <= int(64 * 0.2)


def test_tripwire_when_sampling_and_scoring_diverge(problem, monkeypatch):
    policy = make_policy()
    honest = policy.score
    monkeypatch.setattr(policy, "score", lambda *a, **k: [v + 0.1 for v in honest(*a, **k)])
    with pytest.raises(TripwireError, match="log-prob"):
        run(config(algo="ppo"), policy, problem, allow_dirty=True)


def test_tripwire_when_the_fast_evaluator_is_wrong(problem, monkeypatch):
    monkeypatch.setattr(trainer, "evaluators_agree", lambda *a, **k: False)
    with pytest.raises(TripwireError, match="avaliador"):
        run(config(algo="bon"), make_policy(), problem, allow_dirty=True)


def test_collapse_is_recorded_not_hidden(problem):
    result = run(config(algo="bon", steps=10, validity_floor=1.1, validity_patience=2), make_policy(),
                 problem, allow_dirty=True)
    assert result["status"] == "collapsed" and result["counters"]["steps"] == 2
    validate(result)


def test_dirty_tree_is_refused(problem, monkeypatch):
    monkeypatch.setattr(provenance, "git_state", lambda: ("abc1234", True))
    with pytest.raises(provenance.DirtyTreeError):
        run(config(algo="bon"), make_policy(), problem)


def test_validate_rejects_a_budget_mismatch(problem):
    result = run(config(algo="bon"), make_policy(), problem, allow_dirty=True)
    result["counters"]["samples"] += 1
    with pytest.raises(ValueError, match="orçamento"):
        validate(result)


def test_best_is_selected_on_train_and_ties_prefer_simpler(problem):
    result = run(config(algo="bon", steps=6), make_policy(), problem, allow_dirty=True)
    top = result["top"]
    assert result["best"]["expression"] == top[0]["expression"]
    keys = [(-t["train_reward"], t["complexity"]) for t in top]
    assert keys == sorted(keys)


def test_temperature_schedule():
    cfg = RunConfig(steps=100, t_max=1.0, t_min=0.5)
    assert trainer.temperature(0, cfg) == pytest.approx(1.0)
    assert trainer.temperature(100, cfg) == pytest.approx(0.5)
    assert trainer.temperature(50, cfg) == pytest.approx(0.75)
    assert trainer.temperature(7, RunConfig(t_max=0.7, t_min=0.7)) == pytest.approx(0.7)
