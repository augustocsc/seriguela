"""Testes da análise pré-registrada (R2c) em dados sintéticos com efeitos conhecidos."""
import numpy as np
import pandas as pd
import pytest

from seriguela import analysis


def _synthetic_campaign(n_problems: int = 24, n_seeds: int = 5, seed: int = 42,
                        probs: dict = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = probs or {"bon": 0.10, "ppo": 0.20, "ppo+elite": 0.55, "ppo+random": 0.22,
                      "rspg": 0.25, "rspg+elite": 0.30, "grpo": 0.22}
    rows = []
    # Dificuldade aleatória por problema (efeito misto de cluster realista)
    difficulty = rng.normal(0.0, 0.08, size=n_problems)
    for p in range(n_problems):
        for s in range(n_seeds):
            u = rng.random()  # componente pareada por seed
            for arm, base_p in probs.items():
                eff_p = float(np.clip(base_p + difficulty[p], 0.01, 0.99))
                # Mistura correlação intra-seed com ruído independente por braço
                solved = bool((0.4 * u + 0.6 * rng.random()) < eff_p)
                rows.append({
                    "problem": f"p_{p:02d}", "arm": arm, "seed": s,
                    "status": "ok", "commit": "a" * 40, "dirty": False, "config_hash": "h",
                    "samples": 1000, "seconds": 10.0, "expression": "x_1",
                    "train_r2": 1.0 if solved else 0.5, "test_r2": 1.0 if solved else 0.5,
                    "solved_accuracy": solved, "solved_symbolic": solved,
                    "final_solved_frac": 0.5, "final_valid_rate": 0.8, "kl_from_prior": 0.01,
                    "path": f"p_{p}_{arm}_{s}.json",
                })
    return pd.DataFrame(rows)


def test_detects_known_positive_replay_and_interaction_effects():
    """Com efeito forte em ppo+elite (+0.35) e fraco em rspg+elite (+0.05), H1_ppo, H3, H4 e H2 disparam."""
    df = _synthetic_campaign(n_problems=24, n_seeds=5, seed=7)
    res = {r.name: r for r in analysis.evaluate_contrasts(df, n_boot=2000, n_perm=5000, seed=0)}

    assert res["H1_ppo"].verdict == "positive" and res["H1_ppo"].estimate > 0.20
    assert res["H1_ppo"].odds_ratio_mh is not None and res["H1_ppo"].odds_ratio_mh > 2.0
    assert res["H3_vs_bon"].verdict == "positive"
    assert res["H4_ppo_mechanism"].verdict == "positive"
    assert res["H2_interaction"].verdict == "negative"  # rspg gain < ppo gain


def test_null_campaign_never_claims_equivalence_and_controls_false_positives():
    """Sob H0 estrito (todos os braços = 0.30), veredito é 'inconclusive (not equivalence)'."""
    null_probs = {a: 0.30 for a in ("bon", "ppo", "ppo+elite", "ppo+random", "rspg", "rspg+elite", "grpo")}
    df = _synthetic_campaign(n_problems=20, n_seeds=5, seed=123, probs=null_probs)
    res = analysis.evaluate_contrasts(df, n_boot=2000, n_perm=4000, seed=1)
    assert all(r.verdict == "inconclusive (not equivalence)" for r in res)
    assert all(r.ci_low <= 0.15 and r.ci_high >= -0.15 for r in res)


def test_handles_complete_separation_strata_cleanly():
    """Problemas todos 0/5 ou todos 5/5 não quebram bootstrap, permutação nem Mantel-Haenszel."""
    df = _synthetic_campaign(n_problems=10, n_seeds=5, seed=0,
                             probs={"bon": 0.0, "ppo": 0.0, "ppo+elite": 1.0})
    # Força 0% exato e 100% exato
    df["solved_symbolic"] = df["arm"] == "ppo+elite"
    res = {r.name: r for r in analysis.evaluate_contrasts(df, n_boot=1000, n_perm=2000)}
    assert res["H1_ppo"].estimate == pytest.approx(1.0)
    assert res["H1_ppo"].verdict == "positive"
    assert res["H1_ppo"].odds_ratio_mh > 10.0


def test_holm_adjust_and_required_problems():
    adj = analysis.holm_adjust([0.04, 0.01, 0.20])
    assert adj == pytest.approx([0.08, 0.03, 0.20])
    m = analysis.required_problems([0.2, 0.3, 0.1, 0.25, 0.15], target_delta=0.10, power=0.80)
    assert 4 <= m <= 20


def test_expressivity_ceiling_and_reports():
    ceiling = analysis.expressivity_ceiling()
    assert ceiling["nguyen"]["total"] == 12
    assert ceiling["feynman"]["total"] >= 120
    assert ceiling["strogatz"]["total"] == 14
    df = _synthetic_campaign(n_problems=8, n_seeds=3, seed=5)
    contrasts = analysis.evaluate_contrasts(df, n_boot=500, n_perm=1000)
    md = analysis.render_markdown(df, contrasts, ceiling)
    html_doc = analysis.render_html(md)
    assert "H1_ppo" in md and "<!doctype html>" in html_doc
