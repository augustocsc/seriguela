import argparse

import numpy as np

import pytest

from conftest import make_policy
from seriguela import canary


def args():
    return argparse.Namespace(problem="toy", seed=0, steps=3, batch=32, update_mb=32, model="x", allow_dirty=True)


@pytest.fixture
def varied_rewards(monkeypatch):
    """O modelo aleatório pode não gerar nenhuma expressão com R² > 0 (depende da versão do
    torch). Inválidas recebem uma recompensa determinística por texto para haver sinal."""
    import dataclasses
    import zlib
    from seriguela.reward import Evaluator
    original = Evaluator._invalid

    def invalid(self, text, error):
        return dataclasses.replace(original(self, text, error),
                                   reward=(zlib.crc32(text.encode()) % 100) / 1000.0)
    monkeypatch.setattr(Evaluator, "_invalid", invalid)


@pytest.mark.parametrize("algo,replay", canary.ARMS)
def test_canary_passes_on_every_arm(problem, varied_rewards, algo, replay):
    ok, detail = canary.check_arm(algo, replay, args(), make_policy, problem)
    assert ok, detail


def test_canary_fails_when_a_tripwire_fires(problem, monkeypatch):
    from seriguela import trainer
    monkeypatch.setattr(trainer, "evaluators_agree", lambda *a, **k: False)
    ok, detail = canary.check_arm("bon", "none", args(), make_policy, problem)
    assert not ok and "tripwire" in detail


def test_canary_fails_when_learning_arm_never_sees_signal(problem, monkeypatch):
    from seriguela import trainer
    monkeypatch.setattr(trainer.advantages, "compute",
                        lambda kind, r, eps: (np.zeros(len(r)), np.ones(len(r), bool)))
    ok, detail = canary.check_arm("ppo", "none", args(), make_policy, problem)
    assert not ok and "recompensas distintas" in detail
