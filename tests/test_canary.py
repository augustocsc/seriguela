import argparse

import pytest

from conftest import make_policy
from seriguela import canary


def args():
    return argparse.Namespace(problem="toy", seed=0, steps=3, batch=32, update_mb=32, model="x", allow_dirty=True)


@pytest.mark.parametrize("algo,replay", canary.ARMS)
def test_canary_passes_on_every_arm(problem, algo, replay):
    ok, detail = canary.check_arm(algo, replay, args(), make_policy, problem)
    assert ok, detail


def test_canary_fails_when_a_tripwire_fires(problem, monkeypatch):
    from seriguela import trainer
    monkeypatch.setattr(trainer, "evaluators_agree", lambda *a, **k: False)
    ok, detail = canary.check_arm("bon", "none", args(), make_policy, problem)
    assert not ok and "tripwire" in detail
