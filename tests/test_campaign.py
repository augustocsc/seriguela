import json
import sys

import pytest

from seriguela import campaign


def spec(tmp_path, **cell):
    base = {"problems": ["nguyen_1", "nguyen_2"], "algos": ["bon", "ppo"], "replays": ["none", "elite"],
            "seeds": [0, 1]}
    base.update(cell)
    return {"name": "t", "out": str(tmp_path / "out"), "common": {"steps": 3}, "cells": [base]}


def fake(behaviour):
    """Comando falso: grava o resultado (ok), grava falha de tripwire (trip), dorme (slow) ou quebra."""
    def command(task):
        code = {
            "ok": f"p=r'{task.path}'; import os; os.makedirs(os.path.dirname(p), exist_ok=True); open(p,'w').write('{{}}')",
            "trip": f"p=r'{task.failed_path}'; import os,sys; os.makedirs(os.path.dirname(p), exist_ok=True); open(p,'w').write('{{}}'); sys.exit(2)",
            "slow": "import time; time.sleep(30)",
            "crash": "raise SystemExit(1)",
        }[behaviour(task)]
        return [sys.executable, "-c", code]
    return command


def test_expand_skips_bon_with_replay_and_counts(tmp_path):
    tasks = campaign.expand(spec(tmp_path))
    assert len(tasks) == 2 * (1 + 2) * 2               # bon só sem replay
    assert {t.cfg.arm for t in tasks} == {"bon", "ppo", "ppo+elite"}
    assert all(t.config["steps"] == 3 for t in tasks)


@pytest.mark.parametrize("cell", [{"algos": ["sac"]}, {"replays": ["fifo"]}, {"overrides": {"stepz": 1}},
                                  {"overrides": {"seed": 3}}, {"extra": 1}, {"seeds": [0, 0]}])
def test_expand_rejects_bad_cells(tmp_path, cell):
    with pytest.raises(campaign.CampaignError):
        campaign.expand(spec(tmp_path, **cell))


def test_command_carries_config(tmp_path):
    task = campaign.expand(spec(tmp_path))[0]
    cmd = task.command()
    assert cmd[1:3] == ["-m", "seriguela.run"] and "--steps" in cmd and "--allow_dirty" not in cmd


def test_work_completes_in_parallel_and_resumes(tmp_path):
    s = spec(tmp_path)
    status = campaign.work(s, parallel=3, max_hours=1, poll=0.05, command=fake(lambda t: "ok"))
    assert status["reason"] == "complete" and status["done"] == status["total"] == 12
    again = campaign.work(s, parallel=3, max_hours=1, poll=0.05, command=fake(lambda t: "crash"))
    assert again["reason"] == "complete"               # nada pendente: não roda nada
    assert json.loads(campaign.status_path(s).read_text())["finished"]


def test_work_stops_at_first_tripwire_and_refuses_to_restart(tmp_path):
    s = spec(tmp_path)
    status = campaign.work(s, parallel=2, max_hours=1, poll=0.05,
                           command=fake(lambda t: "trip" if t.cfg.problem == "nguyen_1" and t.cfg.seed == 1 else "slow"))
    assert status["reason"] == "failure" and status["failures"][0]["tripwire"]
    assert status["running"] == [] and status["done"] == 0
    with pytest.raises(campaign.CampaignError):
        campaign.work(s, parallel=2, max_hours=1, poll=0.05, command=fake(lambda t: "ok"))


def test_work_counts_crash_without_result_as_failure(tmp_path):
    status = campaign.work(spec(tmp_path), parallel=1, max_hours=1, poll=0.05, command=fake(lambda t: "crash"))
    assert status["reason"] == "failure" and not status["failures"][0]["tripwire"]


def test_work_respects_deadline_and_kills_running(tmp_path):
    ticks = iter(range(0, 10 ** 6, 1000))              # cada leitura do relógio avança 1000 s
    status = campaign.work(spec(tmp_path), parallel=2, max_hours=1, poll=0.01,
                           command=fake(lambda t: "slow"), clock=lambda: next(ticks))
    assert status["reason"] == "deadline" and status["done"] == 0 and status["running"] == []


@pytest.mark.parametrize("path", sorted(__import__("pathlib").Path("campaigns").glob("*.json")))
def test_committed_campaigns_are_valid(path):
    assert campaign.expand(campaign.load(path))
