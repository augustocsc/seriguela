"""Testes offline do orquestrador Colab CLI (nenhuma chamada de rede nem gasto de CU)."""
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from seriguela import colab


def _make_tgz_bytes(rel_path: str, payload: bytes = b'{"ok": true}') -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(rel_path)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_parse_usage_and_budget_cu_hours():
    sample = "Current balance: 600.00 compute units\nUsage rate: 4.80/hr\nActive assignments: 1\n"
    info = colab.parse_usage(sample)
    assert info == {"balance_cu": 600.0, "rate_cu_per_hr": 4.8, "active_assignments": 1}
    assert colab.budget_cu_hours(12.0, 4.0, setup_hours=0.25) == pytest.approx(2.75)
    with pytest.raises(colab.ColabError, match="setup"):
        colab.budget_cu_hours(0.5, 5.0, setup_hours=0.25)


def test_remote_driver_source_pins_commit_and_gates():
    src = colab.remote_driver_source("deadbeef", "campaigns/g1_timing.json", "results/v3/g1_timing",
                                     parallel=2, max_hours=1.5, drive_mirror="/content/drive/MyDrive/s")
    assert "COMMIT = 'deadbeef'" in src
    assert '"pytest"' in src and '"seriguela.canary"' in src and '"seriguela.campaign"' in src
    assert "start_new_session=True" in src


def test_run_campaign_always_stops_session_even_when_rate_exceeds_budget(tmp_path):
    camp_file = colab.REPO / "campaigns" / "g1_timing.json"
    calls = []

    def fake_runner(cmd, timeout=60):
        calls.append(list(cmd))
        if cmd[0] == "usage":
            started = any(c[0] == "new" for c in calls)
            rate = "50.00" if started else "0.00"
            act = "1" if started and not any(c[0] == "stop" for c in calls) else "0"
            out = f"Current balance: 600.00 compute units\nUsage rate: {rate}/hr\nActive assignments: {act}\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    with pytest.raises(colab.ColabError, match="setup"):
        colab.run_campaign(str(camp_file), gpu="A100", budget_cu=2.0, allow_dirty=True,
                           poll_interval=0.0, runner=fake_runner)

    assert any(c[:3] == ["stop", "-s", colab.SESSION_DEFAULT] for c in calls)
    assert not any(c[0] == "exec" for c in calls)


def test_run_campaign_detached_poll_survives_websocket_drop_and_resumes_local(tmp_path, monkeypatch):
    monkeypatch.setattr(colab, "REPO", tmp_path)
    (tmp_path / "campaigns").mkdir()
    camp_path = tmp_path / "campaigns" / "mini.json"
    camp_path.write_text(json.dumps({
        "name": "mini",
        "out": "results/v3/mini",
        "cells": [{"problems": ["nguyen_1"], "algos": ["bon"], "seeds": [0, 1]}],
    }), encoding="utf-8")
    # Simula 1 run já concluído localmente antes da retomada
    prev_run = tmp_path / "results" / "v3" / "mini" / "nguyen_1" / "bon" / "seed_0"
    prev_run.mkdir(parents=True)
    (prev_run / "summary.json").write_text('{"done": true}', encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    calls = []
    usage_step = {"n": 0}
    poll_step = {"n": 0}

    def fake_runner(cmd, timeout=60):
        calls.append(list(cmd))
        if cmd[0] == "usage":
            usage_step["n"] += 1
            bal = "600.00" if usage_step["n"] < 3 else "598.50"
            rate = "4.00" if usage_step["n"] == 2 else "0.00"
            act = "1" if usage_step["n"] == 2 else "0"
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"Current balance: {bal} compute units\nUsage rate: {rate}/hr\nActive assignments: {act}\n", stderr=""
            )
        if cmd[0] == "exec" and str(cmd[-1]).endswith("poll.py"):
            poll_step["n"] += 1
            if poll_step["n"] == 1:
                # Simula queda transitória de websocket — NÃO pode derrubar a campanha!
                raise colab.ColabError("Connection was lost.")
            if poll_step["n"] == 2:
                payload = {"alive": True, "summary": None, "packed_done": 2,
                           "status": {"done": 2, "total": 2, "running": [], "eta_h": 0.0, "failures": []}}
            else:
                payload = {"alive": False, "packed_done": 2,
                           "summary": {"outcome": "complete", "packed": True, "done": 2, "elapsed_s": 12.0},
                           "status": {"done": 2, "total": 2, "running": [], "eta_h": 0.0, "failures": []}}
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{colab.POLL_PREFIX}{json.dumps(payload)}\n", stderr="")
        if cmd[0] == "download":
            remote, local = cmd[-2], Path(cmd[-1])
            if remote.endswith("job_summary.json"):
                local.write_text(json.dumps({"outcome": "complete", "packed": True, "done": 2, "elapsed_s": 12.0}), encoding="utf-8")
            elif remote.endswith("results_v3.tgz"):
                local.write_bytes(_make_tgz_bytes("results/v3/mini/marker.json"))
            return subprocess.CompletedProcess(cmd, 0, stdout="downloaded", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    res = colab.run_campaign(str(camp_path), gpu="L4", budget_cu=10.0, allow_dirty=True,
                             poll_interval=0.0, runner=fake_runner)
    assert res["outcome"] == "complete" and res["spent_cu"] == pytest.approx(1.5)
    assert res["active_after_stop"] == 0
    assert res["resumed_local_runs"] == 1
    assert (tmp_path / "results" / "v3" / "mini" / "marker.json").exists()
    # Confere que houve upload de retomada, sync incremental + final, e stop garantido
    assert any(c[0] == "upload" and c[-1] == "/content/seed_results.tgz" for c in calls)
    assert sum(1 for c in calls if c[0] == "download" and c[-2] == "/content/results_v3.tgz") >= 2
    assert any(c[:3] == ["stop", "-s", colab.SESSION_DEFAULT] for c in calls)
