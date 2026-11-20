"""Controle do pod sem rede: partes puras e o script do job rodando com comandos falsos."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from seriguela import pod

BASH = shutil.which("bash")


def test_budget_hours_discounts_setup_and_grace():
    assert pod.budget_hours(2.0, 0.25, grace_min=30) == pytest.approx(8 - pod.SETUP_HOURS - 0.5)
    with pytest.raises(pod.PodError):
        pod.budget_hours(0.1, 0.25, grace_min=30)


def test_extract_ssh():
    assert pod.extract_ssh({"publicIp": "1.2.3.4", "portMappings": {"22": 4022}}) == ("1.2.3.4", 4022)
    assert pod.extract_ssh({"publicIp": "", "portMappings": {}}) == (None, None)


def test_job_script_pins_commit_and_caps_time():
    script = pod.job_script("abc123", "campaigns/x.json", parallel=3, max_hours=2.0, grace_min=10)
    assert "git checkout -q --detach abc123" in script
    assert f"timeout {int((2.0 + pod.SETUP_HOURS) * 3600)}" in script
    assert "--parallel 3 --max_hours 2.0" in script


SHIMS = {
    "git": """case "$1" in clone) mkdir -p "$4";; rev-parse) echo abc123;; esac""",
    "python": """case "$*" in
  *"-m venv"*) d="${@: -1}"; mkdir -p "$d/bin"; echo ':' > "$d/bin/activate";;
  *pytest*) exit ${FAKE_G0:-0};;
  *seriguela.canary*) exit ${FAKE_G1:-0};;
  *seriguela.campaign*) exit ${FAKE_CAMPAIGN:-0};;
esac
exit 0""",
    "pip": "exit 0",
    "runpodctl": 'echo "$*" >> "$WORKDIR/runpodctl.calls"',
    "pkill": "exit 0",
}


def run_job(tmp_path, fetched=False, **env):
    if BASH is None:
        pytest.skip("sem bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in SHIMS.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body + "\n", newline="\n")
        path.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    if fetched:
        (work / "FETCHED").touch()
    as_bash = lambda p: subprocess.run([BASH, "-c", f'cd "{p.as_posix()}" && pwd'], capture_output=True,
                                       text=True).stdout.strip()
    work_b, bin_b = as_bash(work), as_bash(bin_dir)
    script = work / "job.sh"
    script.write_text(pod.job_script("abc123", "campaigns/x.json", 2, 1.0, grace_min=0, work=work_b), newline="\n")
    full_env = {**os.environ, "RUNPOD_POD_ID": "pod42", "WORKDIR": work_b, **env}
    cmd = f'export PATH="{bin_b}:$PATH"; bash "{work_b}/job.sh"'
    subprocess.run([BASH, "-c", cmd], env=full_env, timeout=60, capture_output=True)
    calls = (work / "runpodctl.calls").read_text().split("\n") if (work / "runpodctl.calls").exists() else []
    return (work / "JOB_DONE").read_text().strip(), [c for c in calls if c]


@pytest.mark.parametrize("env,expected", [({}, "complete"), ({"FAKE_G0": "1"}, "g0_failed"),
                                          ({"FAKE_G1": "1"}, "g1_failed"), ({"FAKE_CAMPAIGN": "1"}, "campaign_stopped")])
def test_job_reports_outcome_and_always_turns_pod_off(tmp_path, env, expected):
    done, calls = run_job(tmp_path, **env)
    assert done == expected
    assert calls == ["stop pod pod42"]                 # ninguém buscou: para e guarda o volume


def test_job_removes_pod_after_results_were_fetched(tmp_path):
    done, calls = run_job(tmp_path, fetched=True)
    assert done == "complete" and calls == ["remove pod pod42"]
