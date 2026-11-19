import json

import pytest

from conftest import make_policy
from seriguela import collect
from seriguela.trainer import RunConfig, run, save


def write_run(root, problem_obj, algo, seed, commit=None):
    cfg = RunConfig(problem="toy", seed=seed, algo=algo, steps=3, batch=32, update_mb=32, lr=1e-3)
    result = run(cfg, make_policy(seed), problem_obj, allow_dirty=True)
    result["provenance"]["dirty"] = False
    if commit:
        result["provenance"]["commit"] = commit
    path = root / "toy" / result["arm"] / f"seed_{seed}.json"
    save(result, path)
    return path


def test_one_row_per_run_and_summary(tmp_path, problem):
    for algo in ("bon", "ppo"):
        for seed in (0, 1):
            write_run(tmp_path, problem, algo, seed)
    frame = collect.load(tmp_path)
    assert len(frame) == 4 and set(frame["arm"]) == {"bon", "ppo"}
    table = collect.summary(frame)
    assert list(table["seeds"]) == [2, 2] and (table["collapsed"] == 0).all()


def test_mixed_commits_are_refused(tmp_path, problem):
    write_run(tmp_path, problem, "bon", 0, commit="a" * 40)
    write_run(tmp_path, problem, "bon", 1, commit="b" * 40)
    with pytest.raises(collect.MixedProvenanceError):
        collect.load(tmp_path)
    assert len(collect.load(tmp_path, allow_mixed=True)) == 2


def test_failed_runs_are_listed_not_loaded(tmp_path, problem):
    path = write_run(tmp_path, problem, "bon", 0)
    path.with_name("seed_9.failed.json").write_text(json.dumps({"error": "TripwireError"}), encoding="utf-8")
    assert len(collect.load(tmp_path)) == 1 and len(collect.failures(tmp_path)) == 1
    assert collect.main([str(tmp_path)]) == 1


def test_a_tampered_record_does_not_load(tmp_path, problem):
    path = write_run(tmp_path, problem, "bon", 0)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["counters"]["samples"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        collect.load(tmp_path)
