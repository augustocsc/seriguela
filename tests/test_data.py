import numpy as np
import pytest

from seriguela.data import BENCH_DIR, NGUYEN_PROBLEMS, load_problem


def test_nguyen_is_deterministic_and_test_is_disjoint():
    a, b = load_problem("nguyen_5", 7), load_problem("nguyen_5", 7)
    assert a.data_hash() == b.data_hash() != load_problem("nguyen_5", 8).data_hash()
    assert not np.intersect1d(a.x_train, a.x_test).size
    assert np.allclose(a.y_train, np.sin(a.x_train[:, 0] ** 2) * np.cos(a.x_train[:, 0]) - 1)


@pytest.mark.parametrize("name", NGUYEN_PROBLEMS)
def test_nguyen_canonical_domains(name):
    p = load_problem(name, 0)
    lo, hi, n = p.meta["domain"]
    assert p.x_train.shape[0] == n == 20 and p.x_train.min() >= lo and p.x_train.max() <= hi
    assert np.all(np.isfinite(p.y_train)) and np.all(np.isfinite(p.y_test))


def test_nguyen_v2_variant_and_extrapolation():
    assert load_problem("nguyen_1", 0, nguyen_variant="v2").x_train.shape == (100, 1)
    wide = load_problem("nguyen_1", 0, test_scale=2.0)
    assert wide.x_test.min() < -1 and wide.x_test.max() > 1


@pytest.mark.skipif(not (BENCH_DIR / "strogatz" / "strogatz_lv1.csv").exists(), reason="CSV ausente")
def test_srbench_split_and_noise():
    p = load_problem("strogatz_lv1", 0, download=False)
    total = p.meta["n_rows"]
    assert p.x_test.shape[0] == int(total * 0.25) and p.x_train.shape[0] == total - int(total * 0.25)
    noisy = load_problem("strogatz_lv1", 0, noise=0.1, download=False)
    assert np.array_equal(p.x_train, noisy.x_train) and np.array_equal(p.y_test, noisy.y_test)
    assert not np.array_equal(p.y_train, noisy.y_train)
    assert load_problem("strogatz_lv1", 0, max_train=50, download=False).x_train.shape[0] == 50


def test_unknown_problem():
    with pytest.raises(ValueError):
        load_problem("nope", 0)


def test_all_local_srbench_truths_parse_without_unmapped_variables():
    import json, re, sympy
    from seriguela.data import srbench_truth

    known = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "asin", "acos", "atan",
             "sinh", "cosh", "tanh", "cot", "pi", "E"}
    checked = 0
    for family in ("feynman", "strogatz"):
        for meta in sorted((BENCH_DIR / family).glob("*.meta.json")):
            d = json.loads(meta.read_text(encoding="utf-8"))
            cols = [f["name"] for f in d.get("pmlb_metadata", {}).get("features", [])] or d["variables"]
            truth = srbench_truth(d["name"], family, cols)
            assert truth is not None, f"truth ausente em {d['name']}"
            ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", truth))
            leftover = [i for i in ids if not re.match(r"^x_\d+$", i) and i not in known]
            assert not leftover, f"{d['name']}: símbolos não mapeados {leftover} em {truth}"
            assert sympy.sympify(truth) is not None
            checked += 1
    assert checked >= 115
