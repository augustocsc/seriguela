import numpy as np
import pytest

from seriguela.expr import Expression, InvalidExpression, evaluators_agree, r2_score, reference_values

X = np.linspace(0.2, 2.0, 12).reshape(-1, 1)


def test_c_as_exponent_is_one_not_two():
    """Regressão do bug da v2: ``x**C`` com C=1 é x, não x²."""
    assert np.allclose(Expression("x_1**C").evaluate(X), X[:, 0])
    assert np.allclose(Expression("x_1*(x_1 + x_1**C) + x_1").evaluate(X), 2 * X[:, 0] ** 2 + X[:, 0])


def test_each_c_is_its_own_parameter():
    e = Expression("C*x_1 + C")
    assert e.n_const == 2
    assert np.allclose(e.evaluate(X, [2.0, 3.0]), 2 * X[:, 0] + 3)
    assert np.allclose(e.evaluate(X), X[:, 0] + 1)


def test_c_is_matched_as_a_whole_word():
    assert Expression("cos(x_1)").n_const == 0
    assert np.allclose(Expression("cos(x_1)*C").evaluate(X), np.cos(X[:, 0]))


@pytest.mark.parametrize("text", ["", "x_1 +", "x_1**(9**9)", "x_1**2**3", "x_1**999", "__import__(1)",
                                  "x_1; 1", "x_1 + " * 60 + "x_1"])
def test_unsafe_or_malformed_is_rejected(text):
    with pytest.raises(InvalidExpression):
        Expression(text)


def test_constant_expression_broadcasts_and_bad_domain_is_nan():
    assert Expression("C + C").evaluate(X).shape == (12,)
    assert np.isnan(Expression("log(x_1 - 5)").evaluate(X)).all()
    assert np.isnan(Expression("x_2").evaluate(X)).all()          # variável que o problema não tem


@pytest.mark.parametrize("text", ["(9)**(9)**(9)", "x_1 + (9)**((9)**(9))", "(7)**(77)**(7)*x_1",
                                  "exp(exp(exp(exp(9))))"])
def test_numeric_towers_fail_fast_instead_of_hanging(text):
    """Inteiros exatos gigantes travavam a v2 por horas; aqui a expressão é inválida na hora."""
    import time
    start = time.time()
    try:
        assert not np.all(np.isfinite(Expression(text).evaluate(X)))
    except InvalidExpression:
        pass
    assert time.time() - start < 2.0


def test_integer_literals_keep_their_meaning():
    assert np.allclose(Expression("x_1**2 + 3*x_1/2").evaluate(X), X[:, 0] ** 2 + 1.5 * X[:, 0])
    assert np.allclose(Expression("x_1**(1/2)").evaluate(X), np.sqrt(X[:, 0]))


def test_prefix_notation():
    assert np.allclose(Expression("* x_1 + x_1 C", is_prefix=True).evaluate(X), X[:, 0] * (X[:, 0] + 1))


def test_fit_constants_recovers_a_line():
    y = 2.5 * X[:, 0] + 0.5
    e = Expression("C*x_1 + C")
    c = e.fit_constants(X, y)
    assert r2_score(y, e.evaluate(X, c)) > 0.999999


def test_r2_score():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full(3, 2.0)) == 0.0
    assert r2_score(y, np.array([1.0, np.nan, 3.0])) == float("-inf")


def _random_expression(rng, depth=0):
    if depth > 2 or rng.random() < 0.3:
        return rng.choice(["x_1", "x_2", "C", "2"])
    kind = rng.integers(0, 4)
    a, b = _random_expression(rng, depth + 1), _random_expression(rng, depth + 1)
    if kind == 0:
        return f"({a} {rng.choice(['+', '-', '*', '/'])} {b})"
    if kind == 1:
        return f"{rng.choice(['sin', 'cos', 'exp', 'sqrt', 'log'])}({a})"
    if kind == 2:
        return f"({a})**{rng.choice(['2', '3', 'C'])}"
    return f"{a}*{b}"


def test_fast_evaluator_matches_reference_on_random_expressions():
    """Teste diferencial: numpy (rápido) contra sympy->math (referência)."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0.3, 1.7, (8, 2))
    compared = 0
    for _ in range(300):
        text = _random_expression(rng)
        fast = Expression(text).evaluate(points)
        ref = reference_values(text, points)
        both = np.isfinite(fast) & np.isfinite(ref)
        compared += int(both.sum())
        assert np.allclose(fast[both], ref[both], rtol=1e-6, atol=1e-9), text
    assert compared > 1000


def test_reference_check_catches_a_wrong_evaluator(monkeypatch):
    assert evaluators_agree("x_1**C + x_1", X)
    original = Expression.evaluate
    monkeypatch.setattr(Expression, "evaluate", lambda self, X_, c=None: original(self, X_, c) ** 2)
    assert not evaluators_agree("x_1**C + x_1", X)
