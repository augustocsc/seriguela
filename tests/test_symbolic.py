from seriguela.symbolic import symbolic_match


def test_equivalent_forms_match():
    assert symbolic_match("x_1*(x_1 + 1)", "x_1**2 + x_1") is True
    assert symbolic_match("x_1*(x_1*x_1 + x_1 + C)", "x_1**3 + x_1**2 + x_1") is True


def test_constant_offset_and_scale_count_as_solutions():
    assert symbolic_match("x_1**2 + x_1 + 3", "x_1**2 + x_1") is True       # difere por constante
    assert symbolic_match("2*x_1**2 + 2*x_1", "x_1**2 + x_1") is True       # razão constante


def test_wrong_and_constant_models_do_not_match():
    assert symbolic_match("x_1**2", "x_1**2 + x_1") is False
    assert symbolic_match("C + C", "x_1**2 + x_1") is False
    assert symbolic_match("x_1**C + x_1", "x_1**2 + x_1") is False          # C=1: é 2x, não x²+x


def test_clear_mismatch_is_decided_numerically_without_sympy(monkeypatch):
    import seriguela.symbolic as S
    monkeypatch.setattr(S.mp, "get_context", lambda *_: (_ for _ in ()).throw(AssertionError("sympy chamado")))
    assert symbolic_match("C*x_1*exp(x_1)", "x_1**3 + x_1**2 + x_1") is False
    assert symbolic_match("sin(x_1) + x_2", "x_1*x_2") is False


def test_fitted_constants_are_substituted():
    assert symbolic_match("C*x_1", "2.5*x_1", constants=[2.5]) is True
    # Constantes ajustadas por L-BFGS-B com ruído numérico de ~1e-7 devem casar (critério SRBench)
    assert symbolic_match("exp(C*x_1)", "exp(2*x_1)", constants=[1.9999998]) is True
    assert symbolic_match("C*x_1**2 + C*x_1", "x_1**2 + 2.5*x_1", constants=[1.0000001, 2.4999999]) is True
    assert symbolic_match("C*x_1**2 + C*x_1", "x_1**2 + 2.5*x_1", constants=[1.5, 2.5]) is False
