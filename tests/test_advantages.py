import numpy as np
import pytest

from seriguela import advantages as A


def test_zscore_is_centered_and_constant_rewards_give_zero():
    a = A.zscore(np.array([0.0, 0.5, 1.0, 0.5]))
    assert abs(a.mean()) < 1e-12 and abs(a.std() - 1) < 1e-12
    assert np.all(A.zscore(np.ones(8)) == 0)


def test_rank_gives_ties_the_same_advantage():
    """Na v2, quatro recompensas iguais recebiam [-1.34, -0.45, 0.45, 1.34]."""
    assert np.all(A.centered_rank(np.ones(4)) == 0)
    a = A.centered_rank(np.array([0.2, 0.9, 0.2, 0.5]))
    assert a[0] == a[2] < a[3] < a[1]
    assert abs(a.mean()) < 1e-12


def test_rank_is_invariant_to_reward_scale():
    r = np.array([0.1, 0.4, 0.3, 0.9, 0.0])
    assert np.allclose(A.centered_rank(r), A.centered_rank(r * 100 - 7))


def test_risk_seeking_keeps_only_the_top_quantile():
    r = np.linspace(0, 1, 100)
    adv, keep = A.risk_seeking(r, 0.1)
    assert 9 <= keep.sum() <= 11 and np.all(adv[~keep] == 0) and np.all(adv[keep] >= 0)
    assert adv.max() == pytest.approx(r.max() - np.quantile(r, 0.9))


def test_compute_dispatch():
    r = np.array([0.0, 1.0, 0.5])
    assert A.compute("zscore", r)[1].all() and A.compute("rank", r)[1].all()
    with pytest.raises(ValueError):
        A.compute("nope", r)


def test_replay_never_hijacks_fresh_quantile_or_baseline():
    """Elites antigas não podem deslocar o quantil de 5% do rspg nem a média do zscore."""
    fresh = np.linspace(0.0, 0.5, 100)
    elites = np.full(20, 0.99)
    both = np.concatenate([fresh, elites])

    adv_r, keep_r = A.compute_with_replay("risk", both, n_fresh=100, epsilon=0.05)
    # Top 5% das amostras NOVAS deve continuar sendo mantido (5 amostras de 100)
    assert keep_r[:100].sum() == 5
    assert np.allclose(adv_r[:100], A.risk_seeking(fresh, 0.05)[0])
    assert keep_r[100:].all() and (adv_r[100:] > 0).all()

    adv_z, _ = A.compute_with_replay("zscore", both, n_fresh=100)
    assert np.allclose(adv_z[:100], A.zscore(fresh))
    assert (adv_z[100:] > 0).all()

    adv_k, _ = A.compute_with_replay("rank", both, n_fresh=100)
    assert np.allclose(adv_k[:100], A.centered_rank(fresh))
    assert (adv_k[100:] > 0).all()
