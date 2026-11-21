"""PLAN 10.1: statistics on synthetic data with known answers (+ statsmodels cross-checks)."""

import math

import numpy as np
import pytest
from scipy import stats as sps

from qopd.stats.intervals import bootstrap_diff, bootstrap_problems, seed_ci
from qopd.stats.tests import compare_spearman, holm, tost_paired, tost_two_sample

# --- intervals ------------------------------------------------------------------------------


def test_seed_ci_known_values():
    est, lo, hi = seed_ci([1.0, 2.0, 3.0])
    half = sps.t.ppf(0.975, 2) * 1.0 / math.sqrt(3)
    assert est == 2.0 and lo == pytest.approx(2 - half) and hi == pytest.approx(2 + half)
    single = seed_ci([0.7])
    assert single.estimate == 0.7 and math.isnan(single.low)


def test_bootstrap_degenerate_and_coverage():
    assert tuple(bootstrap_problems([[1, 1]] * 50)) == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(0)
    covered = 0
    for rep in range(200):  # true pass@1 = 0.6
        data = (rng.random((300, 1)) < 0.6).astype(float).tolist()
        _, lo, hi = bootstrap_problems(data, n_boot=1000, seed=rep)
        covered += lo <= 0.6 <= hi
    assert 0.90 <= covered / 200 <= 0.99  # nominal 95%


def test_bootstrap_uneven_samples_and_diff():
    est, *_ = bootstrap_problems([[1, 0, 1, 1], [0], [1, 1]])
    assert est == pytest.approx((0.75 + 0 + 1) / 3)
    a = [[1]] * 60 + [[0]] * 40
    b = [[1]] * 50 + [[0]] * 50
    d, lo, hi = bootstrap_diff(a, b, n_boot=2000)
    assert d == pytest.approx(0.10) and lo <= 0.10 <= hi
    with pytest.raises(ValueError):
        bootstrap_diff(a, b[:-1])


# --- TOST -----------------------------------------------------------------------------------------


def test_tost_paired_matches_statsmodels():
    from statsmodels.stats.weightstats import ttost_paired

    rng = np.random.default_rng(1)
    x = rng.normal(0.70, 0.2, 500)
    y = x + rng.normal(0.002, 0.05, 500)
    ours = tost_paired(x - y, margin=0.01)
    p_sm, low, up = ttost_paired(x, y, -0.01, 0.01)
    assert ours.p == pytest.approx(p_sm, rel=1e-6)
    assert ours.p_lower == pytest.approx(low[1], rel=1e-6)
    assert ours.p_upper == pytest.approx(up[1], rel=1e-6)


def test_tost_two_sample_matches_statsmodels_welch():
    from statsmodels.stats.weightstats import ttost_ind

    a, b = [0.701, 0.712, 0.695, 0.708], [0.705, 0.699, 0.710]
    ours = tost_two_sample(a, b, margin=0.02)
    p_sm, *_ = ttost_ind(a, b, -0.02, 0.02, usevar="unequal")
    assert ours.p == pytest.approx(p_sm, rel=1e-6)


def test_tost_known_answers():
    rng = np.random.default_rng(2)
    same = tost_paired(rng.normal(0.0, 0.05, 2000), margin=0.01)
    assert same.equivalent and same.ci90[0] > -0.01 and same.ci90[1] < 0.01
    shifted = tost_paired(rng.normal(0.05, 0.05, 2000), margin=0.01)  # 5-point drop
    assert not shifted.equivalent
    noisy = tost_paired(rng.normal(0.0, 0.5, 20), margin=0.01)  # too little data
    assert not noisy.equivalent


# --- Spearman comparison --------------------------------------------------------------------------


def test_compare_spearman_known_ordering():
    rng = np.random.default_rng(3)
    x1 = np.arange(10, dtype=float)  # perfect predictor
    y = x1.copy()
    x2 = rng.permutation(10).astype(float)  # unrelated
    res = compare_spearman(y, x1, x2, n_boot=2000)
    assert res.rho_1 == pytest.approx(1.0)
    assert res.diff > 0 and res.ci[0] > 0 and res.p_boot < 0.05
    tie = compare_spearman(y, x1, x1, n_boot=500)
    assert tie.diff == 0 and tie.p_boot == 1.0


# --- Holm -----------------------------------------------------------------------------------------


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = [0.01, 0.04, 0.03, 0.005, 0.20]
    rej, adj = holm(p)
    rej_sm, adj_sm, *_ = multipletests(p, alpha=0.05, method="holm")
    assert rej == list(rej_sm)
    assert adj == pytest.approx(list(adj_sm))


def test_holm_textbook_example():
    rej, adj = holm([0.01, 0.02, 0.03])  # 3*0.01, 2*0.02, 1*0.03 -> 0.03, 0.04, 0.04
    assert adj == pytest.approx([0.03, 0.04, 0.04]) and rej == [True, True, True]
