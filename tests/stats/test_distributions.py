import pytest
import numpy as np
from scipy.stats import kstest
from epochutils.stats.distributions import TwoPieceUniform, TwoPieceLogUniform, TwoPieceNegLogUniform, TwoPieceFracLogUniform, TwoPieceInvFracLogUniform


# TODO Add more of these
dist_params_set = [
    (TwoPieceLogUniform,        [1e10, 5e20, 1e40]),
    (TwoPieceNegLogUniform,     [-1e40, -5e20, -1e10]),
    (TwoPieceInvFracLogUniform, [1e01, 1e4, 1e08]),
]


@pytest.fixture(params=dist_params_set, ids=lambda p: f"{p[0].__name__}({p[1]})")
def dist_params(request):
    dist_class = request.param[0]
    params = request.param[1]
    dist = dist_class(*params)
    return (dist, params)


@pytest.fixture(params=[dp for dp in dist_params_set if dp[0] == TwoPieceLogUniform], ids=lambda p: f"{p[0].__name__}({p[1]})")
def dist_params_loguniform(request):
    dist_class = request.param[0]
    params = request.param[1]
    dist = dist_class(*params)
    return (dist, params)


def test_key_points(dist_params):
    dist, params = dist_params

    assert dist.cdf(min(params)) == pytest.approx(0)
    assert dist.cdf(min(params) - 1) == pytest.approx(0)

    assert dist.cdf(params[1]) == pytest.approx(0.5)

    assert dist.cdf(max(params)) == pytest.approx(1)
    assert dist.cdf(max(params) + 1) == pytest.approx(1)

    assert dist.ppf(0.0) == pytest.approx(min(params))
    assert dist.ppf(0.5) == pytest.approx(params[1])
    assert dist.ppf(1.0) == pytest.approx(max(params))


def test_monotonic(dist_params):
    dist, params = dist_params

    cdf = dist.cdf(np.geomspace(min(params), max(params), 100))
    assert np.all(np.diff(cdf) > 0)


def test_equal_mass(dist_params):
    dist, params = dist_params

    n = 100_000
    samples = dist.rvs(size=n)
    assert np.sum(samples < params[1]) / n == pytest.approx(0.5, abs=0.01)
    assert np.sum(samples > params[1]) / n == pytest.approx(0.5, abs=0.01)


def test_loguniform(dist_params_loguniform):
    dist, params = dist_params_loguniform

    n = 100

    for q, x in zip(np.linspace(0, 0.5, n), np.geomspace(min(params), params[1], n)):
        assert dist.cdf(x) == pytest.approx(q, rel=0.02)
        assert dist.ppf(q) == pytest.approx(x, rel=1e-12)

    for q, x in zip(np.linspace(0.5, 1, n), np.geomspace(params[1], max(params), n)):
        assert dist.cdf(x) == pytest.approx(q, rel=0.02)
        assert dist.ppf(q) == pytest.approx(x, rel=1e-12)


def compare_pieceuniform_distributions(dist, baseline_class):
    dumb_self = None

    transformed_params = [baseline_class.backward(dumb_self, dist.forward(p)) for p in dist.quantiles.values()]
    baseline = baseline_class(*sorted(transformed_params))

    n = 100_000
    samples = dist.rvs(size=n)
    transformed_samples = [baseline.backward(dist.forward(s)) for s in samples]
    baseline_samples = baseline.rvs(size=n)

    # Perform a Kolmogorov-Smirnov test for goodness of fit
    distance = kstest(transformed_samples, baseline_samples).statistic
    assert distance < 0.05


def test_compare_loguniform():
    dist = TwoPieceLogUniform(10, 20, 40)
    compare_pieceuniform_distributions(dist, TwoPieceUniform)


def test_compare_transformed_loguniform(dist_params):
    """
    Compare with log uniform (we can't compare directly with regular uniform due to integration difficulties)
    """

    dist, params = dist_params
    #compare_pieceuniform_distributions(dist, TwoPieceLogUniform)
    compare_pieceuniform_distributions(dist, TwoPieceUniform)
