"""
This is a copy of https://github.com/tadamcz/copula-wrapper/blob/main/tests/conftest.py

TODO Figure out how to properly import those tests
"""

import numpy as np
import pytest
import os
from scipy import stats

from epochutil.stats.joint_distribution import JointDistributionCond

import tests.stats.seeds as seeds


@pytest.fixture(params=[-1, 0.5], ids=lambda p: f"mu_n={p}")
def mu_norm(request):
    return request.param


@pytest.fixture(params=[1], ids=lambda p: f"sigma_n={p}")
def sigma_norm(request):
    return request.param


@pytest.fixture(params=[0], ids=lambda p: f"mu_lg={p}")
def mu_lognorm(request):
    return request.param


@pytest.fixture(params=[1], ids=lambda p: f"sigma_lg={p}")
def sigma_lognorm(request):
    return request.param


@pytest.fixture(params=[2, 3], ids=lambda p: f"alpha={p}")
def alpha(request):
    return request.param


@pytest.fixture(params=[1], ids=lambda p: f"beta={p}")
def beta(request):
    return request.param


@pytest.fixture()
def marginals(mu_norm, sigma_norm, mu_lognorm, sigma_lognorm, alpha, beta):
    return {
        "n": stats.norm(mu_norm, sigma_norm),
        "l": stats.lognorm(scale=np.exp(mu_lognorm), s=sigma_lognorm),
        "b": stats.beta(alpha, beta),
    }


@pytest.fixture(params=[
    {("n", "l"): 0.2, ("n", "b"): 0.3, ("l", "b"): 0.6},
    {("n", "l"): 0.5, ("n", "b"): 0.5, ("l", "b"): 0.5},
    {("n", "l"): 0.99, ("n", "b"): 0.99, ("l", "b"): 0.99},
], ids=lambda rs: f"pairwise_corrs={rs}")
def rank_corr(request):
    return request.param


@pytest.fixture(params=['spearman', 'kendall'], ids=lambda p: f"method={p}")
def rank_corr_method(request):
    return request.param


@pytest.fixture()
def joint_distribution(marginals, rank_corr, rank_corr_method):
    return JointDistributionCond(marginals=marginals, rank_corr=rank_corr, rank_corr_method=rank_corr_method)


@pytest.fixture()
def joint_sample(joint_distribution):
    sample = joint_distribution.rvs(5_000_000)
    return sample


def seed_idfn(fixture_value):
    return f"seed={fixture_value}"

# This is a copy of https://github.com/tadamcz/copula-wrapper/blob/main/conftest.py
# TODO Import the contents properly

n_random_seeds = int(os.environ.get('N_RAND_SEED', 1))


@pytest.fixture(autouse=True, params=seeds.RANDOM_SEEDS[:n_random_seeds], ids=seed_idfn, scope='session')
def random_seed(request):
    """
    autouse:
    this fixture will be used by every test, even if not explicitly requested.
    # TODO: remove autouse=True, some tests do not need this fixture, and it's very wasteful!

    params:
    this fixture will be run once for each element in params


    scope:
    setting scope to 'session' is the easiest way to control the ordering,
    so that all tests are run for RANDOM_SEEDS[0], then all tests for RANDOM_SEEDS[1], etc.
    """
    np.random.seed(request.param)
