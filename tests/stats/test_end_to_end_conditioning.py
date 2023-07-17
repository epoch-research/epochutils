import pytest
import random
import itertools
import numpy as np
from scipy import stats
from scipy.stats import kstest

from epochutils.stats.joint_distribution import JointDistributionCond, JointDistSampler

# TODO add a test comparing with a manually calculated example

def test_conditioned_variables(joint_distribution):
    marginals = joint_distribution.marginals

    for n in range(1, len(marginals) + 1):
        for marginal_names in itertools.combinations(marginals.keys(), n):
            # For every combination of marginals

            conditions = {name: marginals[name].rvs() for name in marginal_names}
            samples = joint_distribution.rvs(10, conditions=conditions)
            for name, value in conditions.items():
                assert max(samples[name] - value) == pytest.approx(0, abs=2e-5)


def test_high_correlation_conditioning():
    marginals = {
        'a': stats.norm(0, 1),
        'b': stats.norm(0, 1),
        'c': stats.norm(0, 1),
    }

    rank_corr = {
        # The library doesn't allow singular matrices (so, the correlation between a and b must be less than 1)
        pair: 1-1e-9 for pair in itertools.combinations(marginals.keys(), 2)
    }

    dist = JointDistributionCond(marginals=marginals, rank_corr=rank_corr, rank_corr_method='spearman')

    for m in marginals.keys():
        for x in (-10, -5, 0, 0.5, 1, 8):
            samples = dist.rvs(10_000, conditions={m: x})
            for o in marginals.keys():
                assert max(samples[m] - samples[o]) == pytest.approx(0, abs=1e-3)


def test_no_correlation_conditioning():
    marginals = {
        'a': stats.norm(0, 1),
        'b': stats.norm(0, 1),
        'c': stats.norm(0, 1),
    }

    dist = JointDistributionCond(marginals=marginals, rank_corr={}, rank_corr_method='spearman')

    samples_cond = dist.rvs(1_000_000, conditions={'a': marginals['a'].rvs()})
    for name, marginal in marginals.items():
        if name == 'a': continue

        distance = kstest(samples_cond[name], marginal.cdf).statistic
        assert distance < 0.002


def test_sample_generator(joint_distribution):
    sample_count = 40
    retry_count = 4

    conditions = {}

    rnd_state = np.random.get_state()
    regular_samples = joint_distribution.rvs(nobs=sample_count + retry_count, conditions=conditions).to_numpy()

    np.random.set_state(rnd_state)
    generator_samples = []
    sampler = JointDistSampler(joint_distribution, nobs=sample_count, conditions=conditions)
    for i, sample in enumerate(sampler):
        generator_samples.append(sample)
        if i >= sample_count - retry_count and i < sample_count:
            # Retry the last few samples
            sampler.retry_last_sample()
    generator_samples = np.array(generator_samples)

    assert sampler.retry_count == retry_count
    assert np.all(regular_samples == generator_samples)
