import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import multivariate_normal
from statsmodels.distributions.copula.api import GaussianCopula
from statsmodels.distributions.copula.copulas import CopulaDistribution

from copula_wrapper import CopulaJoint


class JointDistributionCond(CopulaJoint):
    """
    Joint distribution from which you can extract conditional samples.
    """

    def __init__(self, marginals, rank_corr, rank_corr_method='spearman'):
        spearman_rho = None
        kendall_tau = None

        if rank_corr_method == 'spearman':
            spearman_rho = rank_corr
        else:
            kendall_tau = rank_corr

        super().__init__(marginals, spearman_rho=spearman_rho, kendall_tau=kendall_tau)

        # Hackily substitute the copula for conditionatable versions
        copula_instance = GaussianCopulaCond(corr=self._wrapped.copula.corr, k_dim=len(self.marginals))
        self._wrapped = CopulaDistributionCond(copula_instance, self._wrapped.marginals)
        self.dimension_names = {name: i for i, name in enumerate(self.marginals)}

    def rvs(self, nobs=2, random_state=None, conditions={}):
        """
        conditions: a dictionary of conditions with the format {name of marginal: value on which to condition}
        """

        fixed_values = [conditions.get(name, np.nan) for name in self.marginals]
        rvs = self._wrapped.rvs(nobs=nobs, random_state=random_state, conditions=fixed_values)

        return self.samples_to_df(rvs)

    def samples_to_df(self, rvs):
        df = pd.DataFrame()
        for name, i in self.dimension_names.items():
            column = rvs[:, i]
            df[name] = column
        return df


class CopulaDistributionCond(CopulaDistribution):
    """
    Modifies `CopulaDistribution` from `statsmodels` to allow sampling conditionally.
    """

    def rvs(self, nobs=1, cop_args=None, marg_args=None, random_state=None, conditions=[]):
        """
        conditions: [] or tuple of fixed values for each marginal on which to condition (set a value to nan if you don't want to fix it)
        """

        # we'll be using cop_args to pass the conditions to the copula
        assert cop_args == None, 'Copula arguments are not allowed'
        assert len(conditions) == 0 or len(conditions) == len(self.marginals)

        conditions_q = [self.marginals[i].cdf(conditions[i]) for i in range(len(conditions))]
        return super().rvs(nobs=nobs, cop_args=conditions_q, marg_args=marg_args, random_state=random_state)


class GaussianCopulaCond(GaussianCopula):
    """
    Modifies `GaussianCopulaCond` from `statsmodels` to allow sampling conditionally.
    """

    def __init__(self, corr=None, k_dim=2):
        super().__init__(corr=corr, k_dim=k_dim)
        self.mu = np.zeros(len(corr))

    def rvs(self, nobs=1, args=[], random_state=None):
        """
        args: [] or tuple of fixed values in [0, 1] on which to condition (set a value to nan if you don't want to fix it)
        """

        if len(args) == 0: args = [np.nan] * self.k_dim
        assert len(args) == self.k_dim

        # The "0.5 + (1 - 1e-10) * (x - 0.5)" below is to ensure we pass to the normal ppf only values inside (0, 1).
        # TODO: Is that reasonable? sm_copulas.CopulaDistribution does the same
        conditions = [self.distr_uv.ppf(0.5 + (1 - 1e-10) * (x - 0.5)) for x in args]
        x = self.sample_normal_cond(conditions=conditions, nobs=nobs, random_state=random_state)
        return self.distr_uv.cdf(x)

    def sample_normal_cond(self, conditions=[], nobs=1, random_state=None):
        # NOTE: Code by Ege (with some minor modifications)

        assert len(conditions) == self.k_dim

        conditions = np.array(conditions)

        fixed_indices = np.nonzero(~np.isnan(conditions))[0]
        free_indices  = np.nonzero(np.isnan(conditions))[0]

        if len(fixed_indices) == 0:
            # Regular sampling without conditoning
            return multivariate_normal.rvs(cov=self.corr, size=nobs, random_state=random_state)

        if len(free_indices) == 0:
            # All the values are fixed
            return [conditions.copy() for i in range(nobs)]

        mu_1 = self.mu[free_indices]
        mu_2 = self.mu[fixed_indices]

        cov_11 = self.corr[free_indices,  :][:,  free_indices]
        cov_12 = self.corr[free_indices,  :][:, fixed_indices]
        cov_21 = self.corr[fixed_indices, :][:,  free_indices]
        cov_22 = self.corr[fixed_indices, :][:, fixed_indices]

        fixed_values = conditions[fixed_indices]

        mu_bar = mu_1 + np.dot(np.matmul(cov_12, np.linalg.inv(cov_22)), fixed_values - mu_2)
        cov_bar = cov_11 - np.matmul(np.matmul(cov_12, np.linalg.inv(cov_22)), cov_21)

        samples = []
        for s in multivariate_normal.rvs(mean=mu_bar, cov=cov_bar, size=nobs, random_state=random_state):
            sample = conditions.copy()
            sample[free_indices] = s
            samples.append(sample)

        return samples


class JointDistSampler:
    """
    Utility class to draw samples from a CopulaJoint, with retries
    """

    def __init__(self, joint_dist, nobs=1, **dist_kwargs):
        self.joint_dist = joint_dist
        self.nobs = nobs
        self.dist_kwargs = dist_kwargs

        self.samples_to_draw = nobs
        self.retry_count = 0

    def retry_last_sample(self):
        self.samples_to_draw += 1
        self.retry_count += 1

    def __iter__(self):
        min_samples_to_draw = 20

        while self.samples_to_draw > 0:
            samples = self.joint_dist.rvs(nobs=max(self.samples_to_draw, min_samples_to_draw), **self.dist_kwargs)
            for _, sample in samples.iterrows():
                self.samples_to_draw -= 1
                yield sample
                if self.samples_to_draw <= 0:
                    break

