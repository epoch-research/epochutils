# TODO
#  * make the distributions more completed


from scipy.stats import rv_continuous
import numpy as np


class Certainty(rv_continuous):
  def __init__(self, v):
      self.v = v
      super().__init__(a=v, b=v)

  def _ppf(self, q):
      return self.v

  def get_value(self):
      return self.v


class PieceUniformTransformed(rv_continuous):
    # Base classes should implement these two functions

    # TODO Make forward and backward not depend on self

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def __init__(self,
            low=None, med=None, high=None,
            quantiles=None,
        ):

        if (low is not None) and (med is not None) and (high is not None) and not (low <= med <= high): 
          raise ValueError(f"The parameters should be increasing: {low}, {med}, {high}")

        # Add param checking

        if not quantiles:
            quantiles = {}
        else:
            quantiles = quantiles.copy()

        if low is not None:
            quantiles[0] = low
        if med is not None:
            quantiles[0.5] = med
        if high is not None:
            quantiles[1] = high

        quantiles = {k: quantiles[k] for k in sorted(quantiles.keys())}

        if list(quantiles.values()) != sorted(quantiles.values()):
            raise ValueError(f"Inconsistent quantiles (should be increasing): {quantiles}")

        self.quantiles = quantiles

        super().__init__(a=self.quantiles[0], b=self.quantiles[1])

        self.transformed_q = {k: self.forward(v) for k, v in self.quantiles.items()}

        self._cdf = np.vectorize(self._cdf)
        self._ppf = np.vectorize(self._ppf)

    def _cdf(self, x):
        y = self.forward(x)

        # Integration direction
        d = +1 if (self.transformed_q[0] < self.transformed_q[1]) else -1

        qs = list(self.transformed_q.keys())

        integral = 0
        for i in range(len(qs) - 1):
            mass = qs[i+1] - qs[i]
            a = self.transformed_q[qs[i]]
            b = self.transformed_q[qs[i+1]]
            if d*y < d*b:
                integral += mass * ((y - a) / (b - a))
                break
            else:
                integral += mass

        return integral

    def _ppf(self, q):
        if q <= 0:
            y = self.transformed_q[0]
        elif q >= 1:
            y = self.transformed_q[1]
        else:
            qs = list(self.transformed_q.keys())
            for i in range(len(qs) - 1):
                a = qs[i]
                b = qs[i+1]
                if a <= q and q < b:
                    y = self.transformed_q[a] + (self.transformed_q[b] - self.transformed_q[a]) * (q - a) / (b - a)
                    break

        x = self.backward(y)

        return x

class TwoPieceUniformTransformed(rv_continuous):
    def __init__(self, low, mid, high):
        self.super().__init__(low=low, mid=mid, high=high)

class PieceUniform(PieceUniformTransformed):
    forward  = lambda self, x: x
    backward = lambda self, y: y

class PieceLogUniform(PieceUniformTransformed):
    forward  = lambda self, x: np.log(x)
    backward = lambda self, y: np.exp(y)

class PieceNegLogUniform(PieceUniformTransformed):
    forward  = lambda self, x: np.log(-x)
    backward = lambda self, y: -np.exp(y)

class PieceFracLogUniform(PieceUniformTransformed):
    forward  = lambda self, x: np.log(x/(1 - x))
    backward = lambda self, y: np.exp(y)/(1 + np.exp(y))

class PieceInvFracLogUniform(PieceUniformTransformed):
    def forward(self, x):
        z = 1 / x
        z = z / (1. - z)
        y = np.log(z)
        return y

    def backward(self, y):
        z = np.exp(y)
        z = z / (1. + z)
        x = 1 / z
        return x


class Metalog(rv_continuous):
    """
    Metalogistic function as described in Keelin 2016.
    Keelin 2106: http://metalogdistributions.com/images/The_Metalog_Distributions_-_Keelin_2016.pdf
    Reference website: http://metalogdistributions.com/
    """

    # TODO Add inline references to sections, equations...

    def __init__(self, quantiles, n_terms=None):
        self.raw_quantiles = quantiles.copy()
        self.quantiles = quantiles.copy()

        self.lower_bound = None
        if 0 in self.quantiles:
            self.lower_bound = self.quantiles[0]
            del self.quantiles[0]

        self.upper_bound = None
        if 1 in self.quantiles:
            self.upper_bound = self.quantiles[1]
            del self.quantiles[1]

        super().__init__(a=self.lower_bound, b=self.upper_bound)

        self.transform, self.inverse_transform = self._compute_transforms(self.lower_bound, self.upper_bound)

        if n_terms is None:
            n_terms = len(self.quantiles)

        self.a = self._fit_metalog(self.quantiles, n_terms, self.transform)
        if self.a is None:
            raise ValueError('Failed to fit metalog. The Y^T Y matrix is not invertible.')

        self._check_feasibility(self.a, self.raw_quantiles)

    def _compute_transforms(self, lower_bound, upper_bound):
        if (lower_bound is not None) and (upper_bound is not None):
            # TODO: Handle y = 0 and y = 1 cases
            transform = lambda x: np.log((x - lower_bound) / (upper_bound - x))
            inverse_transform = lambda y: (lower_bound + upper_bound * np.exp(y)) / (1 + np.exp(y))
            return transform, inverse_transform
        elif lower_bound is not None:
            # TODO: Handle y = 0 case
            transform = lambda x: np.log(x - lower_bound)
            inverse_transform = lambda y: lower_bound + np.exp(y)
            return transform, inverse_transform
        elif upper_bound is not None:
            # TODO: Handle y = 1 case
            transform = lambda x: -np.log(upper_bound - x)
            inverse_transform = lambda y: upper_bound - np.exp(-y)
            return transform, inverse_transform
        else:
            transform = lambda x: x
            inverse_transform = lambda y: y
            return transform, inverse_transform

    def _check_feasibility(self, a, raw_quantiles):
        # See http://metalogdistributions.com/equations/feasibility.html

        q_y = np.array(list(self.quantiles.keys()))

        if len(q_y) == 2:
            feasible = (a[1] > 0)
        elif len(q_y) == 3:
            feasible = (a[1] > 0 and abs(a[2])/a[1] <= 1.66711)
        else:
            feasible = True
            print('Warning: Feasibility check not implemented for more than 3 quantiles')

        if not feasible:
            raise ValueError(f'Failed feasibility check for quantiles {raw_quantiles}')

    def _fit_metalog(self, quantiles, n_terms, transform):
        assert n_terms <= len(quantiles), 'n_terms must be less than or equal to the number of quantiles'

        q_x = np.array(list(quantiles.values()))
        q_y = np.array(list(quantiles.keys()))

        n = n_terms
        m = len(q_x)
        ln = np.log(q_y/(1-q_y))

        z = transform(q_x)

        assert n >= 2

        # Compute the Yn matrix
        Y = np.zeros((m, n))
        Y[:, 0] = 1
        Y[:, 1] = ln
        if n > 2:
            Y[:, 2] = (q_y - 0.5) * ln
        if n > 3:
            Y[:, 3] = (q_y - 0.5)
        for k in range(5, n + 1, 2):
            Y[:, k] = (q_y - 0.5) ** ((n - 1)//2)
        for k in range(6, n + 1, 2):
            Y[:, k] = (q_y - 0.5) ** (n//2 - 1) * ln

        matrix = Y.T @ Y

        if np.linalg.matrix_rank(matrix) < n:
            return None

        inverse = np.linalg.inv(matrix)
        a = inverse @ Y.T @ z

        return a

    def ppf(self, y):
        a = self.a

        n = len(a)

        mu_coeff_indices = np.concatenate(([1, 4, 5], np.arange(7, n + 1, 2)))
        mu_coeff_indices = mu_coeff_indices[mu_coeff_indices <= n]

        s_coeff_indices = np.concatenate(([2, 3], np.arange(6, n + 1, 2)))
        s_coeff_indices = s_coeff_indices[s_coeff_indices <= n]

        mu = np.dot(a[mu_coeff_indices-1], np.power(y - 0.5, np.vstack(np.arange(len(mu_coeff_indices)))))
        s = np.dot(a[s_coeff_indices-1], np.power(y - 0.5, np.vstack(np.arange(len(s_coeff_indices)))))

        result = self.inverse_transform(mu + s * np.log(y / (1 - y)))
        return result if isinstance(y, np.ndarray) else result[0]

    def quantile(self, q):
      return self.ppf(q)

    def rvs(self, size=1):
        return self.ppf(np.random.uniform(size=size))

# Aliases
TwoPieceUniformTransformed = PieceUniformTransformed
TwoPieceUniform            = PieceUniform
TwoPieceLogUniform         = PieceLogUniform
TwoPieceNegLogUniform      = PieceNegLogUniform
TwoPieceFracLogUniform     = PieceFracLogUniform
TwoPieceInvFracLogUniform  = PieceInvFracLogUniform
