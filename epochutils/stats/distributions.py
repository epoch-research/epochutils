# TODO
#  * make the distributions more completed


from scipy.stats import rv_continuous
import numpy as np

# Import Tom's nice distributions
from rvtools.construct import lognorm, uniform, loguniform, beta, pert, certainty


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

class TwoPieceUniform(PieceUniformTransformed):
    forward  = lambda self, x: x
    backward = lambda self, y: y

class TwoPieceLogUniform(PieceUniformTransformed):
    forward  = lambda self, x: np.log(x)
    backward = lambda self, y: np.exp(y)

class TwoPieceNegLogUniform(PieceUniformTransformed):
    forward  = lambda self, x: np.log(-x)
    backward = lambda self, y: -np.exp(y)

class TwoPieceFracLogUniform(PieceUniformTransformed):
    forward  = lambda self, x: np.log(x/(1 - x))
    backward = lambda self, y: np.exp(y)/(1 + np.exp(y))

class TwoPieceInvFracLogUniform(PieceUniformTransformed):
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
