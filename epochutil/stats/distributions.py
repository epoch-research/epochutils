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

    def __init__(self, low, med, high):
        self.params = [low, med, high]
        self.points = self.params.copy()
        self.points.sort()

        super().__init__(a = self.points[0], b = self.points[-1])

        for i in range(len(self.points)):
            self.points[i] = self.forward(self.points[i])

        self.pieces = len(self.points) - 1
        self._cdf = np.vectorize(self._cdf)
        self._ppf = np.vectorize(self._ppf)

    def _cdf(self, x):
        y = self.forward(x)

        # Integration direction
        d = +1 if (self.points[0] < self.points[-1]) else -1

        integral = 0
        for i in range(self.pieces):
            a = self.points[i]
            b = self.points[i+1]
            if d*y < d*b:
                integral += (y - a) / (b - a)
                break
            integral += 1
        integral /= self.pieces

        return integral

    def _ppf(self, q):
        if q <= 0:
            y = self.points[0]
        elif q >= 1:
            y = self.points[-1]
        else:
            piece = int(q * self.pieces)
            y = self.points[piece] + (q * self.pieces - piece) * (self.points[piece+1] - self.points[piece])

        x = self.backward(y)

        return x

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
