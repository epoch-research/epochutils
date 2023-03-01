from scipy.stats import rv_continuous
import numpy as np


class PieceLogUniformTransformed(rv_continuous):
    # Base classes should implement these two functions

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def __init__(self, low, med, high):
        self.points = [low, med, high]
        self.points.sort()

        super().__init__(a = self.points[0], b = self.points[-1])

        for i in range(len(self.points)):
            self.points[i] = np.log(self.forward(self.points[i]))

        self.pieces = len(self.points) - 1
        self._cdf = np.vectorize(self._cdf)
        self._ppf = np.vectorize(self._ppf)

    def _cdf(self, x):
        y = np.log(self.forward(x))

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

        x = self.backward(np.exp(y))

        return x

class TwoPieceLogUniform(PieceLogUniformTransformed):
    forward  = lambda self, x: x
    backward = lambda self, y: y

class TwoPieceNegLogUniform(PieceLogUniformTransformed):
    forward  = lambda self, x: -x
    backward = lambda self, y: -y

class TwoPieceFracLogUniform(PieceLogUniformTransformed):
    forward  = lambda self, x: x/(1 - x)
    backward = lambda self, y: y/(1 + y)

class TwoPieceInvFracLogUniform(PieceLogUniformTransformed):
    def forward(self, x):
      z = 1 / x
      y = z / (1. - z)
      return y

    def backward(self, y):
      z = y / (1. + y)
      x = 1 / z
      return x
