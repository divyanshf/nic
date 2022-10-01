import math
import numpy as np

class Rastrigin:
    def __init__(self, dimension):
        self.dimension = dimension

    # Evaluate function
    def eval(self, X):
        A = 10
        return A * self.dimension + np.sum([((x*x) - (A * math.cos(2 * math.pi * x))) for x in X])

    # Get Fitness
    def getFitness(self, X):
        f = self.eval(X)
        return 1e9 if f == 0 else 1/f