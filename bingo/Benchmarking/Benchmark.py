import numpy as np

from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression, \
                                                        ExplicitTrainingData
class Benchmark:

    def __init__(self, name, objective_function):
        self.name = name
        self.objective_function = objective_function
        self.training_set = self._make_training_set()
        self.testing_set = None

    def _make_training_set(self, start=-1, stop=1, num_points=20):
        np.random.seed(15)
        x = self._init_x_vals(start, stop, num_points)
        y = self._equation_eval(x)
        return ExplicitTrainingData(x, y)

    def _init_x_vals(self, start, stop, num_points):
        return np.linspace(start, stop, num_points).reshape([-1, 1])

    def _equation_eval(self, x):
        return eval(self.objective_function)
