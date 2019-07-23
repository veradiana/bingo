import numpy as np

from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression, \
                                                        ExplicitTrainingData
class Benchmark:

    def __init__(self, name, objective_function, train_set, has_test_set=False):
        self.name = name
        self.objective_function = objective_function
        self._has_test_set = has_test_set
        self.train_set = train_set
        self.test_set = []
        self.make_training_data()
        
    def equation_eval(self, x):
        return eval(self.objective_function)

    def make_training_data(self):
        np.random.seed(42)
        start = self.train_set[0]
        stop = self.train_set[1]
        num_points = self.train_set[2]
        x = self._init_x_vals(start, stop, num_points)
        y = self.equation_eval(x)
        data = ExplicitTrainingData(x, y)
        if self._has_test_set:
            self._make_test_set(data, 0.2)
        else:
            self.train_set = data
            self.test_set = []

    def _make_test_set(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        self.train_set = data.__getitem__(train_indices)
        self.test_set = data.__getitem__(test_indices)

    def _init_x_vals(self, start, stop, num_points):
        return np.linspace(start, stop, num_points).reshape([-1, 1])

