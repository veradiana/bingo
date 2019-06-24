"""
This Benchmark Suite module contains symbolic regression benchmarks
drawn from 'Genetic Programming Needs Better Benchmarks', (McDermott et al.).
 
"""

import numpy as np

from bingo.Benchmarking.Benchmark import Benchmark

"""
========  ============  ======================================
Number    Name          Objective Function
========  ============  ======================================
0         koza_1        x**4 + x**3 + x**2 + x
1         koza_2        x**5 - 2*x**3 + x
2         koza_3        x**6 - 2*x**4 + x**2
3         nguyen_1      x**3 + x**2 + x
4         nguyen_3      x**5 + x**4 + x**3 + x**2 + x
5         nguyen_4      x**6 + x**5 + x**4 + x**3 + x**2 + x
6         nguyen_5      np.sin(x**2)*np.cos(x) - 1
7         nguyen_6      np.sin(x) + np.sin(x + x**2)
========  ============  ======================================
"""
class BenchmarkSuite:

    def __init__(self, include=[0, 1, 2, 3, 4, 5, 6, 7], exclude=[]):
        self.include = include
        self.exclude = exclude
        self.benchmarks_dict = {0: {'name':'koza_1', 'function':'x**4 + x**3 + x**2 + x'}, \
                                1: {'name':'koza_2', 'function':'x**5 - 2*x**3 + x'}, \
                                2: {'name':'koza_3', 'function':'x**6 - 2*x**4 + x**2'}, \
                                3: {'name':'nguyen_1', 'function':'x**3 + x**2 + x'}, \
                                4: {'name':'nguyen_3', 'function':'x**5 + x**4 + x**3 + x**2 + x'}, \
                                5: {'name':'nguyen_4', 'function':'x**6 + x**5 + x**4 + x**3 + x**2 + x'}, \
                                6: {'name':'nguyen_5', 'function':'np.sin(x**2)*np.cos(x) - 1'}, \
                                7: {'name':'nguyen_6', 'function':'np.sin(x) + np.sin(x + x**2)'}}
        self.benchmarks = self._init_benchmarks()

    def _init_benchmarks(self):
        if len(self.exclude) > 0:
            self.include = [i for i in self.include if i not in self.exclude]
        return [Benchmark(self.benchmarks_dict[i]['name'], self.benchmarks_dict[i]['function']) for i in self.include]

