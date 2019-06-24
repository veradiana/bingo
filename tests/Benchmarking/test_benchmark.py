import pytest 
import numpy as np

from bingo.Benchmarking.Benchmark import Benchmark
from bingo.Benchmarking.BenchmarkSuite import BenchmarkSuite


@pytest.fixture
def benchmark():
    return Benchmark("koza_1", "x**4 + x**3 + x**2 + x")

def test_initialize_benchmark_name(benchmark):
    assert benchmark.name == "koza_1"
    assert benchmark.objective_function == "x**4 + x**3 + x**2 + x"

def test_training_set(benchmark):
    x = 2
    assert benchmark._equation_eval(x) == 30
    assert benchmark.training_set.__len__() == 20 


