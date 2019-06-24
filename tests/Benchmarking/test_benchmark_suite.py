import pytest 

from bingo.Benchmarking.BenchmarkSuite import BenchmarkSuite

@pytest.fixture
def benchmark_suite():
    return BenchmarkSuite()

def test_init_benchmark_suite(benchmark_suite):
    assert benchmark_suite.include == [0, 1, 2, 3, 4, 5, 6, 7]
    assert benchmark_suite.exclude == []

def test_benchmarks_dictionary(benchmark_suite):
    assert benchmark_suite.benchmarks_dict[0]['name'] == 'koza_1'
    assert benchmark_suite.benchmarks_dict[0]['function'] == 'x**4 + x**3 + x**2 + x'
    assert benchmark_suite.benchmarks_dict[7]['name'] == 'nguyen_6'
    assert benchmark_suite.benchmarks_dict[7]['function'] == 'np.sin(x) + np.sin(x + x**2)'

def test_includes_correct_benchmarks():
    benchmark_suite = BenchmarkSuite(include=[1, 3, 4])
    assert benchmark_suite.include == [1, 3, 4]
    for benchmark, i in zip(benchmark_suite.benchmarks, benchmark_suite.include):
        assert benchmark.name == benchmark_suite.benchmarks_dict[i]['name']
        assert benchmark.objective_function == benchmark_suite.benchmarks_dict[i]['function']

def test_excludes_correct_benchmarks():
    benchmark_suite = BenchmarkSuite(exclude=[1, 3, 4])
    assert benchmark_suite.include == [0, 2, 5, 6, 7]
    for benchmark, i in zip(benchmark_suite.benchmarks, benchmark_suite.include):
        assert benchmark.name == benchmark_suite.benchmarks_dict[i]['name']
        assert benchmark.objective_function == benchmark_suite.benchmarks_dict[i]['function']

def test_benchmark_training_set_nonempty(benchmark_suite):
    for benchmark in benchmark_suite.benchmarks:
        assert benchmark.training_set.__len__() == 20 
 