import timeit

import numpy as np

from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover
from bingo.SymbolicRegression.AGraph.AGraphMutation import AGraphMutation
from bingo.SymbolicRegression.AGraph.AGraphGenerator import AGraphGenerator
from bingo.SymbolicRegression.AGraph.ComponentGenerator \
    import ComponentGenerator
from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression
from bingo.Base.DeterministicCrowdingEA import DeterministicCrowdingEA
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from bingo.Benchmarking.BenchmarkSuite import BenchmarkSuite
from performance_benchmarks import StatsPrinter

POP_SIZE = 128
STACK_SIZE = 64
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
NUM_POINTS = 20
START = -1
STOP = 1
ERROR_TOLERANCE = 10e-9
SEED = 20

BENCHMARK_SUITE = BenchmarkSuite()
BENCHMARK = None
def init_island():
    
    training_set = BENCHMARK.training_set

    component_generator = ComponentGenerator(training_set.x.shape[1])
    #component_generator.add_operator(0)
    #component_generator.add_operator(1)
    component_generator.add_operator(2)
    component_generator.add_operator(3)
    component_generator.add_operator(4)
    component_generator.add_operator(5)
    component_generator.add_operator(6)
    #component_generator.add_operator(7)

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_set)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea_algorithm = DeterministicCrowdingEA(evaluator, crossover,
                                mutation, CROSSOVER_PROBABILITY,
                                MUTATION_PROBABILITY)

    island = Island(ea_algorithm, agraph_generator, POP_SIZE)
    return island


class IslandStatsPrinter(StatsPrinter):
    def __init__(self):
        super().__init__()
        self._output = ["-"*24+":::: REGRESSION BENCHMARKS ::::" + "-"*23,
                        self._header_format_string.format("NAME", "MEAN",
                                                          "STD", "MIN", "MAX"),
                        "-"*78]

def explicit_regression_benchmark():
    print("* Name:", BENCHMARK.name)
    print("* Objective function:", BENCHMARK.objective_function)
    island = init_island()
    island.evolve_until_convergence(max_generations=500, min_generations=10, \
                                    fitness_threshold=ERROR_TOLERANCE)
    print("* Best individual:", island.get_best_individual().get_latex_string())

def do_benchmarking():
    np.random.seed(15)
    printer = IslandStatsPrinter()
    for benchmark in BENCHMARK_SUITE.benchmarks:
        global BENCHMARK
        BENCHMARK = benchmark
        printer.add_stats(BENCHMARK.name,
                          timeit.repeat(explicit_regression_benchmark,
                          number=3,
                          repeat=3))
    printer.print()

if __name__ == "__main__":
    do_benchmarking()

