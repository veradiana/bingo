# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time
import numpy as np
from mpi4py import MPI

from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover
from bingo.SymbolicRegression.AGraph.AgraphMutation import AGraphMutation
from bingo.SymbolicRegression.AGraph.AGraphGenerator import AGraphGenerator
from bingo.SymbolicRegression.AGraph.ComponentGenerator import ComponentGenerator
from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression, ExplicitTrainingData

from bingo.Base.AgeFitnessEA import AgeFitnessEA
from bingo.Base.ParallelArchipelago import ParallelArchipelago
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization

POP_SIZE = 100
STACK_SIZE = 10

def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])

def equation_eval(x):
    return x**2 + 3.5*x**3

def execute_generational_steps():
    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    x = None
    y = None

    if rank == 0:
        x = init_x_vals(-10, 10, 100)
        y = equation_eval(x)

    x = MPI.COMM_WORLD.bcast(x, root=0)
    y = MPI.COMM_WORLD.bcast(y, root=0)

    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator(2)
    component_generator.add_operator(3)
    component_generator.add_operator(4)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)


    island = Island(ea, agraph_generator, POP_SIZE)

    archipelago = ParallelArchipelago(island)

    if archipelago.run_islands(2000, 1000, 1000):
        if rank == 0:
            print("print the best indv", archipelago.get_best_individual())


def main():
    time1 = time.time()
    execute_generational_steps()
    time2 = time.time()
    print("Time: ", time2 - time1)

if __name__ == '__main__':

    main()

