# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import sys
import numpy as np
import inspect
from mpi4py import MPI
from unittest.mock import Mock
from bingo.Base.MultipleValues import SinglePointCrossover, \
                                      SinglePointMutation, \
                                      MultipleValueChromosomeGenerator
from bingo.Base.Island import Island
from bingo.Base.MuPlusLambdaEA import MuPlusLambda
from bingo.Base.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.ParallelArchipelago import ParallelArchipelago


POP_SIZE = 5
SELECTION_SIZE = 10
VALUE_LIST_SIZE = 10
OFFSPRING_SIZE = 20
ERROR_TOL = 10e-6

COMM = MPI.COMM_WORLD
COMM_RANK = COMM.Get_rank()
COMM_SIZE = COMM.Get_size()


class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.values)
        self.eval_count += 1
        return fitness


class NumberGenerator:
    def __init__(self, num):
        self.num = num

    def __call__(self):
        return self.num


def evol_alg():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(NumberGenerator(-1))
    selection = Tournament(SELECTION_SIZE)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    return MuPlusLambda(evaluator, selection, crossover, mutation,
                        0.2, 0.4, OFFSPRING_SIZE)


def num_island(num, pop_size=POP_SIZE):
    generator = MultipleValueChromosomeGenerator(NumberGenerator(num),
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg(), generator, pop_size)


def perfect_individual():
    generator = MultipleValueChromosomeGenerator(NumberGenerator(0),
                                                 VALUE_LIST_SIZE)
    return generator()


def test_best_individual_returned():
    island = num_island(COMM_RANK + 1)
    if COMM_RANK == 0:
        island.load_population([perfect_individual()], replace=False)
    archipelago = ParallelArchipelago(island)
    return mpi_assert_equal(archipelago.get_best_individual().fitness, 0)


def test_best_fitness_returned():
    island = num_island(COMM_RANK + 1)
    if COMM_RANK == 0:
        island.load_population([perfect_individual()], replace=False)
    archipelago = ParallelArchipelago(island)
    return mpi_assert_equal(archipelago.get_best_fitness(), 0)


def test_potential_hof_members():
    island_a = Mock(hall_of_fame=[COMM_RANK, COMM_RANK])
    archipelago = ParallelArchipelago(num_island(1))
    archipelago._island = island_a
    actual_members = archipelago._get_potential_hof_members()
    expected_memebers = [i for i in range(COMM_SIZE) for _ in range(2)]
    return mpi_assert_equal(actual_members, expected_memebers)


def test_island_migration_doesnt_chane_pop_size():
    island = num_island(COMM_RANK, pop_size=COMM_RANK + 2)
    archipelago = ParallelArchipelago(island)
    archipelago._coordinate_migration_between_islands()
    expected_pop = (COMM_SIZE * (COMM_SIZE - 1)) // 2 + (2 * COMM_SIZE)
    total_pop_after = COMM.allreduce(len(archipelago._island.population),
                                     MPI.SUM)
    return mpi_assert_equal(total_pop_after, expected_pop)


def test_island_migration():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island)
    archipelago._coordinate_migration_between_islands()

    native_individual_values = [COMM_RANK]*VALUE_LIST_SIZE
    non_native_indv_found = False
    for individual in archipelago._island.population:
        if individual.values != native_individual_values:
            non_native_indv_found = True
            break
    has_unpaired_island = COMM_SIZE % 2 == 1
    if has_unpaired_island:
        return mpi_assert_exactly_n_false(non_native_indv_found, 1)
    return mpi_assert_true(non_native_indv_found)


def test_blocking_fitness_eval_count():
    steps = 1
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, non_blocking=False)
    archipelago.evolve(steps)
    expected_evaluations = COMM_SIZE * (POP_SIZE + steps * OFFSPRING_SIZE)
    actual_evaluations = archipelago.get_fitness_evaluation_count()
    return mpi_assert_equal(actual_evaluations, expected_evaluations)


def test_non_blocking_evolution():
    steps = 200
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                      non_blocking=True)
    archipelago.evolve(steps)
    island_age = archipelago._island.generational_age
    archipelago_age = archipelago.generational_age
    return mpi_assert_mean_near(island_age, archipelago_age, rel=0.1)


def test_convergence():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                      non_blocking=True)
    result = archipelago.evolve_until_convergence(max_generations=100,
                                                  fitness_threshold=0,
                                                  convergence_check_frequency=25)
    return mpi_assert_true(result.success)


# ============================================================================


def mpi_assert_equal(actual, expected):
    equal = actual == expected
    if not equal:
        message = "\tproc {}:  {} != {}\n".format(COMM_RANK, actual, expected)
    else:
        message = ""
    all_equals = COMM.allgather(equal)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_equals), all_messages


def mpi_assert_true(value):
    if not value:
        message = "\tproc {}: False, expected True\n".format(COMM_RANK)
    else:
        message = ""
    all_values = COMM.allgather(value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_values), all_messages


def mpi_assert_exactly_n_false(value, n):
    all_values = COMM.allgather(value)
    if sum(all_values) == len(all_values) - n:
        return True, ""

    message = "\tproc {}: {}\n".format(COMM_RANK, value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    all_messages = "\tExpected exactly " + str(n) + " False\n" + all_messages
    return False, all_messages


def mpi_assert_mean_near(value, expected_mean, rel=1e-6, abs=None):
    actual_mean = COMM.allreduce(value, op=MPI.SUM)
    actual_mean /= COMM_SIZE
    allowable_error = rel * expected_mean
    if abs is not None:
        allowable_error = max(allowable_error, abs)

    if -allowable_error <= actual_mean - expected_mean <= allowable_error:
        return True, ""

    message = "\tproc {}:  {}\n".format(COMM_RANK, value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    all_messages += "\tMean {} != {} +- {}".format(actual_mean, expected_mean,
                                                   allowable_error)
    return False, all_messages


def run_t(test_name, test_func):
    if COMM_RANK == 0:
        print(test_name, end=" ")
    success, message = test_func()
    if success:
        if COMM_RANK == 0:
            print(".")
    else:
        if COMM_RANK == 0:
            print("F")
            print(message, end=" ")
    return success


def driver():
    results = []
    tests = [(name, func)
             for name, func in inspect.getmembers(sys.modules[__name__],
                                                  inspect.isfunction)
             if "test" in name]
    if COMM_RANK == 0:
        print("========== collected", len(tests), "items ==========")

    for name, func in tests:
        results.append(run_t(name, func))

    num_success = sum(results)
    num_failures = len(results) - num_success
    if COMM_RANK == 0:
        print("==========", end="  ")
        if num_failures > 0:
            print(num_failures, "failed,", end=" ")
        print(num_success, "passed ==========")


if __name__ == "__main__":
    driver()