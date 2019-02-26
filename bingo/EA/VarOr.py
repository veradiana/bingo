"""Variation where crossover, mutation, or replication may occur


VarOr.py allows for definition of a variation by crossover, replication, and mutation
probabilities. Offspring may be the result of either crossover, replication :or: mutation,
hence the name. Only one may occur at a time.
"""

import numpy as np

from bingo.Base.Variation import Variation
from bingo.Util.ArgumentValidation import argument_validation


class VarOr(Variation):

    """Variation where either crossover, mutation, or
    replication may occur

    Parameters
    ----------
    crossover : Crossover
                Crossover function class used in variation
    mutation : Mutation
               Mutation function class used in variation
    crossover_probability : float
                            Probabiloity that crossover will occur on an
                            individual
    mutation_probability : float
                           Probability that mutation will occur on an individual


    Attributes
    ----------
    crossover_offspring : array of bool
                          list indicating whether the corresponding member of
                          the last offspring was a result of crossover
    mutation_offspring : array of bool
                         list indicating whether the corresponding member of
                         the last offspring was a result of mutation


    """
    @argument_validation(crossover_probability={">=": 0, "<=": 1},
                         mutation_probability={">=": 0, "<=": 1})
    def __init__(self, crossover, mutation, crossover_probability,
                 mutation_probability):
        super().__init__()
        if (crossover_probability + mutation_probability) >= 1:
            raise ValueError('The sum of crossover and mutation probabilities must be less than 1.0')
        self._crossover = crossover
        self._mutation = mutation
        self._crossover_probability = crossover_probability
        self._mutation_probability = mutation_probability
        self._replication_probability = 1 - crossover_probability - mutation_probability

    @argument_validation(number_offspring={">=": 0})
    def __call__(self, population, number_offspring):
        """Performs "Or" variation on a population.

        Parameters
        ----------
        population : list of Chromosome
                     The population on which to perform selection
        number_offspring : int
                           number of offspring to produce

        Returns
        -------
        list of Chromosome :
            The offspring of the population
        """
        offspring = []
        self.crossover_offspring = np.zeros(number_offspring, bool)
        self.mutation_offspring = np.zeros(number_offspring, bool)
        for i in range(number_offspring):
            choice = np.random.rand()
            if choice <= self._mutation_probability:
                mutant = self._get_random_parent(population)
                mutant = self._mutation(mutant)
                offspring.append(mutant)
                self.mutation_offspring[i] = True

            elif choice <= (self._mutation_probability + self._crossover_probability):
                parent_1 = self._get_random_parent(population)
                parent_2 = self._get_random_parent(population)
                child_1, child_2 = self._crossover(parent_1, parent_2)
                offspring.append(child_1)
                self.crossover_offspring[i] = True

            else:
                child = self._get_random_parent(population)
                offspring.append(child)

        return offspring

    def _get_random_parent(self, population):
        return population[np.random.randint(len(population))].copy()