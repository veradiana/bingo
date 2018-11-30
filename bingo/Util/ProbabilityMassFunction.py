"""
This module implements a probability mass function from which single samples
can be drawn
"""
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class ProbabilityMassFunction:
    """
    The ProbabilityMassFunction (PMF) class is designed to allow for easy
    creation and use of a probability mass function.  Items and associated
    probability weights are given. Samples (items) can then be drawn from the
    pmf according to their relative weights.

    Parameters
    ----------
    items : list, optional
            The starting items in the PMF.
    weights : list-like of numeric, optional
              The relative weights of the items. The default is even weighting.

    Attributes
    ----------
    items : lit
            The current items in the PMF
    normalized_weights : list-like numeric
                         The probabilities of items
    """

    def __init__(self, items=None, weights=None):
        """

        """
        if items is None:
            items = []

        self._is_init_param_listlike(items)
        self.items = items

        if weights is None:
            weights = self._get_default_weights()

        self._is_init_param_listlike(weights)
        self._is_weights_same_size_as_items(weights)
        self._total_weight, self.normalized_weights = \
            self._normalize_weights(weights)

    @staticmethod
    def _is_init_param_listlike(init_param):
        if not hasattr(init_param, "__len__") or isinstance(init_param, str):
            LOGGER.error("Initialization of ProbabilityMassFunction with "
                         "non-listlike parameters")
            LOGGER.error(str(init_param))
            raise ValueError

    def _get_default_weights(self):
        n_items = len(self.items)
        if n_items > 0:
            weights = np.full(n_items, 1.0 / n_items)
        else:
            weights = np.array([])
        return weights

    def _is_weights_same_size_as_items(self, weights):
        if len(weights) != len(self.items):
            LOGGER.error("Initialization of ProbabilityMassFunction with "
                         "items and weights of different dimensions")
            LOGGER.error("items = %s", self.items)
            LOGGER.error("weights = %s", weights)
            raise ValueError

    @staticmethod
    def _normalize_weights(weights):
        total_weight = ProbabilityMassFunction._calculate_total_weight(weights)
        normalized_weights = np.array(weights) / total_weight
        ProbabilityMassFunction._check_valid_weights(normalized_weights,
                                                     weights)
        return total_weight, normalized_weights

    @staticmethod
    def _check_valid_weights(normalized_weights, weights):
        if normalized_weights.size > 0:
            if not np.isclose(np.sum(normalized_weights), 1.0) or \
                            np.min(normalized_weights) < 0.0:
                LOGGER.error("Invalid weights encountered in "
                             "ProbabilityMassFunction")
                LOGGER.error("weights = %s", weights)
                raise ValueError

    @staticmethod
    def _calculate_total_weight(weights):
        try:
            total_weight = np.sum(weights)
        except TypeError:
            LOGGER.error("Initialization of ProbabilityMassFunction with "
                         "non-numeric weights")
            LOGGER.error("weights = %s", weights)
            raise TypeError
        return total_weight

    def add_item(self, new_item, new_weight=None):
        """Adds a single item to the PMF.

        Parameters
        ----------
        new_item :
                   The item to be added.
        new_weight : numeric, optional
                     The weight associated with the item. The default is the
                     average weight of the other items.
        """
        self.items.append(new_item)

        if new_weight is None:
            new_weight = self._get_mean_current_weight()

        weights = self._total_weight * self.normalized_weights
        weights = np.append(weights, new_weight)

        self._total_weight, self.normalized_weights = \
            self._normalize_weights(weights)

    def _get_mean_current_weight(self):
        if not self.normalized_weights:
            return 1.0
        return self._total_weight / len(self.normalized_weights)

    def draw_sample(self):
        """Draw a sample from the PMF

        Draw a random sample from the PMF according to the probabilities
        associated with weighting of items.

        Returns
        -------
            A single item
        """
        return np.random.choice(self.items, 1, p=self.normalized_weights)[0]
