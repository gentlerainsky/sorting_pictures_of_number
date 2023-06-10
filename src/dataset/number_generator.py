import clrs
import numpy as np
from enum import Enum


class AlgorithmName(Enum):
    INSERTION_SORT = 0
    BUBBLE_SORT = 1


def generate_data(algorithm_name: AlgorithmName, n, num_items, min, max):
    dataset = []
    targets = []
    for _ in range(n):
        items = np.random.randint(min, max, num_items)
        # items = np.random.uniform(0, 1, num_items)
        if algorithm_name == AlgorithmName.INSERTION_SORT:
            target, probes = clrs._src.algorithms.sorting.insertion_sort(items)
        elif algorithm_name == AlgorithmName.BUBBLE_SORT:
            target, probes = clrs._src.algorithms.sorting.bubble_sort(items)
        dataset.append(probes)
        targets.append(target)
    return targets, dataset
