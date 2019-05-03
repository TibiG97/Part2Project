import numpy as np
from numpy.random import randint


def permutation_test(model1_sample: list,
                     model2_sample: list,
                     r_limit: int,
                     p_value_limit: int):
    """
    Function that performs the Monte Carlo Permutation Test between samples of two different models

    :param model1_sample: class predictions of the first model
    :param model2_sample: class predictions of the second model
    :param r_limit: cardinality of the random subset of 2^N permutations
    :param p_value_limit: threshold value under which if p_value is means that the samples to come from different
    distributions
    :return: True, if the two samples come from different distributions
             False, otherwise
    """

    sample_size = len(model1_sample)
    reference_value = np.mean(model1_sample) - np.mean(model2_sample)
    s = 0

    for iterator in range(0, r_limit):
        random_permutation = randint(2, size=sample_size)
        model1_copy = model1_sample
        model2_copy = model2_sample
        for index in range(0, sample_size):
            if random_permutation[index] == 1:
                model1_copy[index], model2_copy[index] = model2_copy[index], model1_copy[index]

        current_mean = np.mean(model1_copy) - np.mean(model2_copy)
        if current_mean >= reference_value:
            s += 1

    p_value = (s + 1) / (r_limit + 1)

    if p_value < p_value_limit:
        return True
    return False
