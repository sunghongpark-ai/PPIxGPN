import numpy as np

from ._checks import integer_option

MATLAB_ZERO_SEED = 5489
SEED_LIMIT = 2 ** 32


def random_stream(seed=None):
    if seed is None:
        return np.random.mtrand._rand
    seed = integer_option(seed, "seed", minimum=0, maximum=SEED_LIMIT - 1)
    return np.random.RandomState(MATLAB_ZERO_SEED if seed == 0 else seed)
