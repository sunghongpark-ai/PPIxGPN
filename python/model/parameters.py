import math

import numpy as np

from ._checks import as_array, as_vector, field, integer_option, real_option
from .errors import PPIxGPNError
from .streams import random_stream

PARAMETER_FIELDS = ("Uppi", "Babt", "Bgfa", "Bnfl", "Btau")


def param_init(num_protein, mu, stream=None):
    num_protein = integer_option(num_protein, "num_protein", minimum=1)
    mu = real_option(mu, "mu")
    if mu != 0:
        return np.full(num_protein, mu)
    source = random_stream() if stream is None else stream
    return (2 * source.random_sample(num_protein) - 1) * math.sqrt(6 / (num_protein + 1))


def stack_param(parameter):
    num_protein = np.size(field(parameter, "Uppi", "parameter"))
    if num_protein == 0:
        raise PPIxGPNError("PPIxGPN:InvalidParameter", "parameter['Uppi'] must not be empty.")
    blocks = [as_vector(field(parameter, name, "parameter"), f"parameter['{name}']", size=num_protein)
              for name in PARAMETER_FIELDS]
    param_size = np.tile(np.array([num_protein, 1]), (len(PARAMETER_FIELDS), 1))
    return np.concatenate(blocks), param_size


def resize_param(param_data, param_size):
    data = as_vector(param_data, "param_data", finite=False)
    shape = as_array(param_size)
    if (shape.dtype.kind not in "iuf" or shape.ndim != 2 or shape.shape[1] != 2
            or not np.isfinite(shape).all() or (shape < 0).any() or (shape != np.floor(shape)).any()):
        raise PPIxGPNError("PPIxGPN:InvalidInput", "param_size must be a k-by-2 array of nonnegative integers.")
    shape = shape.astype(np.int64)
    count = shape.prod(axis=1)
    if count.sum() != data.size:
        raise PPIxGPNError("PPIxGPN:SizeMismatch",
                           f"param_size describes {count.sum()} elements, but param_data has {data.size}.")
    stop = np.cumsum(count)
    blocks = []
    for (num_row, num_column), first, last in zip(shape, stop - count, stop):
        block = data[first:last].copy()
        blocks.append(block if num_column == 1 else block.reshape((num_row, num_column), order="F"))
    return tuple(blocks)
