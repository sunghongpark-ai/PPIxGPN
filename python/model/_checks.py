import math
import numbers

import numpy as np
from scipy import sparse

from .errors import PPIxGPNError

NUMERIC_KINDS = "biuf"


def as_array(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def as_matrix(value, name, *, rows=None, columns=None, finite=True, nonempty=False, bounds=None):
    array = as_array(value)
    if array.dtype.kind not in NUMERIC_KINDS or array.ndim != 2:
        raise PPIxGPNError("PPIxGPN:InvalidInput", f"{name} must be a real numeric matrix.")
    array = array.astype(np.float64, copy=False)
    if rows is not None and array.shape[0] != rows:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", f"{name} must have {rows} rows, found {array.shape[0]}.")
    if columns is not None and array.shape[1] != columns:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", f"{name} must have {columns} columns, found {array.shape[1]}.")
    if nonempty and array.size == 0:
        raise PPIxGPNError("PPIxGPN:InvalidInput", f"{name} must not be empty.")
    if finite and not np.isfinite(array).all():
        raise PPIxGPNError("PPIxGPN:InvalidInput", f"{name} must contain only finite values.")
    if bounds is not None and not ((array >= bounds[0]) & (array <= bounds[1])).all():
        raise PPIxGPNError("PPIxGPN:InvalidInput", f"{name} values must lie in [{bounds[0]}, {bounds[1]}].")
    return array


def as_vector(value, name, *, size=None, finite=True):
    array = as_array(value)
    if array.dtype.kind not in NUMERIC_KINDS or array.ndim > 2 or (array.ndim == 2 and min(array.shape) > 1):
        raise PPIxGPNError("PPIxGPN:InvalidInput", f"{name} must be a real numeric vector.")
    array = array.astype(np.float64, copy=False).reshape(-1)
    if size is not None and array.size != size:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", f"{name} must have {size} elements, found {array.size}.")
    if finite and not np.isfinite(array).all():
        raise PPIxGPNError("PPIxGPN:InvalidInput", f"{name} must contain only finite values.")
    return array


def _scalar(value, name):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be a real number.")
    return value


def _check_bounds(value, name, minimum, maximum, open_minimum, open_maximum):
    if minimum is not None and (value <= minimum if open_minimum else value < minimum):
        relation = "greater than" if open_minimum else "at least"
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be {relation} {minimum}.")
    if maximum is not None and (value >= maximum if open_maximum else value > maximum):
        relation = "less than" if open_maximum else "at most"
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be {relation} {maximum}.")


def integer_option(value, name, *, minimum=None, maximum=None):
    value = _scalar(value, name)
    if not math.isfinite(value) or value != math.floor(value):
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be an integer.")
    value = int(value)
    _check_bounds(value, name, minimum, maximum, False, False)
    return value


def real_option(value, name, *, minimum=None, maximum=None, open_minimum=False, open_maximum=False,
                infinite=False):
    value = float(_scalar(value, name))
    if math.isnan(value) or (math.isinf(value) and not infinite):
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be {'a number' if infinite else 'finite'}.")
    _check_bounds(value, name, minimum, maximum, open_minimum, open_maximum)
    return value


def real_list_option(values, name, **bounds):
    array = np.atleast_1d(np.asarray(values, dtype=object))
    if array.ndim != 1 or array.size == 0:
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be a nonempty sequence of numbers.")
    return tuple(real_option(value, name, **bounds) for value in array)


def choice_option(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be one of: {', '.join(choices)}.")
    return value


def flag_option(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"{name} must be True or False.")
    return bool(value)


def field(container, key, owner):
    try:
        return container[key]
    except (KeyError, TypeError, IndexError):
        raise PPIxGPNError("PPIxGPN:MissingField", f"{owner} must define '{key}'.") from None
