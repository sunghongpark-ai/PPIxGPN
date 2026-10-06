from typing import NamedTuple

import numpy as np

from ._checks import as_array, as_matrix, choice_option, integer_option, real_option
from .errors import PPIxGPNError
from .model import GRADIENTS, model_train, risk_predict
from .network import ppi_network
from .parameters import param_init
from .streams import SEED_LIMIT, random_stream


class PPIxGPNResult(NamedTuple):
    pred_risk: np.ndarray
    model_param: np.ndarray
    history: dict
    dataset: dict
    parameter: dict


def ppixgpn(Xdata, Ydata, ppi_data, idx_train, idx_valid, idx_test, *, epoch=1000, rate=0.001, gamma=0.01,
            threshold=0.4, seed=None, gradient="exact", phi_min=-np.inf):
    Xdata = as_matrix(Xdata, "Xdata")
    Ydata = as_matrix(Ydata, "Ydata", columns=4, bounds=(0, 1))
    epoch = integer_option(epoch, "epoch", minimum=1)
    rate = real_option(rate, "rate", minimum=0, open_minimum=True)
    gamma = real_option(gamma, "gamma", minimum=0)
    threshold = real_option(threshold, "threshold", minimum=0, open_minimum=True)
    if seed is not None:
        seed = integer_option(seed, "seed", minimum=0, maximum=SEED_LIMIT - 1)
    gradient = choice_option(gradient, "gradient", GRADIENTS)
    phi_min = real_option(phi_min, "phi_min", infinite=True)

    num_protein, num_participant = Xdata.shape
    if Ydata.shape[0] != num_participant:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", "Ydata must have one row per participant: "
                           f"expected {num_participant} rows, found {Ydata.shape[0]}.")
    if np.shape(ppi_data) != (num_protein, num_protein):
        raise PPIxGPNError("PPIxGPN:SizeMismatch",
                           f"ppi_data must be {num_protein}-by-{num_protein} to match the proteins in Xdata.")

    idx_train = _split_index(idx_train, num_participant, "idx_train", True)
    idx_valid = _split_index(idx_valid, num_participant, "idx_valid", True)
    idx_test = _split_index(idx_test, num_participant, "idx_test", False)
    if (np.intersect1d(idx_train, idx_valid).size or np.intersect1d(idx_train, idx_test).size
            or np.intersect1d(idx_valid, idx_test).size):
        raise PPIxGPNError("PPIxGPN:OverlappingSplit",
                           "idx_train, idx_valid, and idx_test must select disjoint participants.")

    dataset = {
        "Xtrain": Xdata[:, idx_train], "Ytrain": Ydata[idx_train],
        "Xvalid": Xdata[:, idx_valid], "Yvalid": Ydata[idx_valid],
        "Xtest": Xdata[:, idx_test], "Ytest": Ydata[idx_test],
        "Lppi": ppi_network(ppi_data, threshold=threshold),
    }
    stream = random_stream(seed)
    parameter = {
        "Uppi": param_init(num_protein, 1),
        "Babt": param_init(num_protein, 0, stream),
        "Bgfa": param_init(num_protein, 0, stream),
        "Bnfl": param_init(num_protein, 0, stream),
        "Btau": param_init(num_protein, 0, stream),
        "epoch": epoch,
        "rate": rate,
        "gamma": gamma,
    }
    model_param, history = model_train(dataset, parameter, gradient=gradient, phi_min=phi_min)
    pred_risk = risk_predict(dataset, parameter, model_param)
    return PPIxGPNResult(pred_risk, model_param, history, dataset, parameter)


def _split_index(index, num_participant, name, required):
    array = as_array(index)
    if array.ndim > 2 or (array.ndim == 2 and min(array.shape) > 1):
        raise PPIxGPNError("PPIxGPN:InvalidIndex", f"{name} must be a vector of indices or a boolean mask.")
    array = array.reshape(-1)
    if array.dtype.kind == "b":
        if array.size != num_participant:
            raise PPIxGPNError("PPIxGPN:InvalidIndex",
                               f"Boolean {name} must contain one element per participant ({num_participant}).")
        selected = np.flatnonzero(array)
    elif array.size == 0:
        selected = np.zeros(0, dtype=np.intp)
    elif (array.dtype.kind not in "iuf" or not np.isfinite(array).all() or (array != np.floor(array)).any()
          or array.min() < 0 or array.max() >= num_participant):
        raise PPIxGPNError("PPIxGPN:InvalidIndex",
                           f"{name} must contain zero-based participant indices from 0 to {num_participant - 1}.")
    else:
        selected = array.astype(np.intp)
    if required and selected.size == 0:
        raise PPIxGPNError("PPIxGPN:EmptySplit", f"{name} must select at least one participant.")
    return selected
