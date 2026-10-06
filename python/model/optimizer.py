import numpy as np

from ._checks import as_vector, field, real_option
from .errors import PPIxGPNError


def adam_init(param_set, learn_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
    size = as_vector(param_set, "param_set", finite=False).size
    return {
        "alpha": real_option(learn_rate, "learn_rate", minimum=0, open_minimum=True),
        "beta1": real_option(beta1, "beta1", minimum=0, maximum=1, open_maximum=True),
        "beta2": real_option(beta2, "beta2", minimum=0, maximum=1, open_maximum=True),
        "epsilon": real_option(epsilon, "epsilon", minimum=0, open_minimum=True),
        "t": 0,
        "m": np.zeros(size),
        "v": np.zeros(size),
    }


def weight_update(w, g, param):
    w = as_vector(w, "w", finite=False)
    g = as_vector(g, "g", finite=False)
    moments = [as_vector(field(param, name, "param"), f"param['{name}']", finite=False) for name in ("m", "v")]
    if not w.shape == g.shape == moments[0].shape == moments[1].shape:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", "The weights, gradient, and Adam moments must have the same length.")
    return adam_step(w, g, {**param, "m": moments[0], "v": moments[1]})


def adam_step(w, g, param):
    t = param["t"] + 1
    m = param["beta1"] * param["m"] + (1 - param["beta1"]) * g
    v = param["beta2"] * param["v"] + (1 - param["beta2"]) * g ** 2
    m_hat = m / (1 - param["beta1"] ** t)
    v_hat = v / (1 - param["beta2"] ** t)
    return w - param["alpha"] * m_hat / (np.sqrt(v_hat) + param["epsilon"]), {**param, "t": t, "m": m, "v": v}
