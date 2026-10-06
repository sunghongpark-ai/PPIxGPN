import warnings
from typing import NamedTuple

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.linalg.lapack import get_lapack_funcs

from ._checks import as_matrix, as_vector, choice_option, field, integer_option, real_option
from .errors import PPIxGPNError
from .optimizer import adam_init, adam_step
from .parameters import stack_param

GRADIENTS = ("exact", "legacy")


class _Problem(NamedTuple):
    lppi: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    gamma: float
    legacy: bool
    num_protein: int
    num_target: int


def model_train(dataset, parameter, gradient="exact", phi_min=-np.inf):
    gradient = choice_option(gradient, "gradient", GRADIENTS)
    phi_min = real_option(phi_min, "phi_min", infinite=True)
    problem, weight = _train_problem(dataset, parameter, gradient)
    num_protein = problem.num_protein
    x_valid = as_matrix(field(dataset, "Xvalid", "dataset"), "dataset['Xvalid']", rows=num_protein, nonempty=True)
    y_valid = as_matrix(field(dataset, "Yvalid", "dataset"), "dataset['Yvalid']", rows=x_valid.shape[1],
                        columns=problem.num_target, bounds=(0, 1))
    num_epoch = integer_option(field(parameter, "epoch", "parameter"), "parameter['epoch']", minimum=1)
    rate = real_option(field(parameter, "rate", "parameter"), "parameter['rate']", minimum=0, open_minimum=True)

    weight[:num_protein] = np.maximum(weight[:num_protein], phi_min)
    adam_param = adam_init(weight, rate)
    loss_train_history = np.zeros((num_epoch, problem.num_target))
    loss_valid_history = np.zeros((num_epoch, problem.num_target))
    best_epoch, best_loss = 0, np.inf
    model_param = weight.copy()
    grad_param = np.zeros_like(weight)

    for epoch in range(1, num_epoch + 1):
        if epoch > 1:
            weight, adam_param = adam_step(weight, grad_param, adam_param)
            weight[:num_protein] = np.maximum(weight[:num_protein], phi_min)
        loss_train, readout, grad_param = _model_evaluate(weight, problem)
        loss_valid = _cross_entropy(x_valid.T @ readout, y_valid)
        if not (np.isfinite(loss_train).all() and np.isfinite(loss_valid).all() and np.isfinite(grad_param).all()):
            raise PPIxGPNError("PPIxGPN:NonfiniteValue",
                               f"Training produced a nonfinite loss or gradient at epoch {epoch}.")
        loss_train_history[epoch - 1] = loss_train
        loss_valid_history[epoch - 1] = loss_valid
        score = loss_valid.mean()
        if score < best_loss:
            best_epoch, best_loss = epoch, float(score)
            model_param = weight.copy()

    history = {"LossTrain": loss_train_history, "LossValid": loss_valid_history,
               "BestEpoch": best_epoch, "BestLoss": best_loss}
    return model_param, history


def model_gradient(dataset, parameter, model_param, gradient="exact"):
    gradient = choice_option(gradient, "gradient", GRADIENTS)
    problem, _ = _train_problem(dataset, parameter, gradient)
    weight = as_vector(model_param, "model_param", size=problem.num_protein * (problem.num_target + 1))
    loss_train, _, grad_param = _model_evaluate(weight, problem)
    return grad_param, loss_train


def risk_predict(dataset, parameter, model_param, return_effect=False):
    _, param_size = stack_param(parameter)
    num_protein = int(param_size[0, 0])
    num_target = param_size.shape[0] - 1
    lppi = as_matrix(field(dataset, "Lppi", "dataset"), "dataset['Lppi']", rows=num_protein, columns=num_protein)
    x_test = as_matrix(field(dataset, "Xtest", "dataset"), "dataset['Xtest']", rows=num_protein)
    weight = as_vector(model_param, "model_param", size=num_protein * (num_target + 1))
    uppi, bset = _split_weight(weight, num_protein, num_target)
    factor, _, readout = _propagation_readout(lppi, uppi, bset)
    pred_risk = _sigmoid(x_test.T @ readout)
    if not return_effect:
        return pred_risk
    return pred_risk, _solve(factor, uppi[:, None] * x_test)


def _train_problem(dataset, parameter, gradient):
    weight, param_size = stack_param(parameter)
    num_protein = int(param_size[0, 0])
    num_target = param_size.shape[0] - 1
    lppi = as_matrix(field(dataset, "Lppi", "dataset"), "dataset['Lppi']", rows=num_protein, columns=num_protein)
    x_train = as_matrix(field(dataset, "Xtrain", "dataset"), "dataset['Xtrain']", rows=num_protein, nonempty=True)
    y_train = as_matrix(field(dataset, "Ytrain", "dataset"), "dataset['Ytrain']", rows=x_train.shape[1],
                        columns=num_target, bounds=(0, 1))
    gamma = real_option(field(parameter, "gamma", "parameter"), "parameter['gamma']", minimum=0)
    problem = _Problem(lppi, x_train, y_train, gamma, gradient == "legacy", num_protein, num_target)
    return problem, weight


def _model_evaluate(weight, problem):
    uppi, bset = _split_weight(weight, problem.num_protein, problem.num_target)
    factor, adjoint, readout = _propagation_readout(problem.lppi, uppi, bset)
    logit = problem.x_train.T @ readout
    loss_train = _cross_entropy(logit, problem.y_train)
    num_train = problem.x_train.shape[1]
    x_residual = problem.x_train @ (_sigmoid(logit) - problem.y_train)
    grad_bset = _solve(factor, uppi[:, None] * x_residual) / num_train
    if problem.legacy:
        twice = _solve(factor, _solve(factor, x_residual, transpose=True), transpose=True)
        grad_uppi = np.sum(bset * (problem.lppi.T @ twice), axis=1) / num_train
    else:
        grad_uppi = np.sum(adjoint * _solve(factor, problem.lppi @ x_residual), axis=1) / num_train
    grad_param = np.concatenate((grad_uppi, grad_bset.ravel(order="F"))) + 2 * problem.gamma * weight
    return loss_train, readout, grad_param


def _split_weight(weight, num_protein, num_target):
    return weight[:num_protein], weight[num_protein:].reshape((num_protein, num_target), order="F")


def _propagation_readout(lppi, uppi, bset):
    system = lppi.copy()
    system[np.diag_indices_from(system)] += uppi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        factor = lu_factor(system, check_finite=False)
    gecon = get_lapack_funcs("gecon", (factor[0],))
    reciprocal_condition, info = gecon(factor[0], np.linalg.norm(system, 1), norm="1")
    if info != 0 or not reciprocal_condition >= np.finfo(np.float64).eps:
        raise PPIxGPNError("PPIxGPN:IllConditionedSystem",
                           "The propagation system Lppi + diag(Uppi) is singular to working precision.")
    adjoint = _solve(factor, bset, transpose=True)
    return factor, adjoint, uppi[:, None] * adjoint


def _solve(factor, right_hand_side, transpose=False):
    if right_hand_side.size == 0:
        return np.zeros(right_hand_side.shape)
    return lu_solve(factor, right_hand_side, trans=1 if transpose else 0, check_finite=False)


def _sigmoid(logit):
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-logit))


def _cross_entropy(logit, label):
    return np.mean(np.maximum(logit, 0) - label * logit + np.log1p(np.exp(-np.abs(logit))), axis=0)
