import math

import numpy as np

from ._checks import as_array, as_matrix, real_option
from .data import TARGETS
from .errors import PPIxGPNError
from .table import Table

METRICS = ("AUROC", "AUPRC", "Accuracy", "F1")


def evaluate_risk(pred_risk, label, cutoff=0.5, target=TARGETS):
    score = _column_matrix(pred_risk, "pred_risk")
    truth = _column_matrix(label, "label")
    cutoff = real_option(cutoff, "cutoff", minimum=0, maximum=1)
    target = [str(name) for name in np.atleast_1d(np.asarray(target, dtype=str))]
    if np.isnan(score).any():
        raise PPIxGPNError("PPIxGPN:InvalidInput", "pred_risk must not contain NaN.")
    if score.shape != truth.shape:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", "pred_risk and label must have the same size.")
    if not ((truth == 0) | (truth == 1)).all():
        raise PPIxGPNError("PPIxGPN:InvalidLabel", "Labels must be 0 or 1.")
    if len(target) != truth.shape[1]:
        raise PPIxGPNError("PPIxGPN:SizeMismatch", "Provide one target name per label column.")

    value = np.zeros((truth.shape[1], len(METRICS)))
    for k in range(truth.shape[1]):
        positive = truth[:, k] == 1
        predicted = score[:, k] >= cutoff
        true_positive = np.count_nonzero(predicted & positive)
        false_positive = np.count_nonzero(predicted & ~positive)
        false_negative = np.count_nonzero(~predicted & positive)
        denominator = 2 * true_positive + false_positive + false_negative
        value[k] = (_rank_area(score[:, k], positive), _average_precision(score[:, k], positive),
                    np.mean(predicted == positive), 2 * true_positive / denominator if denominator else math.nan)
    value = np.vstack((value, value.mean(axis=0)))
    return Table({"Target": target + ["Mean"], **{name: value[:, j] for j, name in enumerate(METRICS)}})


def _column_matrix(value, name):
    array = as_array(value)
    return as_matrix(array[:, None] if array.ndim == 1 else array, name, finite=False)


def _rank_area(score, positive):
    num_positive = np.count_nonzero(positive)
    num_negative = positive.size - num_positive
    if num_positive == 0 or num_negative == 0:
        return math.nan
    order = np.argsort(score, kind="stable")
    ordered = score[order]
    start = np.concatenate(([True], np.diff(ordered) != 0))
    first = np.flatnonzero(start)
    last = np.concatenate((first[1:], [ordered.size])) - 1
    group = np.cumsum(start) - 1
    rank = np.empty(score.size)
    rank[order] = (first[group] + last[group]) / 2 + 1
    return (rank[positive].sum() - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)


def _average_precision(score, positive):
    num_positive = np.count_nonzero(positive)
    if num_positive == 0:
        return math.nan
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    hit = positive[order]
    boundary = np.concatenate((np.diff(ordered) != 0, [True]))
    true_positive = np.cumsum(hit)[boundary]
    false_positive = np.cumsum(~hit)[boundary]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_positive
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
