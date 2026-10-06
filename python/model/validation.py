import math

import numpy as np

from ._checks import as_matrix, field, flag_option, integer_option, real_option
from .errors import PPIxGPNError
from .metrics import METRICS, evaluate_risk
from .pipeline import ppixgpn
from .streams import SEED_LIMIT, random_stream
from .table import Table

TRAIN_OPTIONS = ("epoch", "rate", "gamma", "threshold", "gradient", "phi_min")


def cross_validate(data, *, repeat=1, fold=5, valid_ratio=0.2, seed=1, train=None, cutoff=0.5, verbose=False):
    repeat = integer_option(repeat, "repeat", minimum=1)
    fold = integer_option(fold, "fold", minimum=2)
    valid_ratio = real_option(valid_ratio, "valid_ratio", minimum=0, maximum=1, open_minimum=True, open_maximum=True)
    seed = integer_option(seed, "seed", minimum=0)
    train = {} if train is None else dict(train)
    cutoff = real_option(cutoff, "cutoff", minimum=0, maximum=1)
    verbose = flag_option(verbose, "verbose")
    if seed + repeat - 1 >= SEED_LIMIT:
        raise PPIxGPNError("PPIxGPN:InvalidOption", "seed + repeat - 1 must be below 2^32.")
    if "seed" in train:
        raise PPIxGPNError("PPIxGPN:InvalidOption",
                           "Initialization seeds are drawn from the cross-validation stream; remove seed from train.")
    unknown = sorted(set(train) - set(TRAIN_OPTIONS))
    if unknown:
        raise PPIxGPNError("PPIxGPN:InvalidOption", f"Unsupported training options: {', '.join(unknown)}.")

    Xdata = as_matrix(field(data, "Xdata", "data"), "data['Xdata']")
    Ydata = as_matrix(field(data, "Ydata", "data"), "data['Ydata']")
    ppi_data = field(data, "ppi_data", "data")
    num_participant = Xdata.shape[1]
    num_target = Ydata.shape[1]
    if num_participant < fold:
        raise PPIxGPNError("PPIxGPN:InvalidOption", "fold cannot exceed the number of participants.")
    if "target" in data:
        target = [str(name) for name in np.atleast_1d(data["target"])]
    else:
        target = [f"Y{k}" for k in range(1, num_target + 1)]
    names = target + ["Mean"]

    assignment = np.zeros((num_participant, repeat), dtype=np.int64)
    risk = np.full((num_participant, num_target, repeat), np.nan)
    best_epoch = np.zeros((repeat, fold), dtype=np.int64)
    repeat_column, fold_column, target_column, metric_value = [], [], [], []
    for r in range(1, repeat + 1):
        stream = random_stream(seed + r - 1)
        order = np.argsort(stream.random_sample(num_participant), kind="stable")
        labels = np.empty(num_participant, dtype=np.int64)
        labels[order] = np.arange(num_participant) % fold + 1
        assignment[:, r - 1] = labels
        for k in range(1, fold + 1):
            idx_test = np.flatnonzero(labels == k)
            rest = np.flatnonzero(labels != k)
            permutation = np.argsort(stream.random_sample(rest.size), kind="stable")
            num_valid = _round_half_away(valid_ratio * rest.size)
            if num_valid < 1 or num_valid >= rest.size:
                raise PPIxGPNError("PPIxGPN:InvalidOption", "valid_ratio leaves an empty training or validation set.")
            idx_valid = np.sort(rest[permutation[:num_valid]])
            idx_train = np.sort(rest[permutation[num_valid:]])
            init_seed = math.floor(stream.random_sample() * 2 ** 31)
            fit = ppixgpn(Xdata, Ydata, ppi_data, idx_train, idx_valid, idx_test, **train, seed=init_seed)
            risk[idx_test, :, r - 1] = fit.pred_risk
            best_epoch[r - 1, k - 1] = fit.history["BestEpoch"]
            metrics = evaluate_risk(fit.pred_risk, Ydata[idx_test], cutoff=cutoff, target=target)
            repeat_column += [r] * len(names)
            fold_column += [k] * len(names)
            target_column += names
            metric_value.append(np.column_stack([metrics[name] for name in METRICS]))
            if verbose:
                print(f"Repeat {r}/{repeat}, fold {k}/{fold}: best epoch {fit.history['BestEpoch']}, "
                      f"mean test AUROC {metrics['AUROC'][-1]:.4f}")

    metric_value = np.vstack(metric_value)
    target_column = np.array(target_column, dtype=str)
    fold_metrics = Table({
        "Repeat": np.array(repeat_column, dtype=np.int64),
        "Fold": np.array(fold_column, dtype=np.int64),
        "Target": target_column,
        **{name: metric_value[:, j] for j, name in enumerate(METRICS)},
    })
    summary = {"Target": names}
    for j, name in enumerate(METRICS):
        selected = [metric_value[target_column == label, j] for label in names]
        summary[f"{name}_mean"] = [np.mean(values) for values in selected]
        summary[f"{name}_sd"] = [np.std(values, ddof=1) for values in selected]
    options = {"repeat": repeat, "fold": fold, "valid_ratio": valid_ratio, "seed": seed, "train": train,
               "cutoff": cutoff, "verbose": verbose}
    return {"Assignment": assignment, "Risk": risk, "BestEpoch": best_epoch, "FoldMetrics": fold_metrics,
            "Summary": Table(summary), "Options": options}


def _round_half_away(value):
    whole = math.floor(value)
    return whole + (1 if value - whole >= 0.5 else 0)
