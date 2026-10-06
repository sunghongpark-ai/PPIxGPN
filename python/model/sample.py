import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from ._checks import choice_option, flag_option, integer_option, real_list_option, real_option
from .data import SPLITS, load_dataset
from .errors import PPIxGPNError
from .metrics import METRICS, evaluate_risk
from .model import GRADIENTS, risk_predict
from .pipeline import ppixgpn
from .streams import SEED_LIMIT
from .table import Table
from .validation import cross_validate


def run_sample(*, file=None, gamma=(0.01, 0.001, 0.0001), phi_min=(0.01,), epoch=3000, rate=0.001,
               gradient="exact", seed=1, cutoff=0.5, cv_repeat=0, cv_fold=5, output=None, verbose=True):
    options = {
        "file": None if file is None else str(file),
        "gamma": real_list_option(gamma, "gamma", minimum=0),
        "phi_min": real_list_option(phi_min, "phi_min", infinite=True),
        "epoch": integer_option(epoch, "epoch", minimum=1),
        "rate": real_option(rate, "rate", minimum=0, open_minimum=True),
        "gradient": choice_option(gradient, "gradient", GRADIENTS),
        "seed": integer_option(seed, "seed", minimum=0, maximum=SEED_LIMIT - 1),
        "cutoff": real_option(cutoff, "cutoff", minimum=0, maximum=1),
        "cv_repeat": integer_option(cv_repeat, "cv_repeat", minimum=0),
        "cv_fold": integer_option(cv_fold, "cv_fold", minimum=2),
        "output": None if output is None else str(output),
        "verbose": flag_option(verbose, "verbose"),
    }
    started = time.perf_counter()
    data = load_dataset(options["file"])
    target = data["target"].tolist()
    candidate = [(gamma_value, phi_value) for phi_value in options["phi_min"] for gamma_value in options["gamma"]]
    valid_loss = np.full(len(candidate), np.nan)
    best_epoch = np.zeros(len(candidate), dtype=np.int64)
    status = []
    chosen, fit = None, None
    for c, (gamma_value, phi_value) in enumerate(candidate):
        try:
            attempt = ppixgpn(data["Xdata"], data["Ydata"], data["ppi_data"], data["idx_train"], data["idx_valid"],
                              data["idx_test"], epoch=options["epoch"], rate=options["rate"], gamma=gamma_value,
                              phi_min=phi_value, gradient=options["gradient"], seed=options["seed"])
        except PPIxGPNError as error:
            status.append(str(error))
        else:
            valid_loss[c] = attempt.history["BestLoss"]
            best_epoch[c] = attempt.history["BestEpoch"]
            status.append("ok")
            if chosen is None or valid_loss[c] < valid_loss[chosen]:
                chosen, fit = c, attempt
        if options["verbose"]:
            print(f"Candidate {c + 1}/{len(candidate)}: gamma {_number(gamma_value)}, phi_min {_number(phi_value)}, "
                  f"best epoch {best_epoch[c]}, validation loss {_number(valid_loss[c], '.6f')} ({status[c]})")
    if chosen is None:
        raise PPIxGPNError("PPIxGPN:NoCandidate", "Every candidate configuration failed; see the status messages.")

    selection = Table({
        "gamma": [value for value, _ in candidate],
        "phi_min": [value for _, value in candidate],
        "BestEpoch": best_epoch,
        "ValidLoss": valid_loss,
        "Selected": np.arange(len(candidate)) == chosen,
        "Status": np.array(status, dtype=str),
    })
    selected = {"gamma": candidate[chosen][0], "phi_min": candidate[chosen][1],
                "BestEpoch": int(best_epoch[chosen]), "ValidLoss": float(valid_loss[chosen])}

    scoring = {**fit.dataset, "Xtest": data["Xdata"]}
    all_risk, all_effect = risk_predict(scoring, fit.parameter, fit.model_param, return_effect=True)
    num_protein = data["Xdata"].shape[0]
    uppi = fit.model_param[:num_protein]
    bset = fit.model_param[num_protein:].reshape((num_protein, len(target)), order="F")

    split_column, metric_rows, test_metrics = [], [], None
    for split in SPLITS:
        member = data["split"] == split
        metrics = evaluate_risk(all_risk[member], data["Ydata"][member], cutoff=options["cutoff"], target=target)
        split_column += [split] * metrics.height
        metric_rows.append(metrics)
        if split == "test":
            test_metrics = metrics
    metric_table = Table({
        "Split": np.array(split_column, dtype=str),
        "Target": np.concatenate([metrics["Target"] for metrics in metric_rows]),
        **{name: np.concatenate([metrics[name] for metrics in metric_rows]) for name in METRICS},
    })
    parameters = Table({
        "Protein": data["protein"],
        "phi": uppi,
        **{f"theta_{name}": bset[:, k] for k, name in enumerate(target)},
        "IndependentMean": data["Xdata"].mean(axis=1),
        "SynergeticMean": all_effect.mean(axis=1),
    })
    predictions = Table({
        "Participant": data["participant"],
        "Split": data["split"],
        **{f"Y_{name}": data["Ydata"][:, k].astype(np.int64) for k, name in enumerate(target)},
        **{f"Risk_{name}": all_risk[:, k] for k, name in enumerate(target)},
    })
    history = Table({
        "Epoch": np.arange(1, fit.history["LossTrain"].shape[0] + 1),
        **{f"LossTrain_{name}": fit.history["LossTrain"][:, k] for k, name in enumerate(target)},
        **{f"LossValid_{name}": fit.history["LossValid"][:, k] for k, name in enumerate(target)},
    })

    result = {
        "Selection": selection,
        "Selected": selected,
        "ModelParam": fit.model_param,
        "TestMetrics": test_metrics,
        "Metrics": metric_table,
        "Parameters": parameters,
        "Predictions": predictions,
        "History": history,
    }
    if options["cv_repeat"] > 0:
        train = {"epoch": options["epoch"], "rate": options["rate"], "gamma": selected["gamma"],
                 "phi_min": selected["phi_min"], "gradient": options["gradient"]}
        result["CV"] = cross_validate(data, repeat=options["cv_repeat"], fold=options["cv_fold"],
                                      seed=options["seed"], train=train, cutoff=options["cutoff"],
                                      verbose=options["verbose"])
    result["Options"] = options
    result["Source"] = data["source"]
    result["ElapsedSeconds"] = time.perf_counter() - started

    if options["verbose"]:
        print(f"Selected gamma {_number(selected['gamma'])} and phi_min {_number(selected['phi_min'])} "
              f"(best epoch {selected['BestEpoch']}, validation loss {selected['ValidLoss']:.6f}).")
        print(test_metrics)
    if options["output"] is not None:
        write_outputs(result, options["output"])
    return result


def write_outputs(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("Selection", "Metrics", "Parameters", "Predictions", "History"):
        result[name].to_csv(folder / f"{name.lower()}.csv")
    options = result["Options"]
    summary = {
        "Source": result["Source"],
        "Selected": {key: _json_number(value) for key, value in result["Selected"].items()},
        "TestMetrics": result["TestMetrics"].to_records(),
        "Epoch": options["epoch"],
        "Rate": options["rate"],
        "Gradient": options["gradient"],
        "Seed": options["seed"],
        "Cutoff": options["cutoff"],
        "PythonVersion": platform.python_version(),
        "NumPyVersion": np.__version__,
        "SciPyVersion": scipy.__version__,
        "ElapsedSeconds": result["ElapsedSeconds"],
    }
    if "CV" in result:
        result["CV"]["FoldMetrics"].to_csv(folder / "cv_fold_metrics.csv")
        result["CV"]["Summary"].to_csv(folder / "cv_summary.csv")
        summary["CVRepeat"] = options["cv_repeat"]
        summary["CVFold"] = options["cv_fold"]
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return folder


def _number(value, spec="g"):
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return format(value, spec)


def _json_number(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value
