import argparse
import math
import sys

from .errors import PPIxGPNError
from .model import GRADIENTS
from .sample import run_sample


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ppixgpn",
        description="Fit PPIxGPN to a dataset in the sample.csv layout, choosing gamma and phi_min by validation loss.",
    )
    parser.add_argument("--file", help="dataset CSV (default: ../dataset/sample.csv next to this package)")
    parser.add_argument("--gamma", type=_real_list, default=(0.01, 0.001, 0.0001),
                        help="comma-separated L2 penalties (default: 0.01,0.001,0.0001)")
    parser.add_argument("--phi-min", type=_real_list, default=(0.01,),
                        help="comma-separated lower bounds for phi (default: 0.01); unrestricted mode: --phi-min=-inf")
    parser.add_argument("--epoch", type=int, default=3000, help="evaluated epochs per candidate (default: 3000)")
    parser.add_argument("--rate", type=float, default=0.001, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--gradient", choices=GRADIENTS, default="exact", help="phi gradient (default: exact)")
    parser.add_argument("--seed", type=int, default=1, help="initialization and cross-validation seed (default: 1)")
    parser.add_argument("--cutoff", type=float, default=0.5, help="risk cutoff for accuracy and F1 (default: 0.5)")
    parser.add_argument("--cv-repeat", type=int, default=0, help="repeated cross-validation runs (default: 0)")
    parser.add_argument("--cv-fold", type=int, default=5, help="cross-validation folds (default: 5)")
    parser.add_argument("--output", help="folder for CSV and JSON results")
    parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    arguments = parser.parse_args(argv)
    try:
        run_sample(file=arguments.file, gamma=arguments.gamma, phi_min=arguments.phi_min, epoch=arguments.epoch,
                   rate=arguments.rate, gradient=arguments.gradient, seed=arguments.seed, cutoff=arguments.cutoff,
                   cv_repeat=arguments.cv_repeat, cv_fold=arguments.cv_fold, output=arguments.output,
                   verbose=not arguments.quiet)
    except PPIxGPNError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _real_list(text):
    try:
        return tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from None
