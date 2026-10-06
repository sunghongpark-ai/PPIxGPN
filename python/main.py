import sys

sys.dont_write_bytecode = True

from model.cli import main as cli
from model.sample import run_sample


def main(**options):
    return run_sample(**options)


if __name__ == "__main__":
    raise SystemExit(cli())
