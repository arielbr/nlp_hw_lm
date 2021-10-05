#!/usr/bin/env python3
"""
Optimizes a single smoothing constant lambda to minimize cross-entropy across
two smoothed trigram models. This file is essentially just a driver for the
function `AddLambdaLanguageModel.learn_lambda`.
"""
import argparse
import logging
from pathlib import Path

from probs import LanguageModel, AddLambdaLanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model_2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "dev_1",
        type=Path,
        help="path to the first model's dev folder or file",
    )
    parser.add_argument(
        "dev_2",
        type=Path,
        help="path to the second model's dev folder or file",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    lm1 = LanguageModel.load(args.model_1)
    lm2 = LanguageModel.load(args.model_2)
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    best_lambda, best_xent = AddLambdaLanguageModel.learn_lambda(lm1, lm2, args.dev_1, args.dev_2)
    print(f"Best lambda: {best_lambda:g}")
    print(f"Best cross-entropy: {best_xent:g} bits per token")


if __name__ == "__main__":
    main()
