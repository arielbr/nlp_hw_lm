#!/usr/bin/env python3
"""
Computes which of two language models was more likely to have generated each given piece of text.
"""
import argparse
import logging
import math
import sys
from pathlib import Path

from probs import LanguageModel, num_tokens, read_trigrams

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
        "prior_1",
        type=float,
        help="prior probability of the first trained model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        prob = lm.prob(x, y, z)  # p(z | xy)
        log_prob += math.log(prob)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    # Test if the prior probability is invalid
    if (args.prior_1 <= 0.0 or args.prior_1 >= 1.0):
        log.error(f"Invalid prior probability {args.prior_1:g} (must be strictly between 0 and 1)")
        sys.exit(1)

    log.info("Testing...")
    lm_1 = LanguageModel.load(args.model_1)
    lm_2 = LanguageModel.load(args.model_2)
    corpus_name_1 = str(args.model_1).split(".")[0]
    corpus_name_2 = str(args.model_2).split(".")[0]

    # Test if the language models have different vocabularies
    if (len(lm_1.vocab) != len(lm_2.vocab)):
        log.error("Language models do not have the same vocabulary")
        sys.exit(1)
    if (lm_1.vocab != lm_2.vocab):
        log.error("Language models do not have the same vocabulary")
        sys.exit(1)
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    lm_1_count = 0
    for file in args.test_files:
        log_prob_1: float = file_log_prob(file, lm_1) + math.log(args.prior_1)
        log_prob_2: float = file_log_prob(file, lm_2) + math.log(1 - args.prior_1)
        better_lm = (corpus_name_1 if (log_prob_1 >= log_prob_2) else corpus_name_2)
        print(f"{better_lm}\t{file}")
        if (log_prob_1 >= log_prob_2):
            lm_1_count += 1
    num_files = len(args.test_files)
    lm_2_count = num_files - lm_1_count
    print(f"{lm_1_count} files were more probably {corpus_name_1} ({lm_1_count / num_files:.2%})")
    print(f"{lm_2_count} files were more probably {corpus_name_2} ({lm_2_count / num_files:.2%})")


if __name__ == "__main__":
    main()
