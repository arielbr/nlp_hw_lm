#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in the given
files, according to a smoothed trigram model.  Also makes it possible to
train the model.  Several types of smoothing are supported.
"""
import argparse
import logging
from pathlib import Path

try:    
    import numpy as np  # your program would be so slow without numpy!
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise

from Probs import LanguageModel, num_tokens

TRAIN = "TRAIN"
TEST = "TEST"

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def get_model_filename(smoother: str, lexicon: Path, vocab: Path, train_file: Path) -> Path:
    return Path(f"{train_file.name}_{vocab.name}_{smoother}_{lexicon.name}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices={TRAIN, TEST}, help="execution mode")
    parser.add_argument(
        "vocab",
        type=Path,
        help="location of the vocabulary file",
    )
    parser.add_argument(
        "smoother",
        type=str,
        help=f"Possible values: {LanguageModel.SMOOTHERS}",
    )
    parser.add_argument(
        "lexicon",
        type=Path,
        help="location of the word embedding file",
    )
    parser.add_argument("train_file", type=Path, help="location of the training corpus")
    parser.add_argument("test_files", type=Path, nargs="*")

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

    args = parser.parse_args()

    # Sanity-check the configuration.
    if args.mode == "TRAIN" and args.test_files:
        parser.error("Shouldn't see test files when training.")
    elif args.mode == "TEST" and not args.test_files:
        parser.error("No test files specified.")

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model_path = get_model_filename(args.smoother, args.lexicon, args.vocab, args.train_file)

    if args.mode == TRAIN:
        log.info("Training...")
        lm = LanguageModel.make(args.smoother, args.lexicon, args.vocab)

        lm.train(args.train_file)
        lm.save(destination=model_path)

    elif args.mode == TEST:
        log.info("Testing...")
        lm = LanguageModel.load(model_path)
        # We use natural log for our internal computations and that's
        # the kind of log-probability that file_log_prob returns.
        # But we'd like to print a value in bits: so we convert
        # log base e to log base 2 at print time, by dividing by log(2).

        log.info("Printing per-file log-likelihoods.")
        total_log_prob = 0.0
        for test_file in args.test_files:
            log_prob = lm.file_log_prob(test_file) / np.log(2)  
            print(f"{log_prob:g}\t{test_file}")
            total_log_prob += log_prob

        total_tokens = sum(num_tokens(test_file) for test_file in args.test_files)
        print(f"Overall cross-entropy:\t{-total_log_prob / total_tokens:.5f} bits per token")

    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()

