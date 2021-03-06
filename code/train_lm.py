#!/usr/bin/env python3
"""
Trains a smoothed trigram model over a given vocabulary.
Depending on the smoother, you need to supply hyperparameters and additional files.
"""
import argparse
import logging
from pathlib import Path
import sys

from probs import read_vocab, UniformLanguageModel, AddLambdaLanguageModel, \
    BackoffAddLambdaLanguageModel, EmbeddingLogLinearLanguageModel, ImprovedLogLinearLanguageModel

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

UNIFORM   = "uniform"
ADDLAMBDA = "add_lambda"
BACKOFF   = "add_lambda_backoff"
LOGLINEAR = "log_linear"
IMPROVED  = "log_linear_improved"
SMOOTHERS = [UNIFORM, ADDLAMBDA, BACKOFF, LOGLINEAR, IMPROVED]


def get_model_filename(args: argparse.Namespace) -> Path:
    prefix = f"corpus={args.train_file.name}~vocab={args.vocab_file.name}~smoother={args.smoother}"
    if args.smoother in [ADDLAMBDA, BACKOFF]:
        return Path(f"{prefix}~lambda={args.lambda_}.model")
    elif args.smoother in [LOGLINEAR, IMPROVED]:
        return Path(f"{prefix}~lexicon={args.lexicon.name}~l2={args.l2_regularization}.model")
    else:   
        raise NotImplementedError("smoother {args.smoother}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument(
        "vocab_file",
        type=Path,
        help="Vocabulary file",
    )
    parser.add_argument(
        "smoother",
        type=str,
        help=f"Smoothing method",
        choices=SMOOTHERS
    )
    parser.add_argument(
        "train_file",
        type=Path,
        help="Training corpus (as a single file)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the model (if not specified, will construct a filename with `get_model_filename`)"
    )

    # for add lambda smoothers
    parser.add_argument(
        "--lambda",
        dest="lambda_",  # store in the lambda_ attribute because lambda is a Python keyword 
        type=float,
        default=0.0,
        help="Strength of smoothing for add_lambda and add_lambda_backoff smoothers (default 0)",
    )

    # # for log linear smoothers
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="File of word embeddings (needed for our log-linear models)",
    )
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0.0,
        help="Strength of L2 regularization in log-linear models (default 0)",
    )
    
    # for improved log-linear smoothers
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default 10)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Max number of epochs without val loss improvement before early stopping (default 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate (default 0.001)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Size of each training data batch (default 64)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Size of each validation data batch (default 64)",
    )
    parser.add_argument(
        "--val_file",
        type=Path,
        default=None,
        help="Validation corpus (as a single file) needed for improved log-linear models (default None)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="ConvergentSGD",
        help="Optimizer to use for improved log-linear model (default 'ConvergentSGD', other options are 'SGD' and 'Adam')",
    )

    # for verbosity of output
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_const",
        const=logging.WARNING
    )

    return parser.parse_args()


def check_args(args: argparse.Namespace):
    # Sanity-check the configuration.
    if args.smoother == LOGLINEAR:
        if args.lexicon is None:
            raise ValueError(f"--lexicon is required for {LOGLINEAR} smoother")
    if args.smoother == ADDLAMBDA:
        if args.lambda_ == 0.0:
            log.warning("You're training an add-0 (unsmoothed) model")

def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    check_args(args)

    if args.output is None:
        model_path = get_model_filename(args)
    else:
        model_path = args.output

    if args.smoother not in SMOOTHERS:
        raise ValueError(f"Expected args.smoother to be one of {SMOOTHERS}, but got {args.smoother} instead")
    # Now construct a language model, giving the appropriate arguments to the constructor.

    vocab = read_vocab(args.vocab_file)
    if args.smoother == UNIFORM:
        lm = UniformLanguageModel(vocab)
    elif args.smoother == ADDLAMBDA:
        lm = AddLambdaLanguageModel(vocab, args.lambda_)
    elif args.smoother == BACKOFF:
        lm = BackoffAddLambdaLanguageModel(vocab, args.lambda_)
    elif args.smoother == LOGLINEAR:
        if args.lexicon is None:
            log.error("{args.smoother} requires a lexicon")   # would be better to check this in argparse
            sys.exit(1)
        lm = EmbeddingLogLinearLanguageModel(vocab, args.lexicon, args.l2_regularization)
    elif args.smoother == IMPROVED:
        if args.lexicon is None:
            log.error("{args.smoother} requires a lexicon")   # would be better to check this in argparse
            sys.exit(1)
        # Modification here to account for additional args required for the improved log-linear model's constructor
        lm = ImprovedLogLinearLanguageModel(vocab, args.lexicon, args.l2_regularization, args.train_batch_size, args.val_batch_size)
    else:
        raise ValueError(f"Don't recognize smoother name {args.smoother}")

    # Modification here to account for different arguments for the improved log-linear model's `train` function
    if args.smoother == IMPROVED:
        if args.val_file is None:
            log.error("{args.smoother} requires a validation corpus")   # would be better to check this in argparse
            sys.exit(1)
        log.info("Training...")
        lm.train(args.train_file, args.val_file, args.max_epochs, patience=args.patience, learning_rate=args.learning_rate, opt=args.optimizer)
    elif args.smoother == LOGLINEAR:
        log.info("Training...")
        lm.train(args.train_file, max_epochs=args.max_epochs, learning_rate=args.learning_rate)
    else:
        log.info("Training...")
        lm.train(args.train_file)
    lm.save(destination=model_path)


if __name__ == "__main__":
    main()
