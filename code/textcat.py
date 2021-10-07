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

from typing import List

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
    parser.add_argument(
        "--ln_prior",
        type=bool,
        default=False,
        help="Set to True to interpret `prior_1` as the natural logarithm of the prior probability",
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        type=bool,
        default=False,
        help="Check accuracy of .txt length intervals of 50.",
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
    """
    The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        prob = lm.prob(x, y, z)  # p(z | xy)
        log_prob += math.log(prob)
    return log_prob


# A new function for computing the accuracy of an LM pair that acts as a binary classifier.
# TODO: Write the "driver" code that actually runs this function as part of a larger routine.
def binary_classifier_accuracy(model1: LanguageModel, model2: LanguageModel, dev_files: List[Path], belongs_to_1: List[bool], prior_1: float):
    if (prior_1 <= 0.0 or prior_1 >= 1.0):
        log.error(f"Invalid prior probability {prior_1:g} (must be strictly between 0 and 1)")
        sys.exit(1)
    total = len(dev_files)
    if (total != len(belongs_to_1)):
        log.error("List of dev files and list of ground truths do not have equal length")
        sys.exit(1)
    correct = 0
    log_prior_1 = math.log(prior_1)
    log_prior_2 = math.log(1 - prior_1)
    for i in range(total):
        log_prob_1: float = file_log_prob(dev_files[i], model1) + log_prior_1
        log_prob_2: float = file_log_prob(dev_files[i], model2) + log_prior_2
        if ((log_prob_1 >= log_prob_2) == belongs_to_1[i]):
            correct += 1
    numerical_acc = correct / total
    string_form = str(correct) + "/" + str(total)
    return numerical_acc, string_form


# The base file names in the subtree of `file_directory` should have the form "xx.length.fileID".
# We want to extract the integer value in place of 'length' in this file name format.
# TODO: Write the "driver" code that actually runs this function as part of a larger routine.
def group_files_by_length(file_directory: Path, num_lengths_per_bin: int = 0, num_bins: int = 10):
    # Retrieve all files in the directory subtree
    file_list = []
    length_list = []
    max_file_length = 0
    for f in file_directory.rglob("*"):
        # Just a safety check - we want to skip paths that describe directories
        if f.is_dir():
            continue
        # Extract the length of the file using its properly formatted name
        try:
            new_length = int(f.name.split(".")[-2])
            if (new_length > max_file_length):
                max_file_length = new_length
            length_list.append(new_length)
        except:
            log.error("Improperly formatted file name " + f.name)
            sys.exit(1)
        file_list.append(f)
    # Just a safety check for handling directories with no valid files
    if (max_file_length <= 0):
        return []
    # Calculate the "histogramming" parameters
    if (num_lengths_per_bin <= 0):
        if (num_bins <= 0):
            num_bins = 10
        num_lengths_per_bin = 1 + ((max_file_length - 1) // num_bins)
    else:
        # If both optional arguments are specified, then `num_lengths_per_bin` takes precedence
        num_bins = 1 + ((max_file_length - 1) // num_lengths_per_bin)
    # Group the files
    bins = []
    for _ in range(num_bins):
        bins.append([])
    for i in range(len(file_list)):
        bins[(length_list[i] - 1) // num_lengths_per_bin].append(file_list[i])
    return bins, num_lengths_per_bin


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    # Test if the prior probability is invalid
    if (args.ln_prior):
        if (args.prior_1 > 0.0):
            log.error(f"Invalid natural log of prior probability (must be nonpositive)")
            sys.exit(1)
    else:
        if (args.prior_1 <= 0.0 or args.prior_1 >= 1.0):
            log.error(f"Invalid prior probability {args.prior_1:g} (must be strictly between 0 and 1)")
            sys.exit(1)
    
    if (args.accuracy):
        print("accuracy")
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

    lm_1_count = 0
    log_prior_1 = 0
    log_prior_2 = 0
    if args.ln_prior:
        log_prior_1 = args.prior_1
        log_prior_2 = math.log(1 - math.exp(args.prior_1))
    else:
        log_prior_1 = math.log(args.prior_1)
        log_prior_2 = math.log(1 - args.prior_1)
    for file in args.test_files:
        log_prob_1: float = file_log_prob(file, lm_1) + log_prior_1
        log_prob_2: float = file_log_prob(file, lm_2) + log_prior_2
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
