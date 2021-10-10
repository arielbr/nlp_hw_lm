#!/usr/bin/env python3
"""
Computes which of two language models was more likely to have generated each given piece of text.
"""
import argparse
import logging
import math
import sys
from pathlib import Path

from probs import LanguageModel, num_tokens, read_trigrams, EmbeddingLogLinearLanguageModel, ImprovedLogLinearLanguageModel

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
    parser.add_argument(
        "--eval",
        type=bool,
        default=False,
        help="Evaluate on test data",
    )
    parser.add_argument(
        "--model_1_test_dir",
        type=Path,
        default=None,
        help="Directory containing test files that \"belong to\" model 1",
    )
    parser.add_argument(
        "--model_2_test_dir",
        type=Path,
        default=None,
        help="Directory containing test files that \"belong to\" model 2",
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
    if isinstance(lm, ImprovedLogLinearLanguageModel):
        return lm.total_log_prob(file)
    for (x, y, z) in read_trigrams(file, lm.vocab):
        #try:
        #    # The below computation may fail due to a numerical underflow in the computation of lm.prob(x, y, z)
        #    # If our model has a log_prob function, then let's use it!
        log_prob += lm.log_prob_float(x, y, z)
        #except:
        #    # Try the standard approach if our model lacks a log_prob function.
        #    prob = lm.prob(x, y, z) # p(z | xy)
        #    log_prob += math.log(prob)
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

# An engineered version of the above function that avoids duplicate work and speeds up model evaluation.
# The optional arguments `log_probs_1` and `log_probs_2` should only be used when we have precomputed the log-likelihoods
# of every file in `dev_files` with respect to `model1` and `model2`, respectively.
def binary_classifier_accuracies(model1: LanguageModel, model2: LanguageModel, dev_files: List[Path], belongs_to_1: List[bool], prior_1_list: List[float], \
        log_probs_1=None, log_probs_2=None, return_log_probs=False):
    total = len(dev_files)
    if (total != len(belongs_to_1)):
        log.error("List of dev files and list of ground truths do not have equal length")
        sys.exit(1)
    if log_probs_1 is None:
        log_probs_1 = [file_log_prob(dev_files[i], model1) for i in range(total)]
    if log_probs_2 is None:
        log_probs_2 = [file_log_prob(dev_files[i], model2) for i in range(total)]
    string_accs = []
    numerical_accs = []
    for j in range(len(prior_1_list)):
        prior_1 = prior_1_list[j]
        if (prior_1 <= 0.0 or prior_1 >= 1.0):
            log.error(f"Invalid prior probability {prior_1:g} (must be strictly between 0 and 1)")
            sys.exit(1)
        log_prior_1 = math.log(prior_1)
        log_prior_2 = math.log(1 - prior_1)
        correct = 0
        for i in range(total):
            log_odds_1 = log_probs_1[i] + log_prior_1
            log_odds_2 = log_probs_2[i] + log_prior_2
            if ((log_odds_1 >= log_odds_2) == belongs_to_1[i]):
                correct += 1
        numerical_accs.append(correct / total)
        string_accs.append(str(correct) + "/" + str(total))
    if return_log_probs:
        return numerical_accs, string_accs, log_probs_1, log_probs_2
    return numerical_accs, string_accs



def group_files_by_fixed_length_bins(file_directory: List[Path], num_items_per_bin: int = 10):
    file_directory = sorted(file_directory, key=lambda file: int(file.parts[-1].split(".")[1]))
    # Kyle: changed the line above from int(str(file).split(".")[-2]),
    # so that it can work on the language ID files, which lack the ".txt" extension
    num_bins = (len(file_directory) - 1) // num_items_per_bin + 1 
    #last_bin_length = ((len(file_directory) - 1) % num_items_per_bin) + 1
    bins = []
    for i in range(num_bins - 1):
        temp = file_directory[i*num_items_per_bin : (i+1)*num_items_per_bin]
        bins.append(temp)
    # The last bin needs special treatment since it may have fewer elements.
    bins.append(file_directory[(num_bins - 1)*num_items_per_bin:])
    return bins


# The base file names in the subtree of `file_directory` should have the form "xx.length.fileID(.txt)".
# We want to extract the integer value in place of 'length' in this file name format.
# TODO: Write the "driver" code that actually runs this function as part of a larger routine.
def group_files_by_length(file_directory: List[Path], num_lengths_per_bin: int = 0, num_bins: int = 10):
    # Retrieve all files in the directory subtree
    file_list = []
    length_list = []
    max_file_length = 0

    for f in file_directory:
        # Just a safety check - we want to skip paths that describe directories
        if f.is_dir():
            continue
        # Extract the length of the file using its properly formatted name
        try:
            new_length = int(f.parts[-1].split(".")[1])
            # Kyle: changed the line above from int(f.name.split(".")[-2]),
            # so that it can work on the language ID files, which lack the ".txt" extension
            if (new_length > max_file_length):
                max_file_length = new_length
            length_list.append(new_length)
        except:
            log.error("Improperly formatted file name " + f.name)
            sys.exit(1)
        file_list.append(f)
    # Just a safety check for handling directories with no valid files
    if (max_file_length <= 0):
        return [], 0
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

def check_accuracy(list_test_files, model1, model2, prior_1):
    bins = group_files_by_fixed_length_bins(list_test_files)
    accuracy = []
    for b in bins:
        numerical_acc, _ = binary_classifier_accuracy(model1, model2, b, prior_1)
        accuracy.append(numerical_acc)
        print(b)
        print(accuracy)

# A new function to evaluate a pair of language models on a set of labeled test data
def evaluate_classifier(model1: LanguageModel, model2: LanguageModel, model1name: str, model2name: str, testdir1: Path, testdir2: Path, prior_1: float = 0.0):
    test_list_1 = [stuff for stuff in testdir1.rglob("*") if not(stuff.is_dir())]
    belongs_to_1_1 = [True]*len(test_list_1)
    test_list_2 = [stuff for stuff in testdir2.rglob("*") if not(stuff.is_dir())]
    belongs_to_1_2 = [False]*len(test_list_2)
    low_priors_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    mid_priors_list = [0.05*p for p in range(1, 20)]
    high_priors_list = [0.99, 0.999, 0.9999, 0.99999, 0.999999]
    prior_1_list = ([prior_1] if (prior_1 > 0.0 and prior_1 < 1.0) else low_priors_list + mid_priors_list + high_priors_list)
    test_1_acc, test_1_str, t1m1lps, t1m2lps = binary_classifier_accuracies(model1, model2, test_list_1, belongs_to_1_1, prior_1_list, return_log_probs=True)
    #print("Model 1 data recall: " + str(test_1_acc) + " (" + test_1_str + ")")
    test_2_acc, test_2_str, t2m1lps, t2m2lps = binary_classifier_accuracies(model1, model2, test_list_2, belongs_to_1_2, prior_1_list, return_log_probs=True)
    #print("Model 2 data recall: " + str(test_2_acc) + " (" + test_2_str + ")")
    total_acc, total_str = binary_classifier_accuracies(model1, model2, test_list_1 + test_list_2, belongs_to_1_1 + belongs_to_1_2, prior_1_list, \
            log_probs_1=(t1m1lps+t2m1lps), log_probs_2=(t1m2lps+t2m2lps))
    #print("Total accuracy: " + str(total_acc) + " (" + total_str + ")")
    out = "Prior\tModel 1 data recall\tModel 2 data recall\tTotal Accuracy\n"
    for j in range(len(prior_1_list)):
        prior_str = str(round(prior_1_list[j], 6))
        if (len(prior_str) >= 8):
            prior_str = prior_str[1:]
        out += prior_str + "\t" + str(round(test_1_acc[j], 3)) + " (" + test_1_str[j] + ")\t\t" + \
                str(round(test_2_acc[j], 3)) + " (" + test_2_str[j] + ")\t\t" + \
                str(round(total_acc[j], 3)) + " (" + total_str[j] + ")\n"
    jason = False
    if isinstance(model1, EmbeddingLogLinearLanguageModel):
        out += "\n"
        jason = True
        out += "Model 1 data cross-entropy (with respect to model 1): " + str(round(model1.cross_entropy(test_list_1), 3)) + "\n"
    if isinstance(model2, EmbeddingLogLinearLanguageModel):
        if not(jason):
            out += "\n"
        out += "Model 2 data cross-entropy (with respect to model 2): " + str(round(model2.cross_entropy(test_list_2), 3)) + "\n"
    print(out)
    out += "\nModel 1 (" + model1name + ") Parameters:\n"
    for param_id in model1.hyperparams_dict:
        out += str(param_id) + ": " + str(model1.hyperparams_dict[param_id]) + "\n"
    out += "\nModel 2 (" + model2name + ") Parameters:\n"
    for param_id in model2.hyperparams_dict:
        out += str(param_id) + ": " + str(model2.hyperparams_dict[param_id]) + "\n"
    out += "\n"
    filename = "EVAL_" + model1name.split(".")[0] + "_AND_" + model2name.split(".")[0] + ".txt"
    f = open(filename, "w") # use the 'append' tag "a" instead of the 'write' tag "w" to avoid overwriting an existing file with the same name
    f.write(out)
    f.close()
    print("Saved evaluation results to " + filename)


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    # Test if the prior probability is invalid
    if (args.ln_prior):
        if (args.prior_1 > 0.0):
            log.error(f"Invalid natural log of prior probability (must be nonpositive)")
            sys.exit(1)
    else:
        if not(args.eval):
            if (args.prior_1 <= 0.0 or args.prior_1 >= 1.0):
                log.error(f"Invalid prior probability {args.prior_1:g} (must be strictly between 0 and 1)")
                sys.exit(1)
    
    if (args.accuracy):
        check_accuracy(args.test_files, args.model1, args.model2, args.prior_1)
        sys.exit(0)

    log.info("Testing...")
    lm_1 = LanguageModel.load(args.model_1)
    lm_2 = LanguageModel.load(args.model_2)
    corpus_name_1 = str(args.model_1) # changed from `str(args.model_1).split(".")[0]` as of a recent change in HW instructions
    corpus_name_2 = str(args.model_2) # changed from `str(args.model_2).split(".")[0]` as of a recent change in HW instructions

    if args.eval:
        evaluate_classifier(lm_1, lm_2, str(args.model_1), str(args.model_2), args.model_1_test_dir, args.model_2_test_dir, prior_1=args.prior_1)
        sys.exit(0)

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
