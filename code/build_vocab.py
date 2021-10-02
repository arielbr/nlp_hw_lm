#!/usr/bin/env python3
"""
Builds a vocabulary file from a list of training documents. Allows the specification of
a threshold t such that only words that appear at least t times gets included in
the vocabulary. A vocabulary file is saved as a text files where each line is a word.
"""
import argparse
import sys
from typing import Iterable
from collections import Counter
from pathlib import Path

from Probs import EOS, OOV, Wordtype, get_tokens


VOCAB_THRESHOLD = 3    # minimum number of occurrence for a word to be considered in-vocabulary

def parse_cmdline():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "documents",
        nargs="+",
        type=Path,
        help="a list of text documents from which to extract the vocabulary")
    parser.add_argument(
        "--save_file",
        default="tmp_vocab.txt",
        help="a file to save the vocabulary to"
    )
    parser.add_argument(
        "--vocab_threshold",
        default=VOCAB_THRESHOLD,
        type=int,
        help="the minimum number of times a word has to appear for it to be included in the vocabulary")

    args = parser.parse_args()
    return args


def build_vocab(*files: Iterable[Path], vocab_threshold: int = VOCAB_THRESHOLD, progress_freq: int = 5000) -> set:
    progress = 0
    word_counts: Counter[Wordtype] = Counter()  # count of each word
    for file in files:
        for token in get_tokens(file):
            word_counts[token] += 1
            if progress % progress_freq == 1: # print a dot every progress_freq words processed
                sys.stderr.write(".")
    sys.stderr.write("\n")  # done printing progress dots "...."

    vocab = set(w for w in word_counts if word_counts[w] >= vocab_threshold)
    vocab |= {  # Union equals
        OOV,
        EOS,
    }  # We make sure that EOS is in the vocab, even if get_tokens returns it too few times.
    # But BOS is not in the vocab: it is never a possible outcome, only a context.

    sys.stderr.write(f"Vocabulary size is {len(vocab)} types including OOV and EOS\n")
    return vocab


def save_vocab(vocab, save_file):
    with open(save_file, "wt") as fout:
        for word in vocab:
            print(word, file=fout)

def main():
    """
    A vocab file is just a list of words.

    Before using, change this script to be executable on unix systems via chmod +x build_vocab.py.
    Alternatively, use python3 build_vocab.py instead of ./build_vocab.py in the following example.

    Example usage:

        Build a vocab file out of the union of words in spam and gen

        ./build_vocab.py ../data/gen_spam/train/gen ../data/gen_spam/train/spam --save_file vocab-genspam.txt --vocab_thresold 3

        After which you should see the following saved vocab file:

        vocab-genspam.txt
    """
    args = parse_cmdline()
    vocab = build_vocab(*args.documents, vocab_threshold=args.vocab_threshold)
    save_vocab(vocab, args.save_file)

if __name__ == '__main__':
    main()