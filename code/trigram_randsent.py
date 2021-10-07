#!/usr/bin/env python3
"""
Generate random sentences from user-specified models, number of sentences and max length.
"""
import argparse
import logging
import math
import sys
from pathlib import Path

from probs import LanguageModel, num_tokens, read_trigrams, sample

from typing import List

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        help="number of sentences to be generated",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=False,
        help="Maximum length of each sentence generated",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.max_length):
        sentences = sample(args.model, args.num_sentences, args.max_length)
    else:
        sentences = sample(args.model, args.num_sentences)
    print(sentences)


if __name__ == "__main__":
    main()
