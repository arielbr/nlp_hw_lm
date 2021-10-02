#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""
# JHU NLP HW2
# Name: ____________
# Email: _________@jhu.edu
# Term: Fall 2021

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# For type annotations, which enable you to check correctness of your code:
from typing import List, Optional

try:
    # PyTorch is your friend. Not *using* it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import torch as th
    import torch.nn as nn
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.
# Logging is in general a good practice to check the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# - It prints to standard error (stderr), not standard output (stdout) by
#   default. This means it won't interfere with the real output of your
#   program. 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
#
# In `parse_args`, we provided two command line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'. 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

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
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words(bagpipe)
    """

    def __init__(self, vocab_size, dimension, vocab, embedding) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        self.vocab_size = vocab_size
        self.dimension = dimension
        self.vocab = vocab
        self.embedding = embedding

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        first_line = None
        embedding = [] # will store in a list and convert to Tensor later
        vocab = []

        cur_index = 0
        with open(file) as f:
            first_line = next(f)  # Peel off the special first line.
            for line in f:  # All of the other lines are regular.
                splits = line.strip().split("\t")
                vocab += [splits[0]]
                cur_index += 1
                embedding.append([float(i) for i in splits[1:]])
        dim = first_line.split(" ")
        vocab_size, dimension = int(dim[0]), int(dim[1])
        word_vocab = Integerizer(vocab)
        lexicon = Lexicon(vocab_size, dimension, word_vocab, th.tensor(embedding))
        return lexicon

    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")
        if (minus is None) and (plus is None):
            minus, plus = word, word

        word_vector =self.embedding[self.vocab.index(word)] \
            - self.embedding[self.vocab.index(minus)] \
            + self.embedding[self.vocab.index(plus)]
        similarity = th.div(th.mv(self.embedding, word_vector), (th.linalg.norm(self.embedding, dim=1) * th.linalg.norm(word_vector)))
        # sort from largest to smallest, return cos prob and indices
        sorted_sim, indices = th.sort(similarity)
        indices = th.flip(indices, [0])
        words = []
        # have to use a simple loop here because integerizer does not support tensor input
        for i in indices.tolist():
             words += [self.vocab[i]]
        # pop words: check first 13 (10+potentially 3 repeated)
        idx = 0
        for count in range(0,13):
            if words[idx] == word or words[idx] == minus or words[idx] == plus:
                words = words[:idx] + words[idx+1:]
            else:
                idx += 1
        return words

def format_for_printing(word_list: List[str]) -> str:
    return ' '.join(word_list[:10])

def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(format_for_printing(similar_words))


if __name__ == "__main__":
    main()
