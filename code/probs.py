#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from itertools import chain
from typing import Counter
from collections import Counter
import numpy as np

import time
import tqdm
from SGD_convergent import ConvergentSGD

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Set[Wordtype]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process_the_token(token)
    # Whenever the `for` loop needs another token, read_tokens picks up where it
    # left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.

def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))

def num_tokens_general(file: Path) -> int:
    """Give the number of tokens in file, including EOS. If `file` does not specify
    a directory, then this function behaves identically to `num_tokens`. If `file`
    specifies a directory, then this function returns the total number of tokens in
    all non-directory files in this directory (subdirectory recursion can be turned on or off)."""
    if not(file.is_dir()):
        return num_tokens(file)
    # Use `file.iterdir()` in place of `file.rglob("*")` to avoid subdirectory recursion
    return sum([num_tokens(stuff) for stuff in file.rglob("*") if not(stuff.is_dir())])

def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.

def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

def read_trigrams_general(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file. If `file` does not specify a directory,
    then this function behaves identically to `read_trigrams`. If `file` specifies
    a directory, then this function returns an iterator over all trigrams that occur
    in any non-directory file in this directory (subdirectory recursion can be turned on or off)."""
    if not(file.is_dir()):
        return read_trigrams(file, vocab)
    # Use `file.iterdir()` in place of `file.rglob("*")` to avoid subdirectory recursion
    return chain(*[read_trigrams(stuff, vocab) for stuff in file.rglob("*") if not(stuff.is_dir())])

def sample(model_path: Path, num_sentences = 10, max_depth: int = 100) -> list:
    """sample given number of list with optionally a max depth by a chosen model."""
    sentences = []
    model = LanguageModel.load(model_path)
    vocab = model.vocab
    while len(sentences) < num_sentences:
        sentence = ""
        x, y = 'BOS', 'BOS'
        for j in range(max_depth):
            vocab_list = list(vocab)
            vocab_prob = [model.prob(x, y, z) for z in vocab_list]
            word = np.random.choice(vocab_list, p=vocab_prob)
            if word == 'EOS':
                break
            x, y = y, word
            sentence = sentence + ' ' + word
        if word != 'EOS':
            sentences.append(sentence[1:] + ' ...')
        else:
            sentences.append(sentence[1:]) # get rid of the beginning space 
    return sentences

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    return vocab


def read_repeat_n(n: int, path: Path, vocab: Vocab):
    """Iterator over the trigrams in file. If `file` does not specify a directory,
    then this function behaves identically to `read_trigrams`. If `file` specifies
    a directory, then this function returns an iterator over all trigrams that occur
    in any non-directory file in this directory (subdirectory recursion can be turned on or off)."""
    
    list_trigrams = list(read_trigrams_general(path, vocab))
    result = [0] * len(list_trigrams)
    for i in range(len(list_trigrams)):
        for j in range(max(0, i - n), i):
            if list_trigrams[j][-1] == list_trigrams[i][-1]:
                result[i] += 1
    return torch.FloatTensor(result)

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.
        self.hyperparams_dict = {}

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    #@property
    #def hyperparams(self):
    #    return self._hyperparams

    #@hyperparams.setter
    def hyperparams(self, key, value):
        self.hyperparams_dict[key] = value

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override LanguageModel.prob)"
        )

    # A default implementation of a new log-prob function; our log-linear models
    # later on will prefer to work in log-space and will thus override this method
    # with a more numerically accurate alternative.
    def log_prob_float(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return math.log(self.prob(x, y, z))

    @classmethod
    def load(cls, source: Path) -> "LanguageModel":
        import pickle  # for loading/saving Python objects
        log.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            log.info(f"Loaded model from {source}")
            return pickle.load(f)

    def save(self, destination: Path) -> None:
        import pickle
        log.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {destination}")

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class UniformLanguageModel(LanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(LanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)

        if lambda_ < 0:
            raise ValueError("negative lambda argument of {lambda_}")
        self.lambda_ = lambda_
        self.hyperparams("lambda", self.lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.

    # A new custom-made function to numerically approximate the lambda value that minimizes
    # average cross-entropy per token for two trained models that are smoothed with the same lambda.
    # It is okay for this method to accept any types of LanguageModel, since the only local variables
    # of these models that we use are the event/context counts and the vocabulary.
    @classmethod
    def learn_lambda(cls, model1: LanguageModel, model2: LanguageModel, dev1: Path, dev2: Path):
        # Feel free to customize these parameters
        maxit = 50000
        # Cool choices of (initial_lambda, learning_rate) : (2.0, 0.01), (0.005, 0.001)
        initial_lambda = 0.5
        learning_rate = 0.001
        tol = 0.00000001
        print_level = 1000

        lamb = nn.Parameter(torch.tensor(initial_lambda), requires_grad=True)
        V1 = len(model1.vocab)
        V2 = len(model2.vocab)
        N1 = num_tokens_general(dev1)
        N2 = num_tokens_general(dev2)
        den = (N1 + N2)*torch.log(torch.tensor(2.0)).item()
        optimizer = ConvergentSGD([lamb], learning_rate, 0.1)
        prev_lamb = -1.0
        counts1 = [(model1.event_count[x, y, z], model1.context_count[x, y]) for (x, y, z) in read_trigrams_general(dev1, model1.vocab)]
        counts2 = [(model2.event_count[x, y, z], model2.context_count[x, y]) for (x, y, z) in read_trigrams_general(dev2, model2.vocab)]
        event_counts1 = torch.FloatTensor([float(c[0]) for c in counts1])
        context_counts1 = torch.FloatTensor([float(c[1]) for c in counts1])
        event_counts2 = torch.FloatTensor([float(c[0]) for c in counts2])
        context_counts2 = torch.FloatTensor([float(c[1]) for c in counts2])
        onesies1 = torch.ones(len(counts1), dtype=torch.float)
        onesies2 = torch.ones(len(counts2), dtype=torch.float)
        it = 0
        while ((it < maxit) and (abs(lamb.item() - prev_lamb) >= tol)):
            optimizer.zero_grad()
            prev_lamb = lamb.item()
            xent = torch.sum(torch.log(context_counts1 + V1*lamb*onesies1) - torch.log(event_counts1 + lamb*onesies1))
            xent += torch.sum(torch.log(context_counts2 + V2*lamb*onesies2) - torch.log(event_counts2 + lamb*onesies2))
            xent *= 1/den
            if (print_level >= 1):
                if ((it % print_level) == 0):
                    # TODO: Use log.info instead of stdout here?
                    print(prev_lamb, xent.item())
            xent.backward()
            optimizer.step()
            if (lamb.item() < 0.0):
                lamb.data = torch.tensor(0.0)
            it += 1
        if (it >= maxit):
            # TODO: Use log.info instead of stdout here?
            print("Failed to converge in " + str(it) + " iterations")
        final_xent = torch.sum(torch.log(context_counts1 + V1*lamb*onesies1) - torch.log(event_counts1 + lamb*onesies1)).item()
        final_xent += torch.sum(torch.log(context_counts2 + V2*lamb*onesies2) - torch.log(event_counts2 + lamb*onesies2)).item()
        final_xent *= 1/den
        return lamb.item(), final_xent

    # A new custom-made function to construct an add-lambda smoothed LM from an unsmoothed LM.
    # If the optional argument `lambda_` is not specified, then it defaults to 0, preserving the
    # behavior of the unsmoothed LM.
    @classmethod
    def from_unsmoothed_lm(cls, lm: LanguageModel, lambda_: float = 0.0) -> AddLambdaLanguageModel:
        # If the input language model is already a smoothed LM, then
        # all we need to do is change its existing lambda value.
        if (isinstance(lm, AddLambdaLanguageModel)):
            lm.lambda_ = lambda_
            lm.hyperparams("lambda", lambda_)
            return lm
        smoothed_lm = AddLambdaLanguageModel(lm.vocab, lambda_)
        smoothed_lm.event_count = lm.event_count
        smoothed_lm.context_count = lm.context_count
        return smoothed_lm


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!
        p_z = (self.event_count[(z,)] + self.lambda_)/(self.context_count[()] + self.lambda_*self.vocab_size)
        p_zy = (self.event_count[(y, z)] + self.lambda_*self.vocab_size*p_z)/(self.context_count[(y,)] + self.lambda_*self.vocab_size)
        return (self.event_count[(x, y, z)] + self.lambda_*self.vocab_size*p_zy)/(self.context_count[(x, y)] + self.lambda_*self.vocab_size)


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float) -> None:
        super().__init__(vocab)
        if l2 < 0:
            log.error(f"l2 regularization strength value was {l2}")
            raise ValueError("You must include a non-negative regularization value")
        self.l2: float = l2
        self.hyperparams("L2_weight", self.l2)

        self.dim = 99999999999  # TODO: SET THIS TO THE DIMENSIONALITY OF THE VECTORS
        with open(lexicon_file) as f:
            first_line = next(f).replace("\n", "")  # Peel off the special first line.
            [num_words, self.dim] = [int(s) for s in first_line.split(" ")]
            self.embeddings = torch.zeros([num_words, self.dim], dtype=torch.float)
            self.word_indices = {}
            i = 0
            for line in f:  # All of the other lines are regular.
                tokens = line.split("\t")
                if ((tokens[0] == OOL) or (tokens[0] in self.vocab)):
                    self.word_indices[tokens[0]] = i
                    self.embeddings[i,:] = torch.FloatTensor([float(tokens[j]) for j in range(1, self.dim + 1)])
                    i += 1
        # The line below is probably only necessary if 'OOV' appears in the lexicon (which should not happen).
        #self.word_indices[OOV] = self.word_indices[OOL] # any 'OOV' word in our vocab should correspond to 'OOL' in our lexicon

        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

        # If we are able to use a GPU, then let's use it!
        # Unfortunately, PyTorch does not support the ugrad network GPUs, so we have to turn this feature off for the time being :(
        self.on_gpu = False #torch.cuda.is_available()
        if self.on_gpu:
            self = self.cuda()


    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # This returns an ordinary float probability, using the
        # .item() method that extracts a number out of a Tensor.
        p = self.log_prob(x, y, z).exp().item()
        assert isinstance(p, float)  # checks that we'll adhere to the return type annotation, which is inherited from superclass
        return p

    # Helper function to retrieve the index of a word in our matrix of embeddings
    def _word_index(self, x: Wordtype) -> int:
        return self.word_indices[(x if (x in self.word_indices) else OOL)]

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> torch.Tensor:
        """Return log p(z | xy) according to this language model."""
        # use vectorization for the normalization constant Z for speed

        # We take the transpose of the matrix multiplication expression in equation (7) of the reading materials, so our `self.X` and `self.Y`
        # actually correspond to the transposes of the matrices X and Y defined in the reading materials. However, since X and Y are square
        # matrices and are only being used internally for this log-prob computation, there is no problem with working with X^T and Y^T.
        stuff = torch.mm(self.embeddings, torch.mm(self.X, torch.unsqueeze(self.embeddings[self._word_index(x),:], 1)) + \
                          torch.mm(self.Y, torch.unsqueeze(self.embeddings[self._word_index(y),:], 1)))
        return stuff[self._word_index(z)] - torch.logsumexp(stuff, 0) #torch.sum(stuff.exp()).log()

    def cross_entropy(self, val_file: Path) -> float:
        return -sum([self.log_prob(x, y, z) for (x, y, z) in read_trigrams(val_file, self.vocab)]) / num_tokens(val_file)

    # Same as log_prob but without wrapping the numerical answer in a torch Tensor
    def log_prob_float(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return self.log_prob(x, y, z).item()

    # Faster version of `log_prob` for when we have precomputed the token indices
    def fast_log_prob(self, x_index: int, y_index: int, z_index: int):
        stuff = torch.mm(self.embeddings, torch.mm(self.X, torch.unsqueeze(self.embeddings[x_index,:], 1)) + \
                torch.mm(self.Y, torch.unsqueeze(self.embeddings[y_index,:], 1)))
        if self.on_gpu:
            stuff = stuff.cuda()
        return stuff[z_index] - torch.logsumexp(stuff, 0)

    # This function, which directly loads the index triple for each trigram, is used in this class as well as its subclass.
    def read_trigram_indices(self, p: Path):
        return [(self._word_index(x), self._word_index(y), self._word_index(z)) for (x, y, z) in read_trigrams_general(p, self.vocab)]

    def train(self, file: Path, max_epochs: int = 10, learning_rate: float = 0.001):    # type: ignore
        
        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type). 
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        
        # Optimization hyperparameters.
        gamma0 = learning_rate  # initial learning rate - recommended to use 0.1 for gen-spam task and 0.01 for english-spanish task
        self.hyperparams("initial_learning_rate", gamma0)

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        optimizer = optim.SGD(self.parameters(), lr=gamma0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        # log.info("Start optimizing on {N} training tokens...")
        print("Training on corpus " + file.parts[-1])

        #####################
        # TODO: Implement your SGD here by taking gradient steps on a sequence
        # of training examples.  Here's how to use PyTorch to make it easy:
        #
        # To get the training examples, you can use the `read_trigrams` function
        # we provided, which will iterate over all N trigrams in the training
        # corpus.
        # trigrams_list = list(read_trigrams(file, self.vocab))
        indices_list = self.read_trigram_indices(file)
        total_epochs = max_epochs # previously this was always set to 10
        self.hyperparams("max_epochs", total_epochs)
        verbose = True
        single_example_loss = True
        tqdm_threshold = 60
        epoch_duration = 69
        #
        # For each successive training example i, compute the stochastic
        # objective F_i(θ).  This is called the "forward" computation. Don't
        # forget to include the regularization term.
        for epoch in range(total_epochs):
            if not(single_example_loss):
                optimizer.zero_grad()
                avg_loss = (self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y))) - \
                        sum([self.fast_log_prob(ix, iy, iz) for (ix, iy, iz) in indices_list]))/N
                #avg_loss = (self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y))) - \
                #        sum([self.log_prob(x, y, z) for (x, y, z) in trigrams_list]))/N
                avg_loss.backward()
                optimizer.step()
                if verbose:
                    fval = -avg_loss.cpu().item()
                    print(f"Epoch {epoch+1}: F = {fval:g}")
            else:
                epoch_start_time = time.time()
                for (ix, iy, iz) in (indices_list if (epoch_duration < tqdm_threshold) else tqdm.tqdm(indices_list, total=N)):
                    optimizer.zero_grad()
                    logprob = self.fast_log_prob(ix, iy, iz)
                    #if self.on_gpu:
                    #    logprob = logprob.cuda()
                    loss = self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y)))/N - logprob
                    #loss = self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y)))/N - self.log_prob(x, y, z)
                    #
                    # To get the gradient of this objective (∇F_i(θ)), call the `backward`
                    # method on the number you computed at the previous step.  This invokes
                    # back-propagation to get the gradient of this number with respect to
                    # the parameters θ.  This should be easier than implementing the
                    # gradient method from the handout.
                    loss.backward()
                    #
                    # Finally, update the parameters in the direction of the gradient, as
                    # shown in Algorithm 1 in the reading handout.  You can do this `+=`
                    # yourself, or you can call the `step` method of the `optimizer` object
                    # we created above.  See the reading handout for more details on this.
                    optimizer.step()
                if verbose:
                    with torch.no_grad():
                        #reg_term = self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y))).item() / N
                        logsum = sum([self.fast_log_prob(ix, iy, iz) for (ix, iy, iz) in indices_list])
                        #if self.on_gpu:
                        #    logsum = logsum.cuda()
                        avg_loss = (self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y))) - logsum)/N
                        #avg_loss = (self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y))) - \
                        #        sum([self.log_prob(x, y, z) for (x, y, z) in trigrams_list]))/N
                        fval = -avg_loss.cpu().item()
                        print(f"Epoch {epoch+1}: F = {fval:g}") #, L2 penalty = {reg_term:g}")
                epoch_duration = time.time() - epoch_start_time
        #
        # For the EmbeddingLogLinearLanguageModel, you should run SGD
        # optimization for 10 epochs and then stop.  You might want to print
        # progress dots using the `show_progress` method defined above.  Even
        # better, you could show a graphical progress bar using the tqdm module --
        # simply iterate over
        #     tqdm.tqdm(read_trigrams(file), total=10*N)
        # instead of iterating over
        #     read_trigrams(file)
        #####################
        self.hyperparams("L2_weight", self.l2)
        print(f"Finished training on {N} tokens")
        #log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME!
    
    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #

    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, train_batch_size: int, val_batch_size: int, repeat_str_len: int = 10) -> None:
        super().__init__(vocab, lexicon_file, l2)
        if train_batch_size <= 0:
            log.error(f"Training batch size value was {train_batch_size}")
            raise ValueError("You must include a positive training batch size")
        if val_batch_size <= 0:
            log.error(f"Validation batch size value was {val_batch_size}")
            raise ValueError("You must include a positive validation batch size")
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.repeat_str_len = repeat_str_len
        # weight assigned to each word on its repetition feature
        self.repetition_theta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.unigram_weights = nn.Parameter(torch.zeros((self.dim, 1)), requires_grad=True)
        self.trigram_weights = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

    # For this improved model, we need to look at the entire token sequence at once when computing
    # its log probability because the repetition feature depends on more than one trigram.
    def total_log_prob(self, val_file: Path) -> float:
        """Return log p(z | xy) according to this language model."""
        val_repetition = read_repeat_n(self.repeat_str_len, val_file, self.vocab)
        log_prob = 0
        for (i, (x, y, z)) in enumerate(read_trigrams(val_file, self.vocab)):
            x_col = torch.unsqueeze(self.embeddings[self._word_index(x),:], 1)
            y_col = torch.unsqueeze(self.embeddings[self._word_index(y),:], 1)
            # Features from the original log-linear model
            stuff = torch.mm(self.embeddings, torch.mm(self.X, x_col) + torch.mm(self.Y, y_col))
            # Word repetition feature
            stuff += self.repetition_theta * val_repetition[i]
            # Trigram features
            W = torch.mm(x_col, torch.transpose(y_col, 0, 1)) * self.trigram_weights
            stuff += torch.sum(torch.mm(self.embeddings, W) * self.embeddings, 1)
            # Unigram features
            stuff += torch.mm(self.embeddings, self.unigram_weights)
            log_prob += (stuff[self._word_index(z)] - torch.logsumexp(stuff, 0)).item()
        return log_prob

    def cross_entropy(self, val_file: Path) -> float:
        return -self.total_log_prob(val_file) / num_tokens(val_file)

    def fast_total_log_prob(self, trigram_index_list: list, rep_list: list, batch_index_list: list) -> torch.Tensor:
        """Return log p(z | xy) according to this language model."""
        log_prob = 0
        for i in batch_index_list:
            x_col = torch.unsqueeze(self.embeddings[trigram_index_list[i][0],:], 1)
            y_col = torch.unsqueeze(self.embeddings[trigram_index_list[i][1],:], 1)
            # Features from the original log-linear model
            stuff = torch.mm(self.embeddings, torch.mm(self.X, x_col) + torch.mm(self.Y, y_col))
            # Word repetition feature
            stuff += self.repetition_theta * rep_list[i]
            # Trigram features
            W = torch.mm(x_col, torch.transpose(y_col, 0, 1)) * self.trigram_weights
            stuff += torch.sum(torch.mm(self.embeddings, W) * self.embeddings, 1)
            # Unigram features
            stuff += torch.mm(self.embeddings, self.unigram_weights)
            log_prob += stuff[trigram_index_list[i][2]] - torch.logsumexp(stuff, 0)
        return log_prob

    # L2 regularization term
    def regu(self, N: int) -> torch.Tensor:
        return self.l2*(torch.sum(torch.square(self.X)) + torch.sum(torch.square(self.Y)) + torch.sum(torch.square(self.repetition_theta)))/N

    # Cross-Entropy Loss of a batch with L2 Regularization
    # Correction made on 10/06/21: Regularization term should be divided by N, the number of samples in the underlying dataset.
    # Therefore, the loss of a single mini-batch depends on a parameter that is not an intrinsic property of that mini-batch.
    def xent_loss(self, trigram_batch, N):
        return self.regu(N) - sum([self.log_prob(x, y, z) for (x, y, z) in zip(*trigram_batch)])/len(trigram_batch[0])

    # Version of `xent_loss` that uses word indices in place of the words themselves, so that we can call `fast_log_prob`
    def fast_xent_loss(self, trigram_indices_batch, N):
        return self.regu(N) - sum([self.fast_log_prob(ix, iy, iz) for (ix, iy, iz) in zip(*trigram_indices_batch)])/len(trigram_indices_batch[0])

    def train(self, train_file: Path, val_file: Path, max_epochs: int, patience: int = 10, learning_rate: float = 0.001, opt: str = "ConvergentSGD"):    # type: ignore
        N = num_tokens_general(train_file)
        M = num_tokens_general(val_file)

        # Sanity checks
        learning_rate = max(0.000001, learning_rate)
        max_epochs = max(1, max_epochs)
        patience = max(1, patience)
        # Optimization hyperparameters
        optimizer = None
        if (opt == "SGD"):
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            self.hyperparams("optimizer", "SGD")
        elif (opt == "Adam"):
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            self.hyperparams("optimizer", "Adam")
        else:
            optimizer = ConvergentSGD(self.parameters(), learning_rate, 2*self.l2/N)
            self.hyperparams("optimizer", "ConvergentSGD")
        # Initialize the parameter matrices via uniform distributions.
        nn.init.xavier_uniform_(self.X)   # type: ignore
        nn.init.xavier_uniform_(self.Y)   # type: ignore
        nn.init.xavier_uniform_(self.trigram_weights)   # type: ignore
        nn.init.xavier_uniform_(self.unigram_weights)   # type: ignore

        # initialize rep_list for the repetition number of each word
        rep_list = read_repeat_n(self.repeat_str_len, train_file, self.vocab)
        val_rep_list = read_repeat_n(self.repeat_str_len, val_file, self.vocab)

        # Mini-batch dataloaders
        train_trigrams = self.read_trigram_indices(train_file)
        train_dataloader = DataLoader(list(range(N)), \
                                      batch_size=min(N, self.train_batch_size), \
                                      sampler=SubsetRandomSampler(range(N)))
        val_trigrams = self.read_trigram_indices(val_file)
        val_dataloader = DataLoader(list(range(M)), \
                                    batch_size=min(M, self.val_batch_size), \
                                    sampler=SubsetRandomSampler(range(M)))
        log.info(f"Start optimizing on {N} training tokens...")
        
        tqdm_threshold = 60
        epoch_duration = 69 # any bogus constant greater than `tqdm_threshold` works here
        best_val_loss = 420.0 # any bogus constant works here
        impatience_counter = 0 # this *must* be initialized at zero

        # Save copies of learnable parameters, detached from the computational graph.
        bestX = torch.clone(self.X).detach()
        bestY = torch.clone(self.Y).detach()
        best_rep_theta = torch.clone(self.repetition_theta).detach()
        best_trigram_weights = torch.clone(self.trigram_weights).detach()
        best_unigram_weights = torch.clone(self.unigram_weights).detach()
        
        for epoch in range(max_epochs):
            log.info(f"Running epoch {epoch+1}/{max_epochs} ...")
            epoch_start = time.time()
            train_loss = 0
            for batch in (tqdm.tqdm(train_dataloader, total=len(train_dataloader)) if (epoch_duration > tqdm_threshold) else train_dataloader):
                optimizer.zero_grad()
                len_batch = len(batch)
                loss = self.regu(N) - self.fast_total_log_prob(train_trigrams, rep_list, batch) / len_batch
                train_loss += len_batch*loss.cpu().item()
                loss.backward()
                optimizer.step()
            log.info(f"Training loss: {train_loss/N:g}")
            val_loss = 0
            with torch.no_grad():
                for val_batch in (tqdm.tqdm(val_dataloader, total=len(val_dataloader)) if (epoch_duration > tqdm_threshold) else val_dataloader):
                    len_batch = len(val_batch)
                    val_loss += (len_batch*self.regu(M) - self.fast_total_log_prob(val_trigrams, val_rep_list, val_batch)).item()
            log.info(f"Validation loss: {val_loss/M:g}")
            epoch_duration = time.time() - epoch_start
            log.info(f"Epoch {epoch+1} finished in {epoch_duration:g} seconds.")
            # Early stopping criterion: validation loss does not improve from its current best value for too many epochs
            if (epoch == 0 or val_loss < best_val_loss):
                impatience_counter = 0
                bestX = torch.clone(self.X).detach()
                bestY = torch.clone(self.Y).detach()
                best_rep_theta = torch.clone(self.repetition_theta).detach()
                best_trigram_weights = torch.clone(self.trigram_weights).detach()
                best_unigram_weights = torch.clone(self.unigram_weights).detach()
                best_val_loss = val_loss
            else:
                impatience_counter += 1
                if (impatience_counter >= patience):
                    log.info(f"Validation loss has not decreased in {patience} epochs.")
                    log.info(f"Stopping early and restoring model weights from epoch {epoch+1-impatience_counter}.")
                    self.X = nn.Parameter(bestX, requires_grad=True)
                    self.Y = nn.Parameter(bestY, requires_grad=True)
                    self.repetition_theta = nn.Parameter(best_rep_theta, requires_grad=True)
                    self.trigram_weights = nn.Parameter(best_trigram_weights, requires_grad=True)
                    self.unigram_weights = nn.Parameter(best_unigram_weights, requires_grad=True)
                    break
        # If training was successful, then save the training hyperparameters so they can be recovered later.
        self.hyperparams("train_batch_size", self.train_batch_size)
        self.hyperparams("val_batch_size", self.val_batch_size)
        self.hyperparams("max_epochs", max_epochs)
        self.hyperparams("initial_learning_rate", learning_rate)
        self.hyperparams("L2_weight", self.l2)
        self.hyperparams("patience", patience)
        log.info("Done optimizing.")
