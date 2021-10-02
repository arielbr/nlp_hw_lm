A version of this Python port for an earlier version of this
assignment was kindly provided by Eric Perlman, a previous student in
the NLP class.  Thanks to Prof. Jason Baldridge at U. of Texas for
updating the code and these instructions when he borrowed the
assignment.  They were subsequently modified by Xuchen Yao, Mozhi
Zhang and Chu-Cheng Lin for more recent versions of the assignment.
The current form was prepared by Arya McCarthy and Jason Eisner.

----------------------------------------------------------------------

Hooray, let's kick it with some smoothed language models.

## Downloading the Assignment Materials

We assume that you've made a local copy of <http://www.cs.jhu.edu/~jason/465/hw-lm/> 
(for example, by downloading and unpacking the zipfile there) and that you're currently 
in the `code/` subdirectory.

## Environments and Miniconda

You probably also want to install Miniconda, a minimal tool
for managing and reproducing environments. It does the hard
work of installing packages like NumPy that use faster,
vectorized math compared to the standard Python int and float
data types.

Miniconda (and its big sibling Anaconda) are all the rage in
NLP and deep learning. Install it following your platform-
specific instructions from here:

<https://conda.io/projects/conda/en/latest/user-guide/install/index.html>

One you've installed it, you can create an "environment" that
matches the one on the autograder, so you instantly know whether
your code will work there or not.

    conda env create -f environment.yml

Now that you've cloned our environment (made it available for your use)
from the .yml specification file, you can "activate" it.

    conda activate hw-lm

If this worked, then your prompt should be prefixed by the 
environment name, like this:

    (hw-lm) arya@ugradx:~/hw-lm/code$

This means that third-party packages like PyTorch are now
available for you to "import" in your Python scripts. You
are also, for sure, using the same Python version as we are.

----------

## QUESTION 1.
Type `./build_vocab.py --help` to see documentation.

Once you've familiarized yourself with the arguments, try
to run the script like this:

```
Usage:   ./build_vocab.py corpus1 corpus2 ... corpusN --save_path vocabfilename --vocab_thresold number
Example: ./build_vocab.py ../data/gen_spam/train/gen ../data/gen_spam/train/spam --save_file vocab-genspam.txt --vocab_thresold 3
```

A vocab file is just a text file with a list of words in it. Once you've built a vocab file, you can use it as an input to `./fileprob.py`.

You will need to build a new vocabulary file every time you train on a new corpus, but once built, you can keep reusing the same vocab file for training different models on the same corpus.

Type `./fileprob.py --help` to see documentation.

Once you've familiarized yourself with the arguments, try
to run the script like this:

```
Usage:   ./fileprob.py TRAIN vocab smoother lexicon trainpath
         ./fileprob.py TEST vocab smoother lexicon trainpath files...
Example: ./fileprob.py TRAIN vocab-genspam.txt add0.01 ../lexicons/words-10.txt ../data/speech/train/switchboard-small
         ./fileprob.py TEST  vocab-genspam.txt add0.01 ../lexicons/words-10.txt ../data/speech/train/switchboard-small ../data/speech/sample*
```

Note: It may be convenient to use symbolic links (shortcuts) to avoid
typing long filenames.  For example,

	ln -sr ../data/speech/train sptrain 

will create a subdirectory `sptrain` in the current directory, which
is really a shortcut to `../data/speech/train`.

`fileprob.py` automatically computes the model name from the vocab, smoother, 
lexicon, and training corpus that was used to train the model. 
While some smoothers (including add-lambda and uniform) ignores the lexicon, 
some won't (all the log-linear models will make use of them). 
For consistency of naming, lexicon is always required.

----------

## QUESTION 2.

Copy `fileprob.py` to `textcat.py`.

Modify `textcat.py` so that it does text categorization.

For each training corpus, you should create a new language model.  You
will first need to call `set_vocab()` on the pair of corpora, so
that both models use the same vocabulary (derived from the union of
the two corpora).  Note that `set_vocab` can take multiple files as
arguments.  You can re-use the current LanguageModel object by using the
following strategy.

```
  (In TRAIN mode)
  call lm.set_vocab_size() on the pair of training corpora

  train model 1 from corpus 1
  train model 2 from corpus 2
  store the model parameters
  terminate program

  ===

  (In TEST mode)
  restore model 1 and model 2 the previously saved parameters
  for each file,
    compute its probability under model 1: save this in an array

  for each file,
    compute its probability under model 2: save this in an array
  loop over the arrays and print the results
```

There are other ways to solve this problem, if you prefer. However, we
require your `textcat.py` to have two modes, `TRAIN` and `TEST`. The
`TRAIN` mode should train the models and save the parameters. The
`TEST` mode should load the previously saved parameters and
compute/print the results without looking at corpus 1 and 2.

----------

## QUESTION 5.

Open `Probs.py`.  Implement the `prob()` function for the model BACKOFF_ADDL.

Remember you need to handle OOV words, and make sure the probabilities
of all possible words after a given context sums up to one.

As you are only adding a new model, the behavior of your old models such
as ADDL should not change.

----------------------------------------------------------------------

## QUESTION 6.

Now add the `sample()` method. Did your good OOP principles suggest the best
place to do this?

Be sure to include a maximum length limit.  Otherwise your program may
try to generate very long sequences.

----------------------------------------------------------------------

## QUESTION 7.

(a) Now complete the LOGLIN model in `EmbeddingLogLinearLanguageModel`.

	Remember that you need to look up an embedding for each word, falling
	back to the OOL embedding if that word is not in the lexicon
	(including OOV).

(b) Use stochastic gradient descent (ascent) and back-propagation in
the `train()` function in `Probs.py`.

(d) Complete the IMPROVED model as `ImprovedLogLinearLanguageModel`.
This is a subclass of the LOGLIN model, so you can inherit or override
methods as you like.

As you are only adding new models, the behavior of your old models
should not change.

### Using vector/matrix operations (crucial for speed!):

Training the log-linear model on `en.1K` can be done with simple "for" loops and
2D array representation of matrices.  However, you're encouraged to use
PyTorch's matrix/vector operations, which will reduce training time and 
might simplify your code.

TA's note: my original implementation took 22 hours per epoch. Careful
vectorization of certain operations, leveraging PyTorch, brought that
runtime down to 13 minutes per epoch.

Make sure to use the torch.logsumexp method for computing the log-denominator
in the log-probability.

----------------------------------------------------------------------

## QUESTION 9 (EXTRA CREDIT)

In this question, you're back to having only one language model as in
`fileprob` (not two as in `textcat`).  So, initialize `speechrec.py`
to a copy of `fileprob.py`, and then edit it.

You shouldn't have to change the `TRAIN` mode.

But modify speechrec.py so that in `TEST` mode, instead of evaluating
the prior probability of the entire test file, it separately evaluates
the prior probability of each candidate transcription in the file.  It
can then select the transcription with the highest *posterior*
probability and report its error rate, as required.

The `get_trigrams` function in `Probs.py` is no longer useful at
`TEST` time, since a speech dev or test file has a special format.
You don't want to iterate over all the trigrams in such a file.  You
may want to make an "outer loop" utility function that iterates over
the candidate transcriptions in a given speech dev or test file,
along with an "inner loop" utility function that iterates over the
trigrams in a given candidate transcription.

(The outer loop is specialized to the speechrec format, so it probably
belongs in `speechrec.py`.  The inner loop is similar to
`get_trigrams` and might be more generally useful, so it probably
belongs in `Probs.py`.)

