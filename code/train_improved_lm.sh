#!/usr/bin/env bash

# Do not change these
SMOOTHER="log_linear_improved"
SMOOTHER_ALIAS="improved"
ROOT_DIR=./hw-lm
VOCAB_DIR=.

# Feel free to modify these

GEN_OR_SPAM="gen"
VOCAB_FILE=${VOCAB_DIR}/vocab-genspam.txt
TRAIN_FILE=${ROOT_DIR}/data/gen_spam/train/${GEN_OR_SPAM}
VAL_FILE=${ROOT_DIR}/data/gen_spam/dev/${GEN_OR_SPAM}
LEXICON=${ROOT_DIR}/lexicons/words-gs-50.txt

L2REG=1.0
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
MAX_EPOCHS=50

MODEL_OUTPUT=$GEN_OR_SPAM"_"$SMOOTHER_ALIAS".model"


./train_lm.py $VOCAB_FILE $SMOOTHER $TRAIN_FILE --output $MODEL_OUTPUT --lexicon $LEXICON --max_epochs $MAX_EPOCHS \
	--l2_regularization $L2REG --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE --val_file $VAL_FILE
