#!/usr/bin/env bash

# Do not change these
SMOOTHER="log_linear_improved"
SMOOTHER_ALIAS="improved"
ROOT_DIR=./hw-lm
VOCAB_DIR=.

# Feel free to modify these

# Settings for the gen-spam framework
GEN_OR_SPAM="gen"
VOCAB_FILE=${VOCAB_DIR}/vocab-genspam.txt
TRAIN_FILE=${ROOT_DIR}/data/gen_spam/train/${GEN_OR_SPAM}
VAL_FILE=${ROOT_DIR}/data/gen_spam/dev/${GEN_OR_SPAM}
LEXICON=${ROOT_DIR}/lexicons/words-gs-50.txt
LEXICON_ALIAS="wgs50"

# Settings for the english-spanish framework
#ENGLISH_OR_SPANISH="en"
#VOCAB_FILE=${VOCAB_DIR}/vocab-en1K.txt
#TRAIN_FILE=${ROOT_DIR}/data/english_spanish/train/en.1K
#VAL_FILE=${ROOT_DIR}/data/english_spanish/dev/english
#LEXICON=${ROOT_DIR}/lexicons/chars-10.txt
#LEXICON_ALIAS="c10"

# General settings
L2REG=1.0
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
MAX_EPOCHS=50

MODEL_BASENAME=$GEN_OR_SPAM
MODEL_OUTPUT=$MODEL_BASENAME"_"$LEXICON_ALIAS"_"$SMOOTHER_ALIAS".model"

./train_lm.py $VOCAB_FILE $SMOOTHER $TRAIN_FILE --output $MODEL_OUTPUT --lexicon $LEXICON --max_epochs $MAX_EPOCHS \
	--l2_regularization $L2REG --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE --val_file $VAL_FILE
