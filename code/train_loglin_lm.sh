#!/usr/bin/env bash

# Do not change these
SMOOTHER="log_linear"
SMOOTHER_ALIAS="loglin"
ROOT_DIR=./hw-lm
VOCAB_DIR=.

# Feel free to modify these

# Settings for the gen-spam framework
#GEN_OR_SPAM="gen"
#VOCAB_FILE=${VOCAB_DIR}/vocab-genspam.txt
#TRAIN_FILE=${ROOT_DIR}/data/gen_spam/train/${GEN_OR_SPAM}
#LEXICON=${ROOT_DIR}/lexicons/words-gs-50.txt
#LEXICON_ALIAS="wgs50"

# Settings for the english-spanish framework
ENGLISH_OR_SPANISH="en"
TRAIN_SIZE="1K"
VOCAB_FILE=${VOCAB_DIR}/vocab-ensp1K.txt
TRAIN_FILE=${ROOT_DIR}/data/english_spanish/train/${ENGLISH_OR_SPANISH}.${TRAIN_SIZE}
LEXICON=${ROOT_DIR}/lexicons/chars-10.txt
LEXICON_ALIAS="c10"

# General settings
L2REG=1.0
MODEL_BASENAME=$ENGLISH_OR_SPANISH$TRAIN_SIZE


MODEL_OUTPUT=$MODEL_BASENAME"_"$LEXICON_ALIAS"_"$SMOOTHER_ALIAS".model"

./train_lm.py $VOCAB_FILE $SMOOTHER $TRAIN_FILE --output $MODEL_OUTPUT --lexicon $LEXICON --l2_regularization $L2REG
