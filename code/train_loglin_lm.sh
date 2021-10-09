#!/usr/bin/env bash

# Do not change these
SMOOTHER="log_linear"
SMOOTHER_ALIAS="loglin"
ROOT_DIR=./hw-lm
VOCAB_DIR=.

# Feel free to modify these

# Settings for the gen-spam framework
#GEN_OR_SPAM="spam"
#VOCAB_FILE=${VOCAB_DIR}/vocab-genspam.txt
#TRAIN_FILE=${ROOT_DIR}/data/gen_spam/train/${GEN_OR_SPAM}
#LEXICON=${ROOT_DIR}/lexicons/words-10.txt
#LEXICON_ALIAS="w10"

# Settings for the english-spanish framework
ENGLISH_OR_SPANISH="sp"
TRAIN_SIZE="1K"
VOCAB_FILE=${VOCAB_DIR}/vocab-ensp1K.txt
TRAIN_FILE=${ROOT_DIR}/data/english_spanish/train/${ENGLISH_OR_SPANISH}.${TRAIN_SIZE}
LEXICON=${ROOT_DIR}/lexicons/chars-10.txt
LEXICON_ALIAS="c10"

# General settings
L2REG=1.0
LEARNING_RATE=0.01 # instructions say to choose 0.1 for gen-spam and 0.01 for english-spanish
MAX_EPOCHS=10 # instructions say to stick for 10 for our initial log-linear models
#MODEL_BASENAME=$GEN_OR_SPAM
MODEL_BASENAME=$ENGLISH_OR_SPANISH$TRAIN_SIZE


MODEL_OUTPUT=$MODEL_BASENAME"_"$LEXICON_ALIAS"_"$SMOOTHER_ALIAS".model"

./train_lm.py $VOCAB_FILE $SMOOTHER $TRAIN_FILE --output $MODEL_OUTPUT --lexicon $LEXICON --l2_regularization $L2REG \
	--max_epochs $MAX_EPOCHS --learning_rate $LEARNING_RATE
