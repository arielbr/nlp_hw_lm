#!/usr/bin/env bash

# Do not change these
SMOOTHER="log_linear_improved"
SMOOTHER_ALIAS="improved"
ROOT_DIR=./hw-lm
VOCAB_DIR=.

# Feel free to modify these

# Settings for the gen-spam framework
#GEN_OR_SPAM="spam"
#VOCAB_FILE=${VOCAB_DIR}/vocab-genspam.txt
#TRAIN_FILE=${ROOT_DIR}/data/gen_spam/train/${GEN_OR_SPAM}
#VAL_FILE=${ROOT_DIR}/data/gen_spam/dev/${GEN_OR_SPAM}
#LEXICON=${ROOT_DIR}/lexicons/words-10.txt
#LEXICON_ALIAS="w10"
#MODEL_BASENAME=$GEN_OR_SPAM

# Settings for the english-spanish framework
ENGLISH_OR_SPANISH="en"
TRAIN_SIZE="1K"
VOCAB_FILE=${VOCAB_DIR}/vocab-ensp1K.txt
TRAIN_FILE=${ROOT_DIR}/data/english_spanish/train/${ENGLISH_OR_SPANISH}.${TRAIN_SIZE}
VAL_FILE=${ROOT_DIR}/data/english_spanish/dev/english  # make sure to change this line manually!!!
LEXICON=${ROOT_DIR}/lexicons/chars-10.txt
LEXICON_ALIAS="c10"
MODEL_BASENAME=$ENGLISH_OR_SPANISH$TRAIN_SIZE

# General settings
L2REG=1.0
TRAIN_BATCH_SIZE=64 #originally 64
VAL_BATCH_SIZE=64 #originally 64
MAX_EPOCHS=200
PATIENCE=10
LEARNING_RATE=0.005 # for ConvergentSGD, 0.0005 is a good initial learning rate for gen-spam, and 0.01 is good for english-spanish
OPTIMIZER="Adam"

MODEL_OUTPUT=$MODEL_BASENAME"_"$LEXICON_ALIAS"_"$SMOOTHER_ALIAS"_2.model"

./train_lm.py $VOCAB_FILE $SMOOTHER $TRAIN_FILE --output $MODEL_OUTPUT --lexicon $LEXICON --max_epochs $MAX_EPOCHS \
	--l2_regularization $L2REG --train_batch_size $TRAIN_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE --val_file $VAL_FILE \
	--patience $PATIENCE --learning_rate $LEARNING_RATE --optimizer $OPTIMIZER
