#!/usr/bin/env bash

ROOT_DIR=./hw-lm

# Settings for the gen-spam framework
TEST_DIR_GEN=${ROOT_DIR}/data/gen_spam/test/gen
TEST_DIR_SPAM=${ROOT_DIR}/data/gen_spam/test/spam

# Settings for the english-spanish framework
TEST_DIR_EN=${ROOT_DIR}/data/english_spanish/test/english
TEST_DIR_SP=${ROOT_DIR}/data/english_spanish/test/spanish

MODEL1=$1
MODEL2=$2
PRIOR1=$3

# Modify these to fit the task you want to run (gen-spam vs. english-spanish)
MODEL1_TEST_DIR=$TEST_DIR_GEN
MODEL2_TEST_DIR=$TEST_DIR_SPAM

./textcat.py $MODEL1 $MODEL2 $PRIOR1 --eval True --model_1_test_dir $MODEL1_TEST_DIR --model_2_test_dir $MODEL2_TEST_DIR
