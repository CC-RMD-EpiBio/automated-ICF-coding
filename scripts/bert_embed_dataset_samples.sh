#!/bin/bash
###############################################################################
#
#                           COPYRIGHT NOTICE
#                  Mark O. Hatfield Clinical Research Center
#                       National Institutes of Health
#            United States Department of Health and Human Services
#
# This software was developed and is owned by the National Institutes of
# Health Clinical Center (NIHCC), an agency of the United States Department
# of Health and Human Services, which is making the software available to the
# public for any commercial or non-commercial purpose under the following
# open-source BSD license.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# (1) Redistributions of source code must retain this copyright
# notice, this list of conditions and the following disclaimer.
# 
# (2) Redistributions in binary form must reproduce this copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# (3) Neither the names of the National Institutes of Health Clinical
# Center, the National Institutes of Health, the U.S. Department of
# Health and Human Services, nor the names of any of the software
# developers may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# (4) Please acknowledge NIHCC as the source of this software by including
# the phrase "Courtesy of the U.S. National Institutes of Health Clinical
# Center"or "Source: U.S. National Institutes of Health Clinical Center."
# 
# THIS SOFTWARE IS PROVIDED BY THE U.S. GOVERNMENT AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
# 
# You are under no obligation whatsoever to provide any bug fixes,
# patches, or upgrades to the features, functionality or performance of
# the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to
# the National Institutes of Health Clinical Center, without imposing a
# separate written license agreement for such Enhancements, then you hereby
# grant the following license: a non-exclusive, royalty-free perpetual license
# to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such Enhancements or
# derivative works thereof, in binary and source code form.
#
###############################################################################

#
# Demo script, illustrating steps to get HDF5-format embeddings
# from BERT, with aligned tokenization.
#

set -e

export PYTHONPATH=$(pwd):${PYTHONPATH}
DATA=$(pwd)/data

if [ -z "$1" ]; then
	MODEL=BERT-Base
else
	MODEL=$1
fi

if [ -z "$2" ]; then
    DATASET=demo
else
    DATASET=$2
fi

cd bert_to_hdf5

#PYTHON=python
PYTHON=~/environments/tensorflow-gpu/bin/python3
LAYERS=-1,-2,-3,-4
BERT=bert
GPU=7

## Fetch BERT files from per-model specifications in config.ini
VOCABFILE=$(${PYTHON} -m cli_configparser.read_setting -c config.ini BERT "${MODEL} VocabFile")
CONFIGFILE=$(${PYTHON} -m cli_configparser.read_setting -c config.ini BERT "${MODEL} ConfigFile")
CKPTFILE=$(${PYTHON} -m cli_configparser.read_setting -c config.ini BERT "${MODEL} CkptFile")

INPUT=${DATA}/${DATASET}.BERT__${MODEL}.mention_text
PRE_TOKENIZED=${DATA}/${DATASET}.BERT__${MODEL}.mention_text.pre_tokenized_for_BERT
JSON=${DATA}/${DATASET}.BERT__${MODEL}.mention_text.json1
HDF5=${DATA}/${DATASET}.BERT__${MODEL}.mention_text.hdf5
SEQ_LENGTH=512

export PYTHONPATH=${PYTHONPATH}:${BERT}

if [ ! -e "${PRE_TOKENIZED}.tokens" ]; then
    ${PYTHON} -m pre_tokenize_for_BERT \
        -i ${INPUT} \
        -o ${PRE_TOKENIZED} \
        -s ${SEQ_LENGTH} \
        --trim \
        --tokenized
fi

if [ ! -e "${JSON}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU}
    ${PYTHON} -m extract_features_pretokenized \
        --input_file ${PRE_TOKENIZED}.subsequences \
        --output_file ${JSON} \
        --vocab_file ${VOCABFILE} \
        --bert_config_file ${CONFIGFILE} \
        --init_checkpoint ${CKPTFILE} \
        --layers ${LAYERS} \
        --max_seq_length ${SEQ_LENGTH} \
        --batch_size 8
fi

if [ ! -e "${HDF5}" ]; then
    ${PYTHON} -m recombine_BERT_embeddings \
        --bert-output ${JSON} \
        --overlaps ${PRE_TOKENIZED}.overlaps \
        -o ${HDF5} \
        --tokenized ${HDF5}.aligned_tokens.txt \
        -l ${HDF5}.log
fi
