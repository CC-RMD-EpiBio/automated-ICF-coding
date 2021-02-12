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

PY=python
DATA=data

setup:
	${PY} -m spacy download en_core_web_sm
	git clone https://github.com/google-research/bert.git
	curl -o data/uncased_L-12_H-768_A-12.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
	cd data && unzip uncased_L-12_H-768_A-12.zip
	mkdir data/classification_experiments
	mkdir data/cross_validation_splits
	echo "=========================================="
	echo "Please download pre-trained word2vec GoogleNews vectors from:"
	echo "  https://code.google.com/archive/p/word2vec/"
	echo
	echo "GUnzip the file and place"
	echo "  GoogleNews-vectors-negative300.bin"
	echo "into data/"
	echo "=========================================="

preprocess_dataset_spacy:
	@if [ -z "${DATASET}" ]; then \
		DATASET=demo; \
	else \
		DATASET=${DATASET}; \
	fi; \
	${PY} -m dataset.extract_mentions \
		-c config.ini \
		-t SpaCy \
		--dataset $${DATASET} \
		-o ${DATA}/$${DATASET}.SpaCy.mentions

preprocess_dataset_bert:
	@if [ -z "${DATASET}" ]; then \
		DATASET=demo; \
	else \
		DATASET=${DATASET}; \
	fi; \
	if [ -z "${MODEL}" ]; then \
		MODEL=BERT-Base; \
	else \
		MODEL=${MODEL}; \
	fi; \
	VOCABFILE=$$(python -m cli_configparser.read_setting -c config.ini BERT "$${MODEL} Vocabfile"); \
	${PY} -m dataset.extract_mentions \
		-c config.ini \
		-t BERT \
		--bert-vocab-file $${VOCABFILE} \
		--dataset $${DATASET} \
		-o ${DATA}/$${DATASET}.BERT__$${MODEL}.mentions


generate_xval_splits:
	@if [ -z "${K}" ]; then K=5; else K=${K}; fi; \
	if [ -z "${DATASET}" ]; then DATASET=demo; else DATASET=${DATASET}; fi; \
	${PY} -m experiments.generate_xval_splits \
		-k $${K} \
		-m ${DATA}/$${DATASET}.SpaCy.mentions \
		-o ${DATA}/cross_validation_splits/$${DATASET} \
		--mention-map ${DATA}/$${DATASET}.SpaCy.mentions.mention_map \
		-l ${DATA}/cross_validation_splits/$${DATASET}.log


run_classifier:
	@if [ -z "${DATASET}" ]; then echo "Must specify DATASET"; exit; fi; \
	if [ -z "${MODEL}" ]; then MODEL=SVM; else MODEL=${MODEL}; fi; \
	if [ -z "${DEV}" ]; then \
		DEVFLAG=; \
		DEVLBL=test; \
	else \
		DEVFLAG="--eval-on-dev"; \
		DEVLBL=dev; \
	fi; \
	if [ -z "${ORACLE}" ]; then \
		ORACLEFLAG=; \
		ORACLELLBL=; \
	else \
		ORACLEFLAG="--action-oracle"; \
		ORACLELBL=.action_oracle; \
	fi; \
	if [ -z "${CTXEMBS}" ]; then \
		CTXEMBFLAG=--no-ctx-embeddings; \
		CTXEMBLBL=; \
	else \
		CTXEMBSF=$$(${PY} -m cli_configparser.read_setting -c config.ini "Static Embeddings" "${CTXEMBS}"); \
		CTXEMBSFORMAT=$$(${PY} -m cli_configparser.read_setting -c config.ini "Static Embeddings" "${CTXEMBS} Format"); \
		CTXEMBFLAG="--ctxs $${CTXEMBSF} --ctxs-format $${CTXEMBSFORMAT}"; \
		CTXEMBLBL=.ctx_embs.${CTXEMBS}; \
	fi; \
	if [ -z "${UNIGRAMS}" ]; then \
		UNIGRAMFLAG=; \
		UNIGRAMLBL=; \
	else \
		UNIGRAMFLAG=--unigram-features; \
		UNIGRAMLBL=.unigrams; \
		if [ ! -z "${TFIDF}" ]; then \
			UNIGRAMFLAG="$${UNIGRAMFLAG} --tfidf"; \
			UNIGRAMLBL=$${UNIGRAMLBL}-tfidf; \
		fi; \
	fi; \
	if [ -z "${BERT}" ]; then \
		MENTIONS=${DATA}/${DATASET}.SpaCy.mentions; \
		PREEMBFLAG=; \
		EMBMODEFLAG=; \
	else \
		if [ -z "${BERTMODEL}" ]; then \
			echo BERTMODEL must be specified; \
			exit; \
		fi; \
		MENTIONS=${DATA}/${DATASET}.${BERTMODEL}.IDs_remapped.embedded$${ORACLELBL}.mentions; \
		CTXEMBFLAG=--no-ctx-embeddings; \
		PREEMBFLAG=--pre-embedded; \
		EMBMODEFLAG=.pre_embedded.${BERTMODEL}; \
	fi; \
	${PY} -m experiments.sklearn_classifiers \
		$${MENTIONS} \
		--n-fold 5 \
		--cross-validation-splits ${DATA}/cross_validation_splits/${DATASET} \
		--classifier $${MODEL} \
		--no-entities \
		$${DEVFLAG} \
		$${CTXEMBFLAG} \
		$${UNIGRAMFLAG} \
		$${PREEMBFLAG} \
		$${ORACLEFLAG} \
		-l ${DATA}/classification_experiments/${DATASET}.$${MODEL}$${CTXEMBLBL}$${UNIGRAMLBL}$${EMBMODEFLAG}$${ORACLELBL}.$${DEVLBL}.log
