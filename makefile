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
	git clone https://github.com/drgriffis/bert_to_hdf5.git
	curl -o data/uncased_L-12_H-768_A-12.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
	cd data && unzip uncased_L-12_H-768_A-12.zip
	mkdir data/classification_experiments
	mkdir data/cross_validation_splits
	mkdir data/BERT_FT_baseline
	echo "=========================================="
	echo "Please download pre-trained word2vec GoogleNews vectors from:"
	echo "  https://code.google.com/archive/p/word2vec/"
	echo
	echo "GUnzip the file and place"
	echo "  GoogleNews-vectors-negative300.bin"
	echo "into data/"
	echo "=========================================="



#########################################################################
## Data preprocessing ###################################################
#########################################################################


#### Base preprocessing (SpaCy) #########################################

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



#### BERT preprocessing for sklearn classification / candidate selection 

## BERT preprocessing step (1) - extract the text mentions file
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

## BERT processing step (2) - split out mention files to line-aligned text and mention keys
prep_bert_mentions_for_embedding:
	@if [ -z "${DATASET}" ]; then echo "DATASET must be specified"; fi; \
	if [ -z "${MODEL}" ]; then MODEL=BERT-Base; else MODEL=${MODEL}; fi; \
	${PY} -m utils.prep_mentions_for_contextualized_embedding \
		-m ${DATA}/${DATASET}.BERT__$${MODEL}.mentions \
		-o ${DATA}/${DATASET}.BERT__$${MODEL}.mention_text \
		-k ${DATA}/${DATASET}.BERT__$${MODEL}.mention_keys \
		-l ${DATA}/${DATASET}.BERT__$${MODEL}.prep.log

## BERT processing step (3) - run BERT on mention files
run_bert_on_mention_texts:
	@if [ -z "${DATASET}" ]; then echo "DATASET must be specified"; fi; \
	if [ -z "${MODEL}" ]; then MODEL=BERT-Base; else MODEL=${MODEL}; fi; \
	scripts/bert_embed_dataset_samples.sh $${MODEL} ${DATASET}

## BERT processing step (4) - convert HDF5 format embeddings to embedded mentions file
convert_bert_output_to_embedded_mentions:
	@if [ -z "${DATASET}" ]; then echo "DATASET must be specified"; fi; \
	if [ -z "${MODEL}" ]; then MODEL=BERT-Base; else MODEL=${MODEL}; fi; \
	if [ -z "${ORACLE}" ]; then \
		ORACLEFLAG=; \
		ORACLELBL=; \
	else \
		ORACLEFLAG="--action-oracle"; \
		ORACLELBL=.action_oracle; \
	fi; \
	${PY} -m utils.collapse_hdf5_mention_embeddings \
		$${ORACLEFLAG} \
		-i ${DATA}/${DATASET}.BERT__$${MODEL}.mention_text.hdf5 \
		-k ${DATA}/${DATASET}.BERT__$${MODEL}.mention_keys \
		-m ${DATA}/${DATASET}.BERT__$${MODEL}.mentions \
		-o ${DATA}/${DATASET}.BERT__$${MODEL}.embedded$${ORACLELBL}.mentions \
		-l ${DATA}/${DATASET}.BERT__$${MODEL}.embedded$${ORACLELBL}.mentions.log



#### BERT preprocessing for fine-tuning experiments #####################

## BERT fine-tune baseline processing step (1) - generate version of SpaCy-tokenized mentions for BERT fine tuning
generate_bert_finetune_files:
	@${PY} -m utils.bert_convert \
		-m ${DATA}/BTRIS_Mobility.mobility_ctx.SpaCy.mentions \
		-s ${DATA}/cross_validation_splits/splits.mobility_ctx \
		-o ${DATA}/BERT_FT_baseline \
		-l ${DATA}/BERT_FT_baseline/bert_convert.log



#########################################################################
## Experiments ##########################################################
#########################################################################


generate_xval_splits:
	@if [ -z "${K}" ]; then K=5; else K=${K}; fi; \
	if [ -z "${DATASET}" ]; then DATASET=demo; else DATASET=${DATASET}; fi; \
	${PY} -m experiments.generate_xval_splits \
		-k $${K} \
		-m ${DATA}/$${DATASET}.SpaCy.mentions \
		-o ${DATA}/cross_validation_splits/$${DATASET} \
		--mention-map ${DATA}/$${DATASET}.SpaCy.mentions.mention_map \
		-l ${DATA}/cross_validation_splits/$${DATASET}.log



#### SciKit-learn classifiers ###########################################

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
		MENTIONS=${DATA}/${DATASET}.${BERTMODEL}.embedded$${ORACLELBL}.mentions; \
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




#### BERT fine-tuning (classification) ##################################

## BERT fine-tune baseline processing step (2) - run BERT fine-tuning experiment
run_bert_finetune:
	@if [ -z "${FOLD}" ]; then \
		FOLD=0; \
	else \
		FOLD=${FOLD}; \
	fi; \
	if [ -z "${MODEL}" ]; then \
		MODEL=BERT-Base; \
	else \
		MODEL=${MODEL}; \
	fi; \
	if [ -z "${EPOCHS}" ]; then \
		EPOCHS=3; \
		EPOCHSFLAG=; \
	else \
		EPOCHS=${EPOCHS}; \
		EPOCHSFLAG="-${EPOCHS}"; \
	fi; \
	export CUDA_VISIBLE_DEVICES=${GPU}; \
	VOCABFILE=$$(${PY} -m cli_configparser.read_setting -c config.ini BERT "$${MODEL} Vocabfile"); \
	CONFIGFILE=$$(${PY} -m cli_configparser.read_setting -c config.ini BERT "$${MODEL} ConfigFile"); \
	CKPTFILE=$$(${PY} -m cli_configparser.read_setting -c config.ini BERT "$${MODEL} CkptFile"); \
	OUTPUT_DIR=${DATA}/BERT_FT_baseline/fold-$${FOLD}/$${MODEL}$${EPOCHSFLAG}; \
	if [ ! -d $${OUTPUT_DIR} ]; then mkdir -p $${OUTPUT_DIR}; fi; \
	cp utils/modified_BERT_run_classifier.py bert/run_classifier.py; \
	cd bert; \
	${PY} -m run_classifier \
		--task_name=ICFMobility \
		--do_train=true \
		--do_predict=true  \
		--data_dir=${DATA}/BERT_FT_baseline/fold-$${FOLD} \
		--vocab_file $${VOCABFILE} \
		--bert_config_file $${CONFIGFILE} \
		--init_checkpoint $${CKPTFILE} \
		--max_seq_length=128 \
		--train_batch_size=25 \
		--learning_rate=2e-5 \
		--num_train_epochs=$${EPOCHS}.0 \
		--output_dir=$${OUTPUT_DIR}





#########################################################################
## Analysis #############################################################
#########################################################################


#### SciKit learn classifiers ###########################################

analyze_classifier:
	@if [ -z "${DATASET}" ]; then echo "Must specify DATASET"; exit; fi; \
	if [ -z "${MODEL}" ]; then MODEL=SVM; else MODEL=${MODEL}; fi; \
	PREDSF=$$(ls ${DATA}/classification_experiments/${DATASET}.$${MODEL}.predictions.* | grep -v per_code_performance | sort | tail -n 1); \
	if [ -z "${TEST}" ]; then \
		MODEFLAG="--dev"; \
	else \
		MODEFLAG=; \
	fi; \
	${PY} -m analysis.per_code_performance \
		${DATA}/${DATASET}.SpaCy.mentions \
		$${PREDSF} \
		$${MODEFLAG} \
		--cross-validation-splits ${DATA}/cross_validation_splits/${DATASET} \
		--no-scores \
		-l $${PREDSF}.per_code_performance



#### BERT fine-tuning (classification) ##################################

## BERT fine-tune baseline processing step (3) - combine per-fold predictions into single file
compile_bert_finetune_predictions:
	@if [ -z "${DATASET}" ]; then echo "Must specify DATASET"; exit; fi; \
	if [ -z "${MODEL}" ]; then MODEL=BERT-Base; else MODEL=${MODEL}; fi; \
	${PY} -m utils.compile_bert_predictions \
		-m ${DATA}/${DATASET}.SpaCy.mentions \
		--bert-dir ${DATA}/BERT_FT_baseline \
		--model $${MODEL}


## BERT fine-tune baseline processing step (4) - analyze results from FT experiment
analyze_bert_finetune_predictions:
	@if [ -z "${DATASET}" ]; then echo "Must specify DATASET"; exit; fi; \
	if [ -z "${MODEL}" ]; then MODEL=BERT-Base; else MODEL=${MODEL}; fi; \
	${PY} -m analysis.per_code_performance \
		${DATA}/${DATASET}.SpaCy.mentions \
		${DATA}/BERT_FT_baseline/$${MODEL}.compiled_output.predictions \
		-l ${DATA}/BERT_FT_baseline/$${MODEL}.compiled_output.evaluation.log
