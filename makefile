PY=python

setup:
	${PY} -m spacy download en_core_web_sm
	git clone https://github.com/google-research/bert.git
	curl -o data/uncased_L-12_H-768_A-12.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
	cd data && unzip uncased_L-12_H-768_A-12.zip

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
		-o data/$${DATASET}.SpaCy.mentions

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
		-o data/$${DATASET}.BERT__$${MODEL}.mentions
