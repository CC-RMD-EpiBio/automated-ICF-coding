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
