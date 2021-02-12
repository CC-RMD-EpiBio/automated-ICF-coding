'''
'''

import sys
import os
import glob
import codecs
import numpy as np
import random
from types import SimpleNamespace
import nltk
import time
from datetime import datetime
from hedgepig_logger import log
import scipy.stats
import pyemblib
import scipy.sparse
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from dataset import mention_file
from . import cross_validation
from .util import *


class Classifier:
    SVM = 'SVM'
    KNN = 'KNN'
    MLP = 'MLP'

    @staticmethod
    def tolist():
        return [
            Classifier.SVM,
            Classifier.KNN,
            Classifier.MLP
        ]
    @staticmethod
    def default():
        return Classifier.SVM

def getTextVocabulary(mentions, preprocessed, options):
    vocab = set()
    for m in mentions:
        strings = [m.left_context, m.right_context, m.mention_text]
        for context_string in strings:
            words = [s.strip() for s in context_string.lower().split()]
            for w in words:
                vocab.add(w)
    return vocab

def extractContexts(full_text):
    # assuming already tokenized at this point
    tokens = [s.strip() for s in full_text.split()]

    try:
        start_ix = tokens.index('<e>')
        end_ix = tokens.index('</e>')
    except Exception as e:
        print(tokens)
        raise e

    left_context = ' '.join(tokens[:start_ix])
    right_context = ' '.join(tokens[end_ix+1:])

    return (left_context, right_context)

def preprocessData(mentions, entity_embeds, ctx_embeds, options):
    preprocessed = SimpleNamespace()

    preprocessed.mentions = mentions
    preprocessed.entity_embeds = entity_embeds
    preprocessed.ctx_embeds = ctx_embeds

    preprocessed.mentions_by_id = {
        m.ID : m
            for m in mentions
    }
    preprocessed.labels_by_id = {
        m.ID : m.CUI.strip().lower()
            for m in mentions
    }

    persistent_path = options.cross_validation_file

    preprocessed.splits = cross_validation.crossValidationSplits(
        preprocessed.labels_by_id,
        n_folds=options.n_folds,
        dev_size=options.dev_size,
        persistent_path=persistent_path,
        random_seed=options.random_seed,
        log=log
    )

    # set up unigram features for all mentions
    # (CountVectorizer needs to call fit_transform on all strings at once to
    #  set up its vocabulary correctly)
    if options.unigram_features:
        # first, compile all of the texts of each mention into a single
        # list, with an index by mention ID
        preprocessed.mention_ID_to_unigram_rows = {}
        texts = []
        for i in range(len(mentions)):
            mention_ID = mentions[i].ID
            preprocessed.mention_ID_to_unigram_rows[mention_ID] = i
            m = mentions[i]
            strings = [m.left_context, m.right_context, m.mention_text]
            cleaned_text = []
            for string in strings:
                cleaned_text.extend([
                    s.strip()
                        for s in string.lower().split()
                ])
            texts.append(' '.join(cleaned_text))

        # identify the words that appear within each fold, then in the
        # overall collection of all texts (may have more words than any
        # individual fold)
        per_fold_unigram_vocabs = [
            getTextVocabulary(
                [
                    m
                        for m in preprocessed.mentions
                        if m.ID in train_ids
                ],
                preprocessed,
                options
            )
                for (train_ids, _, _) in preprocessed.splits
        ]
        global_unigram_vocab = getTextVocabulary(
            preprocessed.mentions,
            preprocessed,
            options
        )

        # vectorize either with TF-IDF values or with straight binary
        # counts, depending on runtime configuration
        if options.unigrams_as_tfidf:
            per_fold_unigram_vectorizers = [
                TfidfVectorizer(
                    vocabulary=per_fold_unigram_vocab,
                    binary=True
                )
                    for per_fold_unigram_vocab in per_fold_unigram_vocabs
            ]
            global_unigram_vectorizer = TfidfVectorizer(
                vocabulary=global_unigram_vocab,
                binary=True
            )
        else:
            per_fold_unigram_vectorizers = [
                CountVectorizer(
                    vocabulary=per_fold_unigram_vocab,
                    binary=True
                )
                    for per_fold_unigram_vocab in per_fold_unigram_vocabs
            ]
            global_unigram_vectorizer = CountVectorizer(
                vocabulary=global_unigram_vocab,
                binary=True
            )

        preprocessed.global_unigram_features = global_unigram_vectorizer.fit_transform(
            texts
        ).tocsr()
        preprocessed.per_fold_unigram_features = [
            per_fold_unigram_vectorizer.fit_transform(
                texts
            ).tocsr()
                for per_fold_unigram_vectorizer in per_fold_unigram_vectorizers
        ]
    else:
        preprocessed.mention_ID_to_unigram_rows = None
        preprocessed.global_unigram_features = None
        preprocessed.per_fold_unigram_features = [
            None for _ in range(len(preprocessed.splits))
        ]

    return preprocessed

def prepSample(mention, preprocessed, unigram_feature_set, options):
    '''Given a mention, a set of word embeddings, and a list of sets of
    entity embeddings, returns:
       (1) Feature vector consisting of:
            - Average embedding of context words
            - Embeddings of each candidate entity, for entities in the vocabulary
           (if no valid context words in vocabulary, this value is None)
       (2) Point label
           (or None if correct entity is not in embedding vocabulary)
    '''
    feature_vector, label = [], None

    # if it's pre-embedded, that's the only context/mention feature we can use
    if options.pre_embedded:
        feature_vector.append(mention.context_repr)
    else:
        # context embedding
        if options.use_ctx_embeddings:
            strings = [mention.left_context, mention.right_context]
            if not options.action_oracle:
                strings.append(mention.mention_text)

            ctx_embedding, valid_ctx_embedding = meanEmbeddings(
                strings,
                preprocessed.ctx_embeds
            )
            if not options.unigram_features:
                feature_vector.append(ctx_embedding)
            else:
                feature_vector.append(scipy.sparse.coo_matrix(ctx_embedding))

            # mention text embedding
            if options.action_oracle:
                mention_embedding, valid_mention_embedding = meanEmbeddings(
                    [mention.mention_text],
                    preprocessed.ctx_embeds
                )
                if not options.unigram_features:
                    feature_vector.append(mention_embedding)
                else:
                    feature_vector.append(scipy.sparse.coo_matrix(ctx_embedding))

        # context/mention unigram features (pre-calculated)
        if options.unigram_features:
            feature_vector.append(
                unigram_feature_set.getrow(
                    preprocessed.mention_ID_to_unigram_rows[mention.ID]
                ).tocoo()
            )

    # entity features
    if options.use_entity_embeddings:
        for i in range(len(entity_embeds)):
            entity_embedding_set = preprocessed.entity_embeds[i]
            for j in range(len(mention.candidates)):
                c = mention.candidates[j].strip().lower()
                if options.use_entity_embeddings and c in entity_embedding_set:
                    if options.full_entity_embeddings:
                        if not options.unigram_features:
                            feature_vector.append(entity_embedding_set[c])
                        else:
                            feature_vector.append(scipy.sparse.coo_matrix([entity_embedding_set[c]]))
                    else:
                        c_emb = entity_embedding_set[c]
                        if valid_ctx_embedding:
                            cos_sim = (
                                np.dot(ctx_embedding, c_emb) /
                                (np.linalg.norm(ctx_embedding) * np.linalg.norm(c_emb))
                            )
                        else:
                            cos_sim = 0
                        if not options.unigram_features:
                            feature_vector.append([cos_sim])
                        else:
                            feature_vector.append(scipy.sparse.coo_matrix([cos_sim]))
                if c.strip().lower() == mention.CUI.strip().lower():
                    label = j
    for j in range(len(mention.candidates)):
        c = mention.candidates[j]
        if c.strip().lower() == mention.CUI.strip().lower():
            label = j

    if len(feature_vector) > 0:
        if options.unigram_features:
            feature_vector = scipy.sparse.hstack(feature_vector)
        else:
            feature_vector = np.concatenate(feature_vector, axis=0)
    else:
        feature_vector = None

    return (
        feature_vector,
        label
    )

def filterMentions(preprocessed, options):
    filtered_mentions, skipped = [], 0
    log.track(message='  >> Processed {0:,}/%s mentions' % '{0:,}'.format(len(preprocessed.mentions)), writeInterval=100)
    for m in preprocessed.mentions:
        valid = True
        (feature_vector, label) = prepSample(
            m,
            preprocessed,
            preprocessed.global_unigram_features,
            options
        )
        # check to ensure we have any features for this point
        if feature_vector is None:
            valid = False
            skipped += 1

        if valid:
            filtered_mentions.append(m)
        log.tick()
    log.flushTracker()

    return filtered_mentions, skipped

def runCrossfoldExperiment(preprocessed, preds_stream, options):
    cross_fold_metrics = []

    for i in range(len(preprocessed.splits)):
        log.writeln(('\n\n{0}\n  Starting fold %d/%d\n{0}\n'.format('#'*80)) % (i+1, len(preprocessed.splits)))

        (train_ids, dev_ids, test_ids) = preprocessed.splits[i]
        train, test = [], []
        for _id in train_ids:
            if _id in preprocessed.mentions_by_id:
                train.append(preprocessed.mentions_by_id[_id])
        for _id in dev_ids:
            if _id in preprocessed.mentions_by_id:
                if options.eval_on_dev:
                    test.append(preprocessed.mentions_by_id[_id])
                else:
                    train.append(preprocessed.mentions_by_id[_id])
        if not options.eval_on_dev:
            for _id in test_ids:
                if _id in preprocessed.mentions_by_id:
                    test.append(preprocessed.mentions_by_id[_id])

        if options.unigram_features:
            unigram_vocab = getTextVocabulary(train, preprocessed, options)
            unigram_vectorizer = CountVectorizer(vocabulary=unigram_vocab, binary=True)
        else:
            unigram_vectorizer = None

        training_features, training_labels = [], []
        for m in train:
            (feature_vector, label) = prepSample(
                m,
                preprocessed,
                preprocessed.per_fold_unigram_features[i],
                options
            )
            if feature_vector is None or label is None:
                continue
            training_features.append(feature_vector)
            training_labels.append(label)

        test_features, test_labels = [], []
        for m in test:
            (feature_vector, label) = prepSample(
                m,
                preprocessed,
                preprocessed.per_fold_unigram_features[i],
                options
            )
            if feature_vector is None or label is None:
                continue
            test_features.append(feature_vector)
            test_labels.append(label)

        log.writeln('Number of training samples: {0:,}'.format(len(training_labels)))
        log.writeln('Number of test samples: {0:,}\n'.format(len(test_labels)))

        if len(test_labels) == 0:
            log.writeln('[WARNING] Test ids list is empty due to rounding in cross-validation splits, skipping...')
            continue

        if len(set(training_labels)) == 1:
            log.writeln('[WARNING] Training samples for this subset have only one label class. Skipping...')
            return None

        if options.unigram_features:
            training_features = scipy.sparse.vstack(training_features)
            test_features = scipy.sparse.vstack(test_features)

        scaler = StandardScaler(with_mean=False)
        if options.normalize_features:
            training_features = scaler.fit_transform(training_features)
            test_features = scaler.transform(test_features)

        if options.classifier == Classifier.SVM:
            t = log.startTimer('Training SVM classifier...')
            classifier = sklearn.svm.SVC(
                kernel='linear',
                random_state=options.random_seed+i
            )
            classifier.fit(training_features, training_labels)
            log.stopTimer(t, message='Training complete in {0:.2f}s.\n')

            t = log.startTimer('Running trained SVM on test set...')
            predictions = classifier.predict(test_features)
            log.stopTimer(t, message='Complete in {0:.2f}s.\n')

        elif options.classifier == Classifier.KNN:
            t = log.startTimer('Training k-NN classifier...')
            classifier = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=5,
                #random_state=options.random_seed+i
            )
            classifier.fit(training_features, training_labels)
            log.stopTimer(t, message='Training complete in {0:.2f}s.\n')

            t = log.startTimer('Running trained k-NN on test set...')
            predictions = classifier.predict(test_features)
            log.stopTimer(t, message='Complete in {0:.2f}s.\n')

        elif options.classifier == Classifier.MLP:
            t = log.startTimer('Training MLP classifier...')
            classifier = sklearn.neural_network.multilayer_perceptron.MLPClassifier(
                max_iter=1000,
                random_state=options.random_seed+i
            )
            classifier.fit(training_features, training_labels)
            log.stopTimer(t, message='Training complete in {0:.2f}s.\n')

            t = log.startTimer('Running trained MLP on test set...')
            predictions = classifier.predict(test_features)
            log.stopTimer(t, message='Complete in {0:.2f}s.\n')

        metrics = SimpleNamespace()
        metrics.correct = 0
        metrics.total = 0

        for j in range(len(predictions)):
            if predictions[j] == test_labels[j]:
                metrics.correct += 1
            metrics.total += 1

            if preds_stream:
                preds_stream.write('Mention %d -- Pred: %d -> %s  Gold: %d -> %s\n' % (
                    test[j].ID,
                    predictions[j],
                    test[j].candidates[predictions[j]],
                    test_labels[j],
                    test[j].candidates[test_labels[j]]
                ))

        metrics.accuracy = float(metrics.correct)/metrics.total
        log.writeln('Fold accuracy: {0:.2f} ({1:,}/{2:,})'.format(metrics.accuracy, metrics.correct, metrics.total))

        cross_fold_metrics.append(metrics)

    overall_metrics = SimpleNamespace()
    overall_metrics.correct = 0
    overall_metrics.total = 0

    log.writeln('\n\n-- Cross-validation report --\n')
    for i in range(len(cross_fold_metrics)):
        m = cross_fold_metrics[i]
        overall_metrics.correct += m.correct
        overall_metrics.total += m.total
        log.writeln('  Fold %d -- Accuracy: %f (%d/%d)' % (i+1, m.accuracy, m.correct, m.total))

    overall_metrics.accuracy = np.mean([m.accuracy for m in cross_fold_metrics])
    log.writeln('\nOverall cross-validation accuracy: %f' % overall_metrics.accuracy)

    return overall_metrics

def experimentWrapper(mentions, entity_embeds, ctx_embeds, options, preds_stream):
    preprocessed = preprocessData(
        mentions,
        entity_embeds,
        ctx_embeds,
        options
    )

    log.writeln('Filtering mentions for these embeddings...')
    preprocessed.mentions, skipped = filterMentions(
        preprocessed,
        options
    )
    # re-calculate mentions_by_id to remove filtered sampled
    preprocessed.mentions_by_id = {
        m.ID : m
            for m in preprocessed.mentions
    }
    log.writeln('  Removed {0:,} mentions with no valid features'.format(skipped))
    log.writeln('Filtered dataset size: {0:,} mentions\n'.format(len(preprocessed.mentions)))

    results = runCrossfoldExperiment(
        preprocessed,
        preds_stream,
        options
    )

    return results

if __name__=='__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog MENTIONS [options] --entities=ENTITY_FILE --ctxs=CTX_FILE',
                description='Runs the LogLinearLinker model using the embeddings in ENTITY_FILE and CTX_FILE'
                            ' on the mentions in MENTIONS.')
        parser.add_option('--entities', dest='entity_embfs',
                help='comma-separated list of entity embedding files (required)')
        parser.add_option('--word-vocab', dest='word_vocabf',
                help='file listing words to load embeddings for (one per line); if unused, loads all embeddings')
        parser.add_option('--ctxs', dest='ctx_embf',
                help='context embedding file (required)')
        parser.add_option('--ctxs-format', dest='ctx_emb_fmt',
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text],
                default=pyemblib.Mode.Text,
                help='file format of embedding file (word2vec format)')
        parser.add_option('--input-predictions', dest='input_predsf',
                help='file with previously generated scores to include as features')
        parser.add_option('--predictions', dest='preds_file',
                help='file to write prediction details to')
        parser.add_option('--n-fold', dest='n_folds',
                type='int', default=10,
                help='number of folds for cross validation (default: %default)')
        parser.add_option('--dev-size', dest='dev_size',
                type='float', default=0.1,
                help='portion of cross-validation training data to hold back for development'
                     ' (default %default; must be >0 and <1)')
        parser.add_option('--cross-validation-splits', dest='cross_validation_file',
                help='path to save cross-validation splits to (generates multiple files; optional)')
        parser.add_option('--normalize-features', dest='normalize_features',
                action='store_true', default=False,
                help='use sklearn feature normalization (default off)')
        parser.add_option('--classifier', dest='classifier',
                type='choice', choices=Classifier.tolist(), default=Classifier.default(),
                help='classification algorithm to use')
        parser.add_option('--random-seed', dest='random_seed',
                type='int', default=-1,
                help='random seed for reproducibility (defaults to epoch time)')
        parser.add_option('-l', '--logfile', dest='logfile',
                help=str.format('name of file to write log contents to (empty for stdout)'),
                default=None)

        hyperparameters = optparse.OptionGroup(parser, 'Hyperparameter options')
        hyperparameters.add_option('--eval-on-dev', dest='eval_on_dev',
                action='store_true', default=False,
                help='evaluate on development data (for hyperparam tuning)')
        hyperparameters.add_option('--no-ctx-embeddings', dest='use_ctx_embeddings',
                action='store_false', default=True,
                help='dont\'t use context embeddings in features')
        hyperparameters.add_option('--no-entities', dest='use_entity_embeddings',
                action='store_false', default=True,
                help='don\'t use entity embeddings at all in features')
        hyperparameters.add_option('--full-entity-embeddings', dest='full_entity_embeddings',
                action='store_true', default=False,
                help='use full entity embeddings instead of cosine similarity to context')
        hyperparameters.add_option('--unigram-features', dest='unigram_features',
                action='store_true', default=False,
                help='use unigram features (indicators unless --tfidf is specified)')
        hyperparameters.add_option('--tfidf', dest='unigrams_as_tfidf',
                action='store_true', default=False,
                help='use TF-IDF values for unigram features (w/r/t input samples as'
                     ' documents; ignored if not using --unigram-features)')
        hyperparameters.add_option('--action-oracle', dest='action_oracle',
                action='store_true', default=False,
                help='use Action oracle')
        hyperparameters.add_option('--pre-embedded', dest='pre_embedded',
                action='store_true', default=False,
                help='mention file is pre-embedded (overrides --unigram-features)')

        (options, args) = parser.parse_args()

        if options.random_seed < 0:
            options.random_seed = int(time.time())

        if options.logfile and not options.preds_file:
            options.preds_file = '%s.predictions' % (os.path.splitext(options.logfile)[0])
        
        now_stamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        if options.logfile:
            options.logfile = '%s.%s' % (options.logfile, now_stamp)
        if options.preds_file:
            options.preds_file = '%s.%s' % (options.preds_file, now_stamp)

        if options.pre_embedded and options.unigram_features:
            log.writeln('[WARNING] Cannot use --unigram-features together with --pre-embedded')
            log.writeln('[WARNING] Disabling --unigram-features')
            options.unigram_features = False

        if options.use_entity_embeddings:
            options.entity_embfs = options.entity_embfs.split(',')
        else:
            options.entity_embfs = []

        def _bail(msg):
            import sys
            print(sys.argv)
            parser.print_help()
            print('\n' + msg)
            exit()

        if len(args) != 1:
            _bail('Must supply only MENTIONS')
        elif (options.use_entity_embeddings and len(options.entity_embfs) == 0):
            _bail('Must supply --entities')
        elif (options.use_ctx_embeddings and not options.ctx_embf):
            _bail('Must supply --ctxs')
        elif (options.dev_size <= 0 or options.dev_size >= 1):
            _bail('--dev-size must be between (0,1)')

        (mentionf,) = args
        return mentionf, options

    ## Getting configuration settings
    mentionf, options = _cli()
    log.start(logfile=options.logfile, stdout_also=True)
    entity_settings = [
            ('Entities %d' % i, options.entity_embfs[i])
                for i in range(len(options.entity_embfs))
        ]
    log.writeConfig([
        ('Mention file', mentionf),
        ('Entity embedding settings', entity_settings),
        ('Word/ctx embeddings', options.ctx_embf),
        ('Word vocabulary (unused if empty)', options.word_vocabf),
        ('Writing predictions to', options.preds_file),
        ('Using feature normalization', options.normalize_features),
        ('Classification algorithm', options.classifier),
        ('Training settings', [
            ('Cross validation splits file', options.cross_validation_file),
            ('Number of folds', options.n_folds),
            ('Fraction of training used for dev', options.dev_size),
            ('Random seed', options.random_seed),
        ]),
        ('Hyperparameter settings', [
            ('Evaluating on development data', options.eval_on_dev),
            ('Using entity embeddings at all', options.use_entity_embeddings),
            ('Using full entity embeddings instead of cos sim', options.full_entity_embeddings),
            ('Using context embeddings', options.use_ctx_embeddings),
            ('Including unigram features', options.unigram_features),
            ('Using TF-IDF values for unigram features', options.unigrams_as_tfidf if options.unigram_features else 'N/A'),
            ('Using Action oracle', options.action_oracle),
            ('Input predictions file', options.input_predsf),
            ('Pre-embedded mentions', options.pre_embedded),
        ]),
    ], title="Entity linking (disambiguation) experiment using scikit-learn baseline algorithms")

    ## Data loading/setup
    entity_embeds = []
    for i in range(len(options.entity_embfs)):
        f = options.entity_embfs[i]
        t_sub = log.startTimer('Reading set %d of entity embeddings from %s...' % (i+1, f))
        entity_embeds.append(pyemblib.read(f, lower_keys=True))
        log.stopTimer(t_sub, message='Read %s embeddings ({0:.2f}s)\n' % ('{0:,}'.format(len(entity_embeds[-1]))))
    
    if options.word_vocabf:
        t_sub = log.startTimer('Reading word/context vocabulary from %s...' % options.word_vocabf)
        word_vocab = readVocab(options.word_vocabf)
        log.stopTimer(t_sub, message='Read %s words ({0:.2f}s)\n' % ('{0:,}'.format(len(word_vocab))))
    else:
        word_vocab = None

    if options.use_ctx_embeddings:
        t_sub = log.startTimer('Reading context embeddings from %s...' % options.ctx_embf)
        ctx_embeds = pyemblib.read(options.ctx_embf, replace_errors=True, filter_to=word_vocab,
            lower_keys=True, mode=options.ctx_emb_fmt)
        log.stopTimer(t_sub, message='Read %s embeddings ({0:.2f}s)\n' % ('{0:,}'.format(len(ctx_embeds))))
    else:
        ctx_embeds = None

    t_sub = log.startTimer('Reading mentions from %s...' % mentionf)
    mentions = mention_file.read(mentionf)
    log.stopTimer(t_sub, message='Read %s mentions ({0:.2f}s)\n' % ('{0:,}'.format(len(mentions))))
    
    if options.preds_file:
        preds_stream = open(options.preds_file, 'w')
    else:
        preds_stream = None

    results = experimentWrapper(
        mentions,
        entity_embeds,
        ctx_embeds,
        options,
        preds_stream
    )

    if options.preds_file:
        preds_stream.close()

    log.stop()
