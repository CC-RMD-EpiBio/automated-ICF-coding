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
import nltk.corpus
from nltk.stem import *
import time
from datetime import datetime
from hedgepig_logger import log
from dataset import mention_file
from experiments import cross_validation
from experiments.util import *
from dataset.tokenizer import Tokenizer


tokenizer = Tokenizer(Tokenizer.SpaCy)
stemmer = PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))


def preprocessData(mentions, options):
    preprocessed = SimpleNamespace()

    preprocessed.mentions = mentions

    preprocessed.mentions_by_id = {
        m.ID : m
            for m in mentions
    }
    preprocessed.labels_by_id = {
        m.ID : m.CUI.strip().lower()
            for m in mentions
    }

    return preprocessed

def readCodeDefinitions(f, main_only=False):
    definitions = {}
    with open(f, 'r') as stream:
        for line in stream:
            (code, _type, defn) = [s.strip() for s in line.split('\t')]
            if main_only and _type != 'main':
                continue
            if not code in definitions:
                definitions[code] = set()
            for tok in tokenizer.tokenize(defn):
                if not tok.lower() in stopwords:
                    definitions[code].add(stemmer.stem(tok))
    return definitions

def getMostSimilar(mention, definitions, default):
    mention_toks = set()
    for string in [mention.left_context, mention.right_context, mention.mention_text]:
        toks = string.split()
        for tok in toks:
            if not tok.lower() in stopwords:
                mention_toks.add(stemmer.stem(tok))

    similarities = {
        code: calculateSimilarity(mention_toks, defn_toks)
            for (code, defn_toks)
            in definitions.items()
    }

    default_ix = mention.candidates.index(default)
    ordered_similarities = [
        similarities.get(c, 0.)
            for c in mention.candidates
    ]

    if max(ordered_similarities) == 0:
        return default_ix
    else:
        return np.argmax(ordered_similarities)

def calculateSimilarity(mention_toks, defn_toks):
    numerator = len(mention_toks.intersection(defn_toks))
    denominator = (
        np.sqrt(len(mention_toks))
        * np.sqrt(len(defn_toks))
    )
    return (numerator/denominator)

def runLeskExperiment(preprocessed, definitions, preds_stream, options):
    log.writeln(('\n\n{0}\n  Starting experiment\n{0}\n'.format('#'*80)))

    test_labels, predictions = [], []
    for m in preprocessed.mentions:
        test_labels.append(m.CUI.lower())
        predictions.append(getMostSimilar(m, definitions, default='d450'))

    metrics = SimpleNamespace()
    metrics.correct = 0
    metrics.total = 0

    for j in range(len(predictions)):
        m = preprocessed.mentions[j]

        if m.candidates[predictions[j]] == test_labels[j]:
            metrics.correct += 1
        metrics.total += 1

        if preds_stream:
            preds_stream.write('Mention %d -- Pred: %d -> %s  Gold: %d -> %s\n' % (
                preprocessed.mentions[j].ID,
                predictions[j],
                m.candidates[predictions[j]],
                m.candidates.index(test_labels[j]),
                test_labels[j]
            ))

    metrics.accuracy = float(metrics.correct)/metrics.total
    log.writeln('Accuracy: {0:.2f} ({1:,}/{2:,})'.format(metrics.accuracy, metrics.correct, metrics.total))

def experimentWrapper(mentions, definitions, options, preds_stream):
    preprocessed = preprocessData(
        mentions,
        options
    )

    results = runLeskExperiment(
        preprocessed,
        definitions,
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
        parser.add_option('--predictions', dest='preds_file',
            help='file to write prediction details to')
        parser.add_option('--definitions', dest='definitions_file',
            help='(required) file with definitions for labels')
        parser.add_option('--main-only', dest='main_only',
            action='store_true', default=False,
            help='use main definitions only')
        parser.add_option('-l', '--logfile', dest='logfile',
            help=str.format('name of file to write log contents to (empty for stdout)'),
            default=None)

        (options, args) = parser.parse_args()

        if options.logfile and not options.preds_file:
            options.preds_file = '%s.predictions' % (os.path.splitext(options.logfile)[0])
        
        now_stamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        if options.logfile:
            options.logfile = '%s.%s' % (options.logfile, now_stamp)
        if options.preds_file:
            options.preds_file = '%s.%s' % (options.preds_file, now_stamp)

        if len(args) != 1:
            parser.print_help()
            parser.error('Must supply only MENTIONS')
        if not options.definitions_file:
            parser.print_help()
            parser.error('Must supply --definitions')

        (mentionf,) = args
        return mentionf, options

    ## Getting configuration settings
    mentionf, options = _cli()
    log.start(logfile=options.logfile)
    log.writeConfig([
        ('Mention file', mentionf),
        ('Entity definitions file', options.definitions_file),
        ('Restricting to main definitions only', options.main_only),
    ], title="Adapted Lesk similarity baseline")

    t_sub = log.startTimer('Reading mentions from %s...' % mentionf)
    mentions = mention_file.read(mentionf)
    log.stopTimer(t_sub, message='Read %s mentions ({0:.2f}s)\n' % ('{0:,}'.format(len(mentions))))

    log.writeln('Reading definitions from %s...' % options.definitions_file)
    definitions = readCodeDefinitions(options.definitions_file, options.main_only)
    log.writeln('Read definitions for {0:,} codes.\n'.format(len(definitions)))
    
    if options.preds_file:
        preds_stream = open(options.preds_file, 'w')
    else:
        preds_stream = None

    results = experimentWrapper(
        mentions,
        definitions,
        options,
        preds_stream
    )

    if options.preds_file:
        preds_stream.close()

    log.stop()
