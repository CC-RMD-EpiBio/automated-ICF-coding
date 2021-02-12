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

import os
import glob
import numpy as np
from hedgepig_logger import log
from dataset import mention_file
from analysis import predictions_parser

def readMentionIDsFromBERTFile(f):
    m_IDs = []
    with open(f, 'r') as stream:
        # skip the header
        stream.readline()
        # for the rest of the file, the mention ID is the first column
        for line in stream:
            (m_ID, m_text, m_lbl) = [s.strip() for s in line.split('\t')]
            m_IDs.append(int(m_ID))
    return m_IDs

def readScoresFromBERTOutput(f):
    scores = []
    with open(f, 'r') as stream:
        for line in stream:
            scores.append([float(f) for f in line.split('\t')])
    return scores

def reformatScoresToPredictionsFile(ordered_m_IDs, ordered_scores, mentions_by_ID, stream):
    for i in range(len(ordered_m_IDs)):
        m_ID = ordered_m_IDs[i]
        scores = ordered_scores[i]
        m = mentions_by_ID[m_ID]

        predictions_parser.writePredictionsToStream(
            stream,
            m,
            scores,
            np.argmax(scores),
            m.candidates.index(m.CUI),
            m.CUI
        )

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-m', '--mentions', dest='mentions_f',
            help='(REQUIRED) mentions file to map back to')
        parser.add_option('--bert-dir', dest='bert_dir',
            help='(REQUIRED) directory containing BERT baseline files/results')
        parser.add_option('--model', dest='model',
            help='(REQUIRED) model setting to get results for')
        (options, args) = parser.parse_args()

        if not options.mentions_f:
            parser.error('Must supply --mentions')
        elif not options.bert_dir:
            parser.error('Must supply --bert-dir')
        elif not options.model:
            parser.error('Must supply --model')

        options.output_f = os.path.join(options.bert_dir, '%s.compiled_output.predictions' % options.model)
        options.logfile = '%s.log' % options.output_f

        return options
    options = _cli()
    log.start(options.logfile)

    log.writeConfig([
        ('Mentions file', options.mentions_f),
        ('BERT baseline root directory', options.bert_dir),
        ('Model configuration', options.model),
        ('Output file', options.output_f),
    ], 'BERT baseline results compilation')

    log.writeln('Reading mentions from %s...' % options.mentions_f)
    mentions = mention_file.read(options.mentions_f)
    mentions_by_ID = {
        m.ID: m
            for m in mentions
    }
    log.writeln('Read {0:,} mentions.\n'.format(len(mentions)))

    fold_dirs = glob.glob(os.path.join(options.bert_dir, 'fold-*'))
    log.writeln('Found {0} folds in {1}.\n'.format(len(fold_dirs), options.bert_dir))

    with open(options.output_f, 'w') as stream:
        fold_dirs = sorted(fold_dirs)
        for i in range(len(fold_dirs)):
            log.writeln('Checking fold {0}/{1}'.format(i+1, len(fold_dirs)))
            log.indent()

            test_f = os.path.join(fold_dirs[i], 'test.tsv')
            preds_f = os.path.join(fold_dirs[i], options.model, 'test_results.tsv')

            if not os.path.exists(preds_f):
                log.writeln('Found no predictions, skipping')
            else:
                ordered_m_IDs = readMentionIDsFromBERTFile(test_f)
                ordered_scores = readScoresFromBERTOutput(preds_f)

                log.writeln('Found {0:,} test IDs'.format(len(ordered_m_IDs)))
                log.writeln('Found scores for {0:,} samples'.format(len(ordered_scores)))

                reformatScoresToPredictionsFile(
                    ordered_m_IDs,
                    ordered_scores,
                    mentions_by_ID,
                    stream
                )

                log.writeln('Wrote {0:,} reformatted predictions.\n'.format(len(ordered_m_IDs)))
                
            log.unindent()
            log.writeln()

    log.writeln()
    log.writeln('Wrote reformatted predictions to:%s\n' % options.output_f)

    log.stop()
