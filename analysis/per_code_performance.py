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
Given a set of predictions on BTRIS Mobility data, calculates accuracy,
precision, recall, and F1 for each code.

Also calculates a confusion matrix for code classifications.
'''

import csv
import codecs
import numpy as np
from types import SimpleNamespace
from . import predictions_parser
from . import util
from dataset import mention_file
from experiments import cross_validation
from hedgepig_logger import log

def readCUICodeMap(f):
    _map = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (code, cui) = [s.strip() for s in line.split('\t')]
            _map[cui.lower()] = code
    return _map

def compileEvaluationSet(splits, eval_on_dev):
    eval_set = set()
    for (train, dev, test) in splits:
        these_evals = set(
            dev if eval_on_dev else test
        )
        eval_set = eval_set.union(these_evals)
    return eval_set

def remapIxesToCodes(predictions, mentions):
    mentions_by_id = {
        m.ID : m
            for m in mentions
    }

    remapped_predictions = {}
    for (mention_id, (scores, pred_ix, gold_ix, correct)) in predictions.items():
        candidates = mentions_by_id[mention_id].candidates
        remapped_predictions[mention_id] = (
            scores,
            candidates[pred_ix].lower(),
            candidates[gold_ix].lower(),
            correct
        )

    return remapped_predictions

def sortCodesByGoldFrequency(mentions, eval_set):
    freqs = {}
    for m in mentions:
        if m.ID in eval_set:
            freqs[m.CUI] = freqs.get(m.CUI, 0) + 1

    freqs = list(freqs.items())
    freqs.sort(key=lambda k:k[1], reverse=True)
    
    return freqs

def calculateMetricsPerCode(predictions, mentions_by_ID, eval_set):
    preds_keys = set(predictions.keys())
    if len(preds_keys - eval_set) > 0:
        log.writeln('[WARNING] Predictions file includes outputs for {0} samples not included in reference evaluation set\n'.format(len(preds_keys - eval_set)))
        input('[Enter] to continue')

    def initMetric():
        obj = SimpleNamespace()
        obj.tp = 0
        obj.fp = 0
        obj.fn = 0
        return obj

    metrics = {}
    for mention_ID in eval_set:
        results = predictions.get(mention_ID, None)
        if results is None:
            mention = mentions_by_ID[mention_ID]
            gold_ix = mention.candidates.index(mention.CUI)
            if not gold_ix in metrics:
                metrics[gold_ix] = initMetric()
            metrics[gold_ix].fn += 1
        else:
            (scores, pred_ix, gold_ix, correct) = results
            if not pred_ix in metrics:
                metrics[pred_ix] = initMetric()
            if not gold_ix in metrics:
                metrics[gold_ix] = initMetric()

            if correct:
                metrics[gold_ix].tp += 1
            else:
                metrics[pred_ix].fp += 1
                metrics[gold_ix].fn += 1

    for (ix, code_metrics) in metrics.items():
        if code_metrics.tp + code_metrics.fp > 0:
            code_metrics.precision = (
                float(code_metrics.tp)/
                (code_metrics.tp + code_metrics.fp)
            )
        else:
            code_metrics.precision = 0
        if code_metrics.tp + code_metrics.fn > 0:
            code_metrics.recall = (
                float(code_metrics.tp)/
                (code_metrics.tp + code_metrics.fn)
            )
        else:
            code_metrics.recall = 0
        if code_metrics.precision + code_metrics.recall > 0:
            code_metrics.f1 = (
                (2 * code_metrics.precision * code_metrics.recall)/
                (code_metrics.precision + code_metrics.recall)
            )
        else:
            code_metrics.f1 = 0

    return metrics

def calculateConfusionMatrix(predictions):
    matrix = {}
    for (_, pred_ix, gold_ix, _) in predictions.values():
        if not gold_ix in matrix:
            matrix[gold_ix] = {}
        matrix[gold_ix][pred_ix] = matrix[gold_ix].get(pred_ix, 0) + 1
    return matrix

def printConfusionMatrix(confusion_matrix, ordering, remap=None):
    if remap is None:
        remap = lambda k:k

    column_width = max([len(remap(item)) for item in ordering])
    # headers
    log.write(' '*(column_width+1))
    log.write('|')
    for item in ordering:
        log.write((' %{0}s'.format(column_width)) % remap(item))
    log.write('\n')
    # rows
    for gold_item in ordering:
        log.write(('%{0}s '.format(column_width)) % remap(gold_item))
        log.write('|')
        for pred_item in ordering:
            log.write((' %{0}d'.format(column_width)) % confusion_matrix.get(gold_item, {}).get(pred_item, 0))
        log.write('\n')

def writeConfusionMatrixToCSV(confusion_matrix, ordering, outf, remap=None):
    if remap is None:
        remap = lambda k:k

    with open(outf, 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Gold', *ordering])
        writer.writeheader()
        for gold_item in ordering:
            row = {'Gold': gold_item}
            for pred_item in ordering:
                row[pred_item] = confusion_matrix.get(gold_item, {}).get(pred_item, 0)
            writer.writerow(row)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog MENTION_FILE PREDS_F')
        parser.add_option('--remap', dest='keymapf',
            help='file mapping CUIs to ICF codes')
        parser.add_option('--no-scores', dest='no_scores',
            action='store_true', default=False,
            help='predictions file has no scores in it')
        parser.add_option('--cross-validation-splits', dest='splitsf',
            help='(required) base splits file for cross validation')
        parser.add_option('--dev', dest='dev',
            action='store_true', default=False,
            help='evaluate on development splits (rather than test)')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        (options, args) = parser.parse_args()
        if not options.splitsf:
            parser.print_help()
            parser.error('Must provide --splits')
        if len(args) != 2:
            parser.print_help()
            exit()
        return args, options
    (mentionf, predsf), options = _cli()
    log.start(logfile=options.logfile)

    log.writeConfig([
        ('Mention file', mentionf),
        ('Key remapping file', options.keymapf),
        ('Predictions file', predsf),
        ('No scores in predictions', options.no_scores),
        ('Cross-validation splits file', options.splitsf),
        ('Evaluating on development data', options.dev),
    ], 'BTRIS Mobility code-level predictions analysis')

    log.writeln('Reading mentions from %s...' % mentionf)
    mentions = mention_file.read(mentionf)
    log.writeln('Read {0:,} mentions.\n'.format(len(mentions)))

    log.writeln('Reading splits from %s...' % options.splitsf)
    splits = cross_validation.readSplits(options.splitsf)
    log.writeln('Read {0:,} splits.\n'.format(len(splits)))

    log.writeln('Compiling evaluation set...')
    eval_set = compileEvaluationSet(splits, options.dev)
    log.writeln('Evaluating on {0:,} samples.\n'.format(len(eval_set)))

    log.writeln('Parsing predictions from %s...' % predsf)
    predictions = predictions_parser.parsePredictions(predsf, no_scores=options.no_scores)
    log.writeln('Read {0:,} predictions.\n'.format(len(predictions)))

    if options.keymapf:
        log.writeln('Reading CUI->code mapping from %s...' % options.keymapf)
        key_remap = readCUICodeMap(options.keymapf)
        log.writeln('Read {0:,} remappings.\n'.format(len(key_remap)))
        remap = lambda k: key_remap.get(k, k)
    else:
        remap = lambda k: k

    log.writeln('Remapping prediction indices to codes...')
    predictions = remapIxesToCodes(predictions, mentions)
    log.writeln('Done.\n')

    log.writeln('Sorting codes...')
    sorted_codes = sortCodesByGoldFrequency(mentions, eval_set)
    log.writeln('Sorted:')
    for (code, freq) in sorted_codes:
        log.writeln('  {0} --> {1:,}'.format(remap(code), freq))

    log.writeln('\nCalculating performance metrics per code...')
    metrics = calculateMetricsPerCode(predictions, {m.ID:m for m in mentions}, eval_set)
    log.writeln('Metrics table:')
    for (code, _) in sorted_codes:
        if code in metrics:
            m = metrics[code]
        else:
            m = SimpleNamespace(precision=0, tp=0, fp=0, recall=0, fn=0, f1=0)
        log.writeln('  {0:8s} -->  Pr: {1:.4f} ({2:4d}/{3:4d})  Rec: {4:.4f} ({5:4d}/{6:4d})  F1: {7:.4f}'.format(
            remap(code), m.precision, m.tp, m.tp+m.fp, m.recall, m.tp, m.tp+m.fn, m.f1
        ))

    log.writeln('\n>> Overall statistics <<')
    log.writeln('Micro accuracy: {0:.4f} ({1:,}/{2:,})'.format(
        float(sum([m.tp for m in metrics.values()]))/len(eval_set),
        sum([m.tp for m in metrics.values()]), len(eval_set)
    ))
    log.writeln('Macro F1: {0:.4f}'.format(
        np.mean([m.f1 for m in metrics.values()])
    ))

    log.writeln('\n>> Without no_code label <<')
    corr_wo_no_code = sum([m.tp for (k,m) in metrics.items() if k != 'no_code'])
    ttl_wo_no_code = sum([(m.tp+m.fn) for (k,m) in metrics.items() if k != 'no_code'])
    log.writeln('Micro accuracy: {0:.4f} ({1:,}/{2:,})'.format(
        float(corr_wo_no_code/ttl_wo_no_code), corr_wo_no_code, ttl_wo_no_code
    ))
    log.writeln('Macro F1: {0:.4f}'.format(
        np.mean([m.f1 for (k,m) in metrics.items() if k != 'no_code'])
    ))

    log.writeln('\nCalculating confusion matrix...')
    confusion_matrix = calculateConfusionMatrix(predictions)
    log.writeln('Confusion matrix:')
    printConfusionMatrix(confusion_matrix, [code for (code, _) in sorted_codes], remap)
    writeConfusionMatrixToCSV(confusion_matrix, [code for (code, _) in sorted_codes], '/home/griffisd/tmp_confusion_matrix.csv', remap=remap)
