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
Shared functions for experimental settings
'''

import numpy as np
import codecs
from dataset import mention_file

class Indexer:
    def __init__(self, vocab):
        self._vocab = vocab
        self._indices = {vocab[i]:i for i in range(len(vocab))}

    def indexOf(self, key):
        return self._indices.get(key, -1)

    def __getitem__(self, index):
        return self._vocab[index]


def meanEmbeddings(strings, ctx_embeds):
    word_embs, valid_ctx_embedding = [], False
    for context_string in strings:
        words = [s.strip() for s in context_string.lower().split()]
        for w in words:
            if w in ctx_embeds:
                word_embs.append(ctx_embeds[w])
    (ctx_embedding, valid_ctx_embedding) = meanEmbeddingVectors(word_embs, ctx_embeds.size)
    return ctx_embedding, valid_ctx_embedding

def meanEmbeddingVectors(vectors, dim):
    if len(vectors) > 0:
        mean_vector = np.mean(vectors, axis=0)
        valid = True
    else:
        mean_vector = np.zeros(dim)
        valid = False
    return mean_vector, valid

def readVocab(f):
    vocab = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            vocab.add(line.strip())
    return vocab
        
def readPreferredStrings(f):
    pref_strings = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (entity_id, string) = [s.strip() for s in line.split('\t')]
            pref_strings[entity_id.lower()] = string
    return pref_strings

def readSubsetMap(f):
    r'''Takes as input a tab-separated file of format
      <mention ID> \t <subset>
    and returns a dictionary mapping
      <subset> : [ <mention ID list> ]
    '''
    _map = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (mention_id, subset) = [s.strip() for s in line.split('\t')]
            if not subset in _map:
                _map[subset] = []
            _map[subset].append(int(mention_id))
    return _map

def finalPerSubsetReport(log, per_subset_results, subsets):
    log.writeln('\n\n{0}\n  Final report\n{0}'.format('='*30))
    micro_correct, micro_total, subsets_counted = 0, 0, 0
    for subset in subsets:
        if subset in per_subset_results:
            subsets_counted += 1
            m = per_subset_results[subset]
            micro_correct += m.correct
            micro_total += m.total
            log.writeln('  %s --> %f (%d/%d)' % (subset, m.accuracy, m.correct, m.total))

    log.writeln('\nOverall micro accuracy: %f (%d/%d)' % (float(micro_correct)/micro_total, micro_correct, micro_total))
    log.writeln('Overall macro accuracy: %f' % np.mean([m.accuracy for m in per_subset_results.values()]))
    log.writeln('Subsets included in overall analysis: %d/%d' % (subsets_counted, len(subsets)))

def writeDetailedOutcome(stream, mention, probs, batch_entity_masks,
        preferred_strings, correct_candidate, pred_ix, i, fold=None):
    if type(mention) is mention_file.Mention:
        stream.write(
            '\n-----------------------------------------------------\n'
            'Mention %d %s\n'
            '  Left context: %s\n'
            '  Mention text: %s\n'
            '  Right context: %s\n'
            '  Candidates: [ %s ]\n'
            '  Correct answer: %s\n'
            '\nPredictions\n' % (
                mention.ID,
                ( (' (Fold %d)' % i) if not fold is None else ''),
                mention.left_context,
                mention.mention_text,
                mention.right_context,
                ', '.join(mention.candidates),
                mention.CUI
            )
        )
    elif type(mention) is mention_file.EmbeddedMention:
        stream.write(
            '\n-----------------------------------------------------\n'
            'Mention %d %s - pre-embedded\n'
            '  Candidates: [ %s ]\n'
            '  Correct answer: %s\n'
            '\nPredictions\n' % (
                mention.ID,
                ( (' (Fold %d)' % i) if not fold is None else ''),
                ', '.join(mention.candidates),
                mention.CUI
            )
        )
    for j in range(len(probs[i])):
        if batch_entity_masks[i][j] == 1:
            entity_str = mention.candidates[j]
            stream.write('  %s --> %f   Gold: %s  Pred: %s\n' % (
                entity_str,
                probs[i][j],
                ('X' if entity_str.lower() == correct_candidate.lower() else ' '),
                ('X' if pred_ix == j else ' ')
            ))
