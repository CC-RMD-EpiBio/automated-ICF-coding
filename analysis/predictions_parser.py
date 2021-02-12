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
Parsing methods for predictions files (with scores)
'''

import codecs

def parsePredictions(f, get_candidate=False, no_scores=False):
    predictions = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            first_word = line.split()[0].strip()
            if first_word == 'Mention':
                if no_scores:
                    left, right = line.split('--', 1)
                    scores = None
                else:
                    left, right = line.split('[', 1)
                    scores, right = right.split(']', 1)
                    scores = [float(f) for f in scores.split()]
                mention_id = int(left.split()[1])

                pred_ix = int(
                    right.split('->')[0].split(':')[1]
                )
                gold_ix = int(
                    right.split('->')[1].split(':')[-1]
                )

                pred_cand = right.split('->')[1] \
                    .split('Gold')[0] \
                    .strip()
                gold_cand = right.split('->')[2] \
                    .strip()

                info = [
                    scores,
                    pred_ix,
                    gold_ix,
                    pred_ix == gold_ix
                ]
                if get_candidate:
                    info.extend([
                        pred_cand,
                        gold_cand
                    ])
                predictions[mention_id] = tuple(info)
    return predictions

def writePredictionsToStream(stream, mention, scores, predicted_ix, correct_ix,
        correct_candidate, predicted_candidate=None):
    if predicted_candidate is None:
        predicted_candidate = mention.candidates[predicted_ix]
    stream.write('Mention %d -- Scores: [ %s ]  Pred: %d -> %s  Gold: %d -> %s\n' % (
        mention.ID,
        ' '.join([str(s) for s in scores]),
        predicted_ix,
        predicted_candidate,
        correct_ix,
        correct_candidate
    ))
