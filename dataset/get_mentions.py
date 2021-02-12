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
import codecs
import optparse
import configparser
from . import mention_file
from .tokenizer import Tokenizer
from . import mobility_framework
from hedgepig_logger import log

CANDIDATES = [
    'd410',
    'd415',
    'd420',
    'd430',
    'd435',
    'd440',
    'd445',
    'd450',
    'd455',
    'd460',
    'd470',
    'd475',
    'no_code'
]

def convertLinkedMention(action, tokenizer):
    left_action_offset = action.start - action.mobility.start
    right_action_offset = action.end - action.mobility.start

    left_context = action.mobility.text[:left_action_offset].replace('\t', ' ').replace('\n', ' ')
    right_context = action.mobility.text[right_action_offset:].replace('\t', ' ').replace('\n', ' ')

    left_tokens = tokenizer.tokenize(left_context)
    right_tokens = tokenizer.tokenize(right_context)

    mention_tokens = tokenizer.tokenize(action.text)

    if action.code is None:
        code = 'no_code'
    else:
        code = action.code
    if not code.lower() in set(CANDIDATES):
        print('WHOA NELLY!  Action code "%s" is unmapped' % code)
        input()

    return mention_file.Mention(
        mention_text=' '.join(mention_tokens),
        left_context=' '.join(left_tokens),
        right_context=' '.join(right_tokens),
        candidates=CANDIDATES,
        CUI=code.lower()
    )

def getAllMentions(config, options, tokenizer=None, bert_vocab_file=None, log=log):
    if tokenizer is None:
        tokenizer = config['Tokenizer']
    tokenizer = Tokenizer(tokenizer, bert_vocab_file=bert_vocab_file)

    if config['ExtractionMode'] == 'csv':
        (
            mobilities,
            actions,
            assistances,
            quantifications
        ) = mobility_framework.csv_reader.extractAllEntities(
            config['DataDirectories'].split(','),
            config['PlaintextDirectory'],
            config['CSVIdentifierPattern'],
            config['PlaintextIdentifierPattern'],
            log=log,
            by_document=False
        )
    else:
        (
            mobilities,
            actions,
            assistances,
            quantifications
        ) = mobility_framework.xml_reader.extractAllEntities(
            config['DataDirectories'].split(','),
            log=log
        )

    mentions, mention_map = [], {}

    mobility_framework.entity_crosslinker.crosslinkEntities(
        mobilities, actions, assistances, quantifications,
        log=log
    )

    cur_ID = 0
    for action in actions:
        mention = convertLinkedMention(
            action,
            tokenizer
        )
        mention.ID = cur_ID
        mention_map[mention.ID] = action.file_ID
        mentions.append(mention)
        cur_ID += 1

    return mentions, mention_map
