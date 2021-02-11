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
