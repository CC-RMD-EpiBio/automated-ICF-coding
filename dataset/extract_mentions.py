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

import numpy as np
import configparser
from . import mention_file
from . import getAllMentions
from .tokenizer import Tokenizer
from hedgepig_logger import log

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog [options]',
                description='Generates common-format file giving mentions (with contextual information) for full dataset')
        parser.add_option('-c', '--config', dest='configf',
            default='config.ini',
            help='(required) path to dataset configuration file (default: %default)')
        parser.add_option('--dataset', dest='dataset',
            help='(required) dataset configuration to use (section header in --config file)')
        parser.add_option('-o', '--output', dest='outputf',
            help='(required) output file path')
        parser.add_option('-t', '--tokenizer', dest='tokenizer',
            type='choice', choices=Tokenizer.choices(), default=Tokenizer.default(),
            help='tokenizer to use for dataset processing')
        parser.add_option('--bert-vocab-file', dest='bert_vocab_file',
            help='vocab file to use for BERT tokenization')
        (options, args) = parser.parse_args()
        if not options.configf:
            parser.print_help()
            parser.error('Must provide --config')
        if not options.dataset:
            parser.print_help()
            parser.error('Must provide --dataset')
        if not options.outputf:
            parser.print_help()
            parser.error('Must provide --output')

        # add automatically-constructed paths
        options.logfile = '{0}.log'.format(options.outputf)
        options.mention_map_file = '{0}.mention_map'.format(options.outputf)

        return options

    options = _cli()

    config = configparser.ConfigParser()
    config.read(options.configf)
    config = config[options.dataset]

    log.start(logfile=options.logfile)
    settings = [
        ('Configuration file', options.configf),
        ('Dataset to extract features for', options.dataset),
        ('Tokenizer', options.tokenizer),
        ('BERT vocab file', (
            'N/A' if options.tokenizer != Tokenizer.BERT else options.bert_vocab_file
        )),
        ('Extraction mode', config['ExtractionMode']),
        ('Annotation directories', config['DataDirectories']),
    ]
    if config['ExtractionMode'] == 'csv':
        settings.extend([
            ('Plaintext directory', config['PlaintextDirectory']),
            ('CSV file ID pattern', config['CSVIdentifierPattern']),
            ('Plaintext file render pattern', config['PlaintextIdentifierPattern'])
        ])
    settings.extend([
        ('Output mentions file', options.outputf),
        ('Mention map file (automatic)', options.mention_map_file),
    ])
    log.writeConfig(settings, title='Mention extraction for action classification')

    t_sub = log.startTimer('Generating %s features.' % options.dataset)
    mentions, mention_map = getAllMentions(config, options,
        tokenizer=options.tokenizer, bert_vocab_file=options.bert_vocab_file,
        log=log)
    log.stopTimer(t_sub, 'Extracted {0:,} samples.'.format(len(mentions)))

    log.writeln('Writing mention map information to %s...' % options.mention_map_file)
    with open(options.mention_map_file, 'w') as stream:
        for (mention_ID, mention_info) in mention_map.items():
            stream.write('%d\t%s\n' % (mention_ID, mention_info))
    log.writeln('Wrote info for {0:,} mentions.\n'.format(len(mention_map)))

    t_sub = log.startTimer('Writing samples to %s...' % options.outputf, newline=False)
    mention_file.write(mentions, options.outputf)
    log.stopTimer(t_sub, message='Done ({0:.2f}s).')

    log.stop()
