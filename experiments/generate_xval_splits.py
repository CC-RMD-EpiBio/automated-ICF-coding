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

import time
from dataset import mention_file
from dataset import mention_map_lib
from . import cross_validation
from hedgepig_logger import log

def readFilterDocIDSet(f):
    doc_IDs = set()
    with open(f, 'r') as stream:
        for line in stream:
            doc_IDs.add(line.strip())
    return doc_IDs

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-m', '--mentions', dest='mentions_f',
            help='(REQUIRED) mentions file to generate xval splits for')
        parser.add_option('--mention-map', dest='mention_map_f',
            help='file mapping mention IDs to document IDs')
        parser.add_option('-k', '--num-folds', dest='num_folds',
            type='int', default=10,
            help='number of folds for cross validation (default %default)')
        parser.add_option('--dev-size', dest='dev_size',
            type='float', default=0.1,
            help='size of dataset to reserve for dev data (default %default)')
        parser.add_option('--filter-doc-IDs', dest='filter_doc_ID_f',
            help='file listing document IDs to filter to'
                 ' (requires --mention-map)')
        parser.add_option('--random-seed', dest='random_seed',
            type='int', default=-1,
            help='random seed for splits generation (defaults to epoch time)')
        parser.add_option('-o', '--output', dest='output_f',
            help='(REQUIRED) base path to write splits files to')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        (options, args) = parser.parse_args()

        if options.random_seed < 0:
            options.random_seed = int(time.time())

        if not options.mentions_f:
            parser.error('Must supply --mentions')
        elif not options.output_f:
            parser.error('Must supply --output')
        elif options.filter_doc_ID_f and not options.mention_map_f:
            parser.error('Must supply --mention-map if using --filter-doc-IDs')

        return options

    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Mentions file', options.mentions_f),
        ('Mention map file', options.mention_map_f),
        ('Number of folds', options.num_folds),
        ('Dev set size', options.dev_size),
        ('Document ID filter list', options.filter_doc_ID_f),
        ('Random seed', options.random_seed),
        ('Output file', options.output_f),
    ], 'Cross-validation splits generation')

    log.writeln('Loading mentions from %s...' % options.mentions_f)
    mentions = mention_file.read(options.mentions_f)
    log.writeln('Read {0:,} mentions.\n'.format(len(mentions)))

    if options.filter_doc_ID_f:
        log.writeln('Reading mention map from %s...' % options.mention_map_f)
        mention_map = mention_map_lib.load(options.mention_map_f)
        log.writeln('Read mapping info for {0:,} mentions.\n'.format(len(mention_map)))

        log.writeln('Reading doc ID filter list from %s...' % options.filter_doc_ID_f)
        filter_doc_IDs = readFilterDocIDSet(options.filter_doc_ID_f)
        filtered_mentions = []
        for m in mentions:
            if mention_map[m.ID] in filter_doc_IDs:
                filtered_mentions.append(m)
        log.writeln('  Filtered to {0:,} document IDs.'.format(len(filter_doc_IDs)))
        log.writeln('  Filtered mentions: {0:,}.\n'.format(len(filtered_mentions)))

        mentions = filtered_mentions

    log.writeln('Generating cross-validation splits to %s...' % options.output_f)
    labels_by_ID = {
        m.ID : m.CUI.strip().lower()
            for m in mentions
    }
    cross_validation.crossValidationSplits(
        labels_by_ID,
        n_folds=options.num_folds,
        dev_size=options.dev_size,
        persistent_path=options.output_f,
        random_seed=options.random_seed,
        log=log
    )
    log.writeln('Wrote {0}-fold splits files.\n'.format(options.num_folds))

    log.stop()
