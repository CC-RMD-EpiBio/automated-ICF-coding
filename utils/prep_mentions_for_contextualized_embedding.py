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
from dataset import mention_file
from experiments import cross_validation
from dataset import mention_map_lib
from hedgepig_logger import log

def cleanAndSplit(string):
    tokens = [s.strip() for s in string.split()]
    tokens = [
        t
            for t in tokens
            if len(t) > 0
    ]
    return tokens

def writeKeyFile(ordered_keys, mentions_by_id, outf):
    with open(outf, 'w') as stream:
        for m_id in ordered_keys:
            m = mentions_by_id[m_id]
            left_ctx = cleanAndSplit(m.left_context)
            mention = cleanAndSplit(m.mention_text)
            mention_start = len(left_ctx)
            mention_end = len(left_ctx) + len(mention)
            stream.write('%s\t%d\t%d\n' % (
                str(m_id), mention_start, mention_end
            ))

def writeTextFile(ordered_keys, mentions_by_id, outf):
    with open(outf, 'w') as stream:
        for m_id in ordered_keys:
            m = mentions_by_id[m_id]
            left_ctx = cleanAndSplit(m.left_context)
            mention = cleanAndSplit(m.mention_text)
            right_ctx = cleanAndSplit(m.right_context)
            stream.write('%s\n' % (' '.join([
                *left_ctx, *mention, *right_ctx
            ])))

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-m', '--mentions', dest='mentions_f',
            help='(REQUIRED) mentions file to remap to BERT format')
        parser.add_option('-o', '--output', dest='output_f',
            help='(REQUIRED) output file for bare definition texts')
        parser.add_option('-k', '--keys', dest='key_f',
            help='(REQUIRED) output file for keys of definition texts')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        (options, args) = parser.parse_args()

        if not options.mentions_f:
            parser.error('Must supply --mentions')
        elif not options.output_f:
            parser.error('Must provide --output')
        elif not options.key_f:
            parser.error('Must provide --keys')

        return options
    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Mentions file', options.mentions_f),
        ('Output bare text file', options.output_f),
        ('Output key file', options.key_f),
    ], 'Format conversion for contextualized embedding')

    log.writeln('Reading mentions from %s...' % options.mentions_f)
    mentions = mention_file.read(options.mentions_f)
    mentions_by_id = {
        m.ID: m
            for m in mentions
    }
    log.writeln('Read {0:,} mentions.\n'.format(len(mentions)))

    ordered_keys = tuple(sorted(mentions_by_id.keys()))
    log.writeln('Writing output files...')
    log.indent()

    writeKeyFile(ordered_keys, mentions_by_id, options.key_f)
    log.writeln('Wrote keys to %s' % options.key_f)

    writeTextFile(ordered_keys, mentions_by_id, options.output_f)
    log.writeln('Wrote bare mention text to %s' % options.output_f)

    log.unindent()
    log.stop()
