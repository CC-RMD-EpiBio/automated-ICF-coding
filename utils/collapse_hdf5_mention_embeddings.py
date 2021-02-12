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
Given an output HDF5 file from running BERT/ELMo on definitions,
takes the average of hidden states for each key to create
per-definition embeddings.
'''

import h5py
import codecs
import numpy as np
from dataset import mention_file
from hedgepig_logger import log

AVERAGE_LAYERS=-150

def readKeys(f):
    keys = []
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (m_id, mention_start, mention_end) = [int(s) for s in line.split('\t')]
            keys.append( (m_id, mention_start, mention_end) )
    return keys
    
def collapseMentionEmbeddings(f, keys, layer, mentions_by_id, action_oracle):
    log.track(message='  >> Processed {0}/{1:,} mentions'.format(
        '{0:,}', len(keys)
    ), writeInterval=50)
    new_mentions = []

    with h5py.File(f, 'r') as stream:
        for i in range(len(keys)):
            (m_id, mention_start, mention_end) = keys[i]
            mention_token_embeddings = stream[str(i)][...]

            if layer == AVERAGE_LAYERS:
                mention_token_embeddings = np.mean(mention_token_embeddings, axis=0)
            else:
                mention_token_embeddings = mention_token_embeddings[layer, :, :]

            if action_oracle:
                mention_token_embeddings = mention_token_embeddings[mention_start:mention_end]
            mention_embedding = np.mean(mention_token_embeddings, axis=0)

            old_mention = mentions_by_id[m_id]
            new_mentions.append(mention_file.EmbeddedMention(
                CUI=old_mention.CUI,
                mention_repr=None,
                context_repr=mention_embedding,
                candidates=old_mention.candidates,
                ID=old_mention.ID
            ))

            log.tick()
    log.flushTracker()

    return new_mentions

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--input', dest='input_f',
            help='(REQUIRED) HDF5 embeddings file')
        parser.add_option('-o', '--output', dest='output_f',
            help='(REQUIRED) output embedding file (binary)')
        parser.add_option('-k', '--keys', dest='key_f',
            help='(REQUIRED) file listing keys for HDF5 rows output')
        parser.add_option('-m', '--mentions', dest='mentions_f',
            help='(REQUIRED) textual mentions file to use for reference')
        parser.add_option('--layer', dest='layer',
            type='int', default=AVERAGE_LAYERS,
            help='Hidden state layer index; default averages over all layers')
        parser.add_option('--action-oracle', dest='action_oracle',
            action='store_true', default=False,
            help='use Action location in output')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)
        (options, args) = parser.parse_args()
        if not options.input_f:
            parser.error('Must provide --input')
        elif not options.output_f:
            parser.error('Must provide --output')
        elif not options.key_f:
            parser.error('Must provide --keys')
        return options
    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('HDF5 embeddings', options.input_f),
        ('HDF5 layer', ('Average' if options.layer == AVERAGE_LAYERS else options.layer)),
        ('Per-row keys', options.key_f),
        ('Mentions file', options.mentions_f),
        ('Using Action oracle', options.action_oracle),
        ('Output embedded mentions file', options.output_f),
    ], 'Embedded mentions file generation with pre-generated HDF5 features')

    log.writeln('Reading keys from %s...' % options.key_f)
    keys = readKeys(options.key_f)
    log.writeln('Read {0:,} keys.\n'.format(len(keys)))

    log.writeln('Reading textual mentions from %s...' % options.mentions_f)
    mentions = mention_file.read(options.mentions_f)
    mentions_by_id = {
        m.ID: m
            for m in mentions
    }
    log.writeln('Read {0:,} mentions.\n'.format(len(mentions)))

    log.writeln('Generating embedded mentions from HDF5 file %s...' % options.input_f)
    new_mentions = collapseMentionEmbeddings(options.input_f, keys, options.layer, mentions_by_id, options.action_oracle)
    log.writeln('Generated {0:,} embedded mentions.\n'.format(len(new_mentions)))

    log.writeln('Writing mentions to %s...' % options.output_f)
    mention_file.write(new_mentions, options.output_f)
    log.writeln('Done.\n')

    log.stop()
