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
I/O wrappers for mention files (used for streaming feature generation)
'''

import codecs

class Mention:
    def __init__(self, CUI, mention_text, left_context, right_context, candidates, ID=None):
        self.CUI = CUI
        self.mention_text = mention_text
        self.left_context = left_context
        self.right_context = right_context
        self.candidates = candidates
        self.ID = ID

    def copy(self):
        return Mention(
            CUI=str(self.CUI),
            mention_text=str(self.mention_text),
            left_context=str(self.left_context),
            right_context=str(self.right_context),
            candidates=self.candidates.copy(),
            ID=self.ID
        )

class EmbeddedMention:
    def __init__(self, CUI, mention_repr, context_repr, candidates, ID=None):
        self.CUI = CUI
        self.mention_repr = mention_repr
        self.context_repr = context_repr
        self.candidates = candidates
        self.ID = ID

    def copy(self):
        return EmbeddedMention(
            CUI=str(self.CUI),
            mention_repr=self.mention_repr,
            context_repr=self.context_repr,
            candidates=self.candidates.copy(),
            ID=self.ID
        )

def write(mentions, outf, encoding='utf-8'):
    max_ID = 0
    for m in mentions:
        if not m.ID is None and m.ID > max_ID:
            max_ID = m.ID

    fmt_str = 'TXT'
    if type(mentions[0]) is EmbeddedMention:
        fmt_str = 'BIN'

    with codecs.open(outf, 'w', encoding) as stream:
        stream.write('%s\n' % fmt_str)
        for m in mentions:
            if m.ID is None:
                m.ID = max_ID
                max_ID += 1
            if type(m) is Mention:
                stream.write('%s\n' % '\t'.join([
                    str(m.ID),
                    m.mention_text,
                    m.left_context,
                    m.right_context,
                    m.CUI,
                    '||'.join(m.candidates)
                ]))
            elif type(m) is EmbeddedMention:
                stream.write('%s\n' % '\t'.join([
                    str(m.ID),
                    'None' if not m.mention_repr else (' '.join([str(f) for f in m.mention_repr])),
                    ' '.join([str(f) for f in m.context_repr]),
                    m.CUI,
                    '||'.join(m.candidates)
                ]))
            else:
                raise Exception("Can't write mention of type '%s'!" % repr(type(m)))

def read(mentionf, encoding='utf-8'):
    mentions = []
    with codecs.open(mentionf, 'r', encoding) as stream:
        fmt = stream.read(3)
        stream.seek(0)
        if not fmt in ['BIN', 'TXT']:
            fmt = 'TXT'
        else:
            print(stream.readline())

        for line in stream:
            chunks = [s.strip() for s in line.split('\t')]
            if fmt == 'TXT':
                if len(chunks) == 5:
                    ix = 0
                    _id = len(mentions)
                elif len(chunks) == 6:
                    ix = 1
                    _id = int(chunks[0])
                (
                    mention_text,
                    left_context,
                    right_context,
                    CUI,
                    candidates
                ) = chunks[ix:]
                candidates = candidates.split('||')
                if len(candidates) == 1 and candidates[0] == '':
                    candidates = []
                mentions.append(Mention(
                    CUI, mention_text, left_context, right_context, candidates, ID=_id
                ))
            else:
                _id = int(chunks[0])
                if chunks[1] == 'None':
                    mention_repr = None
                else:
                    mention_repr = [float(f) for f in chunks[1].split()]
                context_repr = [float(f) for f in chunks[2].split()]
                CUI = chunks[3]
                candidates = chunks[4].split('||')
                if len(candidates) == 1 and candidates[0] == '':
                    candidates = []
                mentions.append(EmbeddedMention(
                    CUI, mention_repr, context_repr, candidates, ID=_id
                ))
    return mentions
