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
