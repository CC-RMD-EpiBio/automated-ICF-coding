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
