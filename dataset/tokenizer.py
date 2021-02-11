'''
Wrapper for different tokenizers (choice made in configuration file).
'''

class Tokenizer:
    
    SpaCy = 'SpaCy'
    PreTokenized = 'PreTokenized'
    BERT = 'BERT'

    @staticmethod
    def default():
        return Tokenizer.SpaCy
    @staticmethod
    def choices():
        return [
            Tokenizer.SpaCy,
            Tokenizer.PreTokenized,
            Tokenizer.BERT
        ]
    
    def __init__(self, setting, bert_vocab_file=None):
        if setting == Tokenizer.SpaCy:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            self.tokenize = lambda line: [t.text for t in nlp(line)]
        elif setting == Tokenizer.PreTokenized:
            self.tokenize = lambda line: line.split()
        elif setting == Tokenizer.BERT:
            from bert import tokenization
            tokenizer = tokenization.FullTokenizer(
                vocab_file=bert_vocab_file,
                do_lower_case=True
            )
            self.tokenize = tokenizer.tokenize
