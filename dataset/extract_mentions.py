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
