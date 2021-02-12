'''
Methods for cross validation, including split persistence to disk
'''

import os
import glob
import math
import codecs
import numpy as np
from hedgepig_logger import log

def readSingleSplits(f, id_cast=int):
    data = []
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (_id, lbl) = [s.strip() for s in line.split('\t')]
            data.append(id_cast(_id))
    return data

def readSplits(f, n_folds=None, id_cast=int):
    splits = []
    if n_folds is None:
        n_folds = len(glob.glob('%s.fold-*.train' % f))
    for i in range(n_folds):
        train = readSingleSplits('%s.fold-%d.train' % (f, i), id_cast=id_cast)
        dev = readSingleSplits('%s.fold-%d.dev' % (f, i), id_cast=id_cast)
        test = readSingleSplits('%s.fold-%d.test' % (f, i), id_cast=id_cast)
        splits.append((train, dev, test))
    return splits

def writeSingleSplits(data, f):
    with codecs.open(f, 'w', 'utf-8') as stream:
        for (_id, lbl) in data:
            stream.write('%s\t%s\n' % (str(_id), str(lbl)))

def writeSplits(splits, f):
    for i in range(len(splits)):
        (train, dev, test) = splits[i]
        writeSingleSplits(train, '%s.fold-%d.train' % (f, i))
        writeSingleSplits(dev, '%s.fold-%d.dev' % (f, i))
        writeSingleSplits(test, '%s.fold-%d.test' % (f, i))

def stratifyByClass(dataset):
    # stratify the data by class
    ids_by_class = {}
    if type(dataset) is dict:
        item_iterator = iter(dataset.items())
    else:
        item_iterator = iter(dataset)
    for (_id, label) in item_iterator:
        if not label in ids_by_class:
            ids_by_class[label] = []
        ids_by_class[label].append(_id)

    classes = list(ids_by_class.keys())
    classes.sort()

    return ids_by_class, classes

def getFoldAndDevSizeByClass(ids_by_class, n_folds, dev_size, no_test=False):
    fold_size_by_class, dev_size_by_class = {}, {}

    dev_multiplier = n_folds

    for (_class, subset) in ids_by_class.items():
        # using ceil ensures a more balanced distribution
        # for everything other than the last fold
        fold_size_by_class[_class] = math.ceil(
            len(subset) / n_folds
        )
        if len(subset) < 3:
            dev_size_by_class[_class] = 0
        else:
            dev_size_by_class[_class] = int(
                fold_size_by_class[_class] * dev_size * dev_multiplier
            )
    return fold_size_by_class, dev_size_by_class

def collapseFromByClass(dct):
    labeled_data, id_data = [], []
    for (k,v) in dct.items():
        labeled_data.extend([
            (v_item, k) for v_item in v
        ])
        id_data.extend([
            v_item for v_item in v
        ])
    return labeled_data, id_data

def crossValidationSplits(dataset, n_folds, dev_size, persistent_path=None, random_seed=1, log=log):
    if persistent_path and os.path.isfile('%s.fold-0.train' % persistent_path):
        log.writeln('Reading pre-existing cross validation splits from %s.' % persistent_path)
        splits = readSplits(persistent_path, n_folds, id_cast=int)
    else:
        log.writeln('Generating cross-validation splits...')
        np.random.seed(random_seed)

        ids_by_class, classes = stratifyByClass(dataset)

        total_size = 0
        for (lbl, ids) in ids_by_class.items():
            total_size += len(ids)
        log.writeln('  Dataset size: {0:,}'.format(total_size))
        log.writeln('  Number of classes: {0:,}'.format(len(classes)))

        # shuffle it
        for _class in classes:
            np.random.shuffle(ids_by_class[_class])

        # figure out how many points of each class per fold
        fold_size_by_class, dev_size_by_class = getFoldAndDevSizeByClass(
            ids_by_class, n_folds, dev_size
        )

        labeled_splits, id_splits = [], []
        for i in range(n_folds):
            train_by_class = {}
            for _class in classes:
                train_by_class[_class] = []

            for j in range(n_folds):
                fold_by_class = {}
                for _class in classes:
                    fold_size = fold_size_by_class[_class]
                    if j < (n_folds - 1):
                        fold_by_class[_class] = ids_by_class[_class][j*fold_size:(j+1)*fold_size]
                    else:
                        fold_by_class[_class] = ids_by_class[_class][j*fold_size:]

                # pull test
                if j == i:
                    test_by_class = fold_by_class.copy()
                # pull dev (portion)
                elif j == ((i + 1) % n_folds):
                    dev_by_class = {}
                    for (_class, subset) in fold_by_class.items():
                        dev_by_class[_class] = subset[:dev_size_by_class[_class]]
                        train_by_class[_class].extend(
                            subset[dev_size_by_class[_class]:]
                        )
                # everything else goes to training
                else:
                    for (_class, subset) in fold_by_class.items():
                        train_by_class[_class].extend(subset)

            # collapse train, dev, test to flat ID lists
            lbl_train, id_train = collapseFromByClass(train_by_class)
            lbl_dev, id_dev = collapseFromByClass(dev_by_class)
            lbl_test, id_test = collapseFromByClass(test_by_class)
            
            labeled_splits.append((lbl_train, lbl_dev, lbl_test))
            id_splits.append((id_train, id_dev, id_test))

            log.writeln('  Fold {0} -- Train: {1:,}  Dev: {2:,}  Test: {3:,}'.format(
                i+1, len(id_train), len(id_dev), len(id_test)
            ))

        if persistent_path:
            log.writeln('Writing cross validation splits to %s.' % persistent_path)
            writeSplits(labeled_splits, persistent_path)

        splits = id_splits
    log.writeln()

    return splits
