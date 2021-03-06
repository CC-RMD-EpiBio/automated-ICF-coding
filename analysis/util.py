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
import glob

LOG_IGNORE_EXTENSIONS = set(['.dev_results', '.dev_accs', '.dev_losses', '.UNCORRECTED'])
WSD_LOG_IGNORE_EXTENSIONS = set()
PREDICTIONS_IGNORE_EXTENSIONS = set(['.detailed', '.correct', '.wrong', '.VERB', '.NOUN', '.ADJ', '.ADV', '.with_backoff'])

def getLatestFile(base, extension, ignores):
    globs = list(glob.glob('%s.%s.*' % (base, extension)))

    try:
        valid_logs = []
        for f in globs:
            if os.path.splitext(f)[1] in ignores:
                pass
            else:
                valid_logs.append(f)

        valid_logs.sort(reverse=True)
        return valid_logs[0]

    except:
        return None

def getLatestLog(base):
    return getLatestFile(base, 'log', LOG_IGNORE_EXTENSIONS)

def getLatestWSDLog(base):
    return getLatestFile(base, 'log_wsd_eval_framework', WSD_LOG_IGNORE_EXTENSIONS)

def getLatestPredictions(base):
    return getLatestFile(base, 'predictions', PREDICTIONS_IGNORE_EXTENSIONS)
