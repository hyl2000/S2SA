#!/usr/bin/env python
# -*- coding: utf-8 -*- 
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: build_vocabulary.py
"""

from __future__ import print_function

import sys
import re
from collections import Counter
import importlib
import json
importlib.reload(sys)

# reload(sys)
# sys.setdefaultencoding('utf8')


def tokenize(s):
    """
    tokenize
    """
    s = re.sub('\d+', '<num>', s).lower()
    tokens = s.split(' ')
    return tokens


def build_vocabulary(corpus_file, vocab_file,
                     vocab_size=30004, min_frequency=0,
                     min_len=1, max_len=500):
    """
    build words dict
    """
    counter = Counter()
    for line in open(corpus_file, 'r', encoding='utf-8'):
        session = json.loads(line.strip(), encoding="utf-8")
        src = ' '.join(session['history'])
        tgt = session['response']
        # src, tgt, knowledge = line.rstrip('\n').split('\t')[:3]
        filter_knowledge = []

        src = tokenize(src)
        tgt = tokenize(tgt)

        if len(src) < min_len or len(src) > max_len or \
           len(tgt) < min_len or len(tgt) > max_len:
            continue

        goal = session['goal']
        goal_full = []
        for g in goal:
            g_type, _, g_entity = g
            goal_full.append(g_type)
            goal_full.append(g_type)

        knowledge = session['knowledge']
        knowledge_full = []
        for k in knowledge:
            a, b, c = k
            knowledge_full.append(a)
            knowledge_full.append(b)
            knowledge_full.append(c)

        counter.update(src + tgt + goal_full + knowledge_full)

    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    words_and_frequencies = words_and_frequencies[:vocab_size]

    fout = open(vocab_file, 'w', encoding='utf-8')
    for word, frequency in words_and_frequencies:
        if frequency < min_frequency:
            break
        fout.write(word + '\n')

    fout.close()


def main():
    """
    main
    """

    build_vocabulary('./data/sample.train.txt', './data/vocab.txt')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
