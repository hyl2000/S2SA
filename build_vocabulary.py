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


def build_vocabulary(corpus_file, vocab_file, entites_file, relations_file, vocab_size=30004, min_frequency=0,
                     min_len=1, max_len=500):
    """
    build words dict
    """
    counter = Counter()
    entites_counter = Counter()
    relations_counter = Counter()
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
        entites_vocab = []
        relations_vocab = []
        for k in knowledge:
            a, b, c = k
            entites_vocab.append(a)
            relations_vocab.append(b)
            entites_vocab.append(c)

        counter.update(src + tgt + goal_full)
        entites_counter.update(entites_vocab)
        relations_counter.update(relations_vocab)

    for line in open('./data/sample.xtest.txt', 'r', encoding='utf-8'):
        session = json.loads(line.strip(), encoding="utf-8")

        knowledge = session['knowledge']
        entites_vocab = []
        relations_vocab = []
        for k in knowledge:
            a, b, c = k
            entites_vocab.append(a)
            relations_vocab.append(b)
            entites_vocab.append(c)

        entites_counter.update(entites_vocab)
        relations_counter.update(relations_vocab)

    for line in open('./data/sample.xdev.txt', 'r', encoding='utf-8'):
        session = json.loads(line.strip(), encoding="utf-8")

        knowledge = session['knowledge']
        entites_vocab = []
        relations_vocab = []
        for k in knowledge:
            a, b, c = k
            entites_vocab.append(a)
            relations_vocab.append(b)
            entites_vocab.append(c)

        entites_counter.update(entites_vocab)
        relations_counter.update(relations_vocab)

    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    words_and_frequencies = words_and_frequencies[:vocab_size]

    fout = open(vocab_file, 'w', encoding='utf-8')
    for word, frequency in words_and_frequencies:
        if frequency < min_frequency:
            break
        fout.write(word + '\n')

    fout.close()

    entites_frequencies = sorted(entites_counter.items(), key=lambda tup: tup[0])
    entites_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    entites_frequencies = entites_frequencies[:vocab_size]

    fout = open(entites_file, 'w', encoding='utf-8')
    for word, frequency in entites_frequencies:
        if frequency < min_frequency:
            break
        fout.write(word + '\n')

    fout.close()

    relations_frequencies = sorted(relations_counter.items(), key=lambda tup: tup[0])
    relations_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    relations_frequencies = relations_frequencies[:vocab_size]

    fout = open(relations_file, 'w', encoding='utf-8')
    for word, frequency in relations_frequencies:
        if frequency < min_frequency:
            break
        fout.write(word + '\n')

    fout.close()


def main():
    """
    main
    """

    build_vocabulary('./data/sample.xtrain.txt', './data/vocab.txt', './data/entities.txt', './data/relations.txt')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
