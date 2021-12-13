#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: convert_session_to_sample.py
"""

from __future__ import print_function

import sys
import json
import collections
import importlib
importlib.reload(sys)

# reload(sys)
# sys.setdefaultencoding('utf8')


def clean(text: str):
    if text.startswith('['):
        text = text[3:]
    text = text.strip()
    return text


def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    fout = open(sample_file, 'w', encoding='utf-8')  # 或许Linux下不需要加encoding
    with open(session_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)
            '''
            if "test" in session_file:
                conversation = session["history"]
            else:
                conversation = session["conversation"]
            '''
            conversation = session["conversation"]
            label = session['label']
            count = 0
            length = len(conversation)
            for i in range(length):
                conversation[i] = clean(conversation[i])
            if label == 'bot':
                for j in range(0, len(conversation), 2):
                    sample = collections.OrderedDict()
                    sample["goal"] = session["goal"]
                    sample["knowledge"] = session["knowledge"]
                    sample["history"] = []
                    # sample["history"] = [session["name"]]
                    # sample["history"].append(session["situation"])
                    sample["history"] += conversation[:j]
                    sample["response"] = conversation[j]

                    if j < len(conversation) - 1:
                        sample["next"] = conversation[j + 1]  # 用户的回复
                    else:
                        sample["next"] = ""
                    if j != 0:
                        count += 1
                    sample["emotion"] = session["emotion"][:count + 1]  # emotion的序列

                    sample["reverse"] = conversation[j + 1:]
                    sample["reverse"] = list(reversed(sample["reverse"]))

                    sample = json.dumps(sample, ensure_ascii=False)
                    sample.encode("utf-8")

                    fout.write(sample + "\n")
            else:
                for j in range(1, len(conversation), 2):
                    sample = collections.OrderedDict()
                    sample["goal"] = session["goal"]
                    sample["knowledge"] = session["knowledge"]
                    sample["history"] = []
                    # sample["history"] = [session["name"]]
                    # sample["history"].append(session["situation"])
                    sample["history"] += conversation[:j]
                    sample["response"] = conversation[j]

                    if j < len(conversation) - 1:
                        sample["next"] = conversation[j + 1]  # 用户的回复
                    else:
                        sample["next"] = ""
                    count += 1
                    sample["emotion"] = session["emotion"][:count + 1]  # emotion的序列，最后一个是下一句话的emotion

                    sample["reverse"] = conversation[j + 1:]
                    sample["reverse"] = list(reversed(sample["reverse"]))

                    sample = json.dumps(sample, ensure_ascii=False)
                    sample.encode("utf-8")


                    fout.write(sample + "\n")

    fout.close()


def main():
    """
    main
    """
    convert_session_to_sample('./data/train.txt', './data/sample.train.txt')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
