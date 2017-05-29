# coding utf-8
""" Load, tokenize, and organize MultiNLI data into train and dev sets """
import os.path as osp
import json

from nltk.tokenize import TweetTokenizer

import logging
logger = logging.getLogger(__name__)

def load_data(multi_dir, matched=True, genres=[], drop_confused=True, lowercase=True):
    """ Acceptable _matched_ genres: {u'travel', u'fiction', u'slate', u'telephone', u'government'} """
    tokenizer = TweetTokenizer()
    if lowercase:
        tokenize = lambda x: [ t.lower() for t in tokenizer.tokenize(x)]
    else:
        tokenize = tokenizer.tokenize
    train_data = []
    logging.info("Loading Training Data...")
    for i, line in enumerate(open(osp.join(multi_dir, 'multinli_0.9_train.jsonl'))):
        datum = json.loads(line)
        if genres and datum['genre'] not in genres: # skip unspecified genres if we have a specification
            continue
        if drop_confused and datum['gold_label'] == '-':
            continue
        train_data.append({
            'genre':datum['genre'],
            'p':tokenize(datum['sentence1']),
            'h':tokenize(datum['sentence2']),
            'c':datum['gold_label'],
            'cs':datum['annotator_labels']
        })
        # train_data['genres'].append(datum['genre'])
        # train_data['hs'].append(tokenize(datum['sentence1']))
        # train_data['ps'].append(tokenize(datum['sentence2']))
        # train_data['ys'].append(datum['gold_label'])
    dev_data = [] #{'genre':[], 'h':[], 'p':[], 'c':[]}
    if matched:
        fname = 'multinli_0.9_dev_matched.jsonl' 
        logging.info("Loading Matched Dev Data...")
    else:
        fname = 'multinli_0.9_dev_mismatched.jsonl'
        logging.info("Loading Mismatched Dev Data...")
    for i, line in enumerate(open(osp.join(multi_dir, fname))):
        datum = json.loads(line)
        if genres and matched and datum['genre'] not in genres: # skip unspecified genres if we have a specification
            continue
        if drop_confused and datum['gold_label'] == '-':
            continue
        dev_data.append({
            'genre':datum['genre'],
            'p':tokenize(datum['sentence1']),
            'h':tokenize(datum['sentence2']),
            'c':datum['gold_label'],
            'cs':datum['annotator_labels']
        })
        #dev_data['genres'].append(datum['genre'])
        #dev_data['hs'].append(tokenize(datum['sentence1']))
        #dev_data['ps'].append(tokenize(datum['sentence2']))
        #dev_data['ys'].append(datum['gold_label'])
    logging.info('Data Loaded')
    return train_data, dev_data