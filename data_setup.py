from __future__ import print_function

from dataset import load_data
from vocab import Vocab
from word_vectors import get_pretrained_vectors

def setup(config):
    """ Take a setup config dictionary and return a train and dev iter. """
    train_data, dev_data = load_data(config['data_dir'], 
                                     matched=config['matched'], 
                                     genres=config['genres'],
                                     drop_confused=config['drop_confused'], 
                                     lowercase=config['lowercase'])
    print("{} training examples, {} dev examples".format(len(train_data), len(dev_data)))
    
    token_vocab = Vocab(min_count=1)
    token_vocab.add([ token for datum in train_data for token in datum['h'] ])
    token_vocab.add([ token for datum in train_data for token in datum['p'] ])
    token_vocab.add([ token for datum in dev_data for token in datum['h'] ])
    token_vocab.add([ token for datum in dev_data for token in datum['p'] ])
    token_vocab.drop_infrequent()
    
    token_embeddings = get_pretrained_vectors(token_vocab, 
                                              config['pretrained_vectors'], 
                                              normed=config['normalize_vectors'])
    
    class_vocab = Vocab([ datum['c'] for datum in train_data ],
                        min_count=0,
                        pad_token=None,
                        unk_token=None)
    
    batch_size = config['batch_size']
    k = config['max_examples']
    if k == 'all': k = len(train_data)
    train_iter = ch.iterators.SerialIterator(train_data[:k], batch_size, shuffle=True, repeat=True)
    dev_iter = ch.iterators.SerialIterator(dev_data[:k], batch_size, shuffle=False, repeat=False)
    extras = {
        'token_embeddings': token_embeddings
        'token_vocab': token_vocab,
        'class_vocab': class_vocab
    }
    return train_iter, dev_iter, extras