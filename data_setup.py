import logging
logging.basicConfig(level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s: %(name)s: %(message)s')
logger = logging.getLogger(__name__)

import chainer as ch
from chainer_bw.vocab import Vocab
from chainer_bw.word_vectors import get_pretrained_vectors

from dataset import load_data

def setup(config):
    """ Take a setup config dictionary and return a train and dev iter. """
    train_data, dev_data = load_data(config['data_dir'], 
                                     matched=config['matched'], 
                                     genres=config['genres'],
                                     drop_confused=config['drop_confused'], 
                                     lowercase=config['lowercase'])
    logger.info("{} training examples, {} dev examples".format(len(train_data), len(dev_data)))
    
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
    
    return {
        'token_embeddings': token_embeddings,
        'token_vocab': token_vocab,
        'class_vocab': class_vocab
    }