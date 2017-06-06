import numpy as np
import chainer as ch

class NLIBatchConverter(object):
    """ Converter for use with :class:`VariableConverterUpdater` and
    :class:`VariableConverterEvaluator`.

    This object converts string tokens to ids for each sequence,
    then wraps that sequence in a ch.Variable, passing back a full
    dictionary of variables.
    """
    def __init__(self, token_vocab, class_vocab):
        """
        Args:
            token_vocab (:class:`Vocab`): the token vocab object
            class_vocab (:class:`Vocab`): the class vocab object
        """
        self.token_vocab = token_vocab
        self.class_vocab = class_vocab

    def __call__(self, batch):
        """ Convert a batch of data into dictionary containing input data
        for downstream NLI model.

        Args:
            batch (list): list of tokenized NLI data

        Returns:
            in_vars (dict): dictionary of converted NLI data
              'hs': list of variable arrays of hypothesis sentences
              'ps': list of variable arrays of premise sentences
              'cs': variable array of classes
        """
        hs = [ ch.Variable(np.array(
                 [ self.token_vocab.idx(token) for token in datum['h'] ],
               dtype=np.int32))
               for datum in batch ]
        ps = [ ch.Variable(np.array(
                 [ self.token_vocab.idx(token) for token in datum['p'] ],
               dtype=np.int32))
               for datum in batch ]
        cs = ch.Variable(np.array(
               [ self.class_vocab.idx(datum['c']) for datum in batch ],
             dtype=np.int32))
        in_vars = {
            'hs':hs,
            'ps':ps,
            'cs':cs
        }
        return in_vars
