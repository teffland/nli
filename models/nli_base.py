from chainer import Chain, functions as F, links as L
from chainer_bw.monitor import monitor

class NLIPredictor(Chain):
    """ Convert pairs of premise and hypothesis sentences (tokenized) into an
    entailment class distribution.
    
    """
    def __init__(self, h_model, c_model, p_model=None):
        super(NLIPredictor, self).__init__()
        self.add_link('h_model', h_model)
        if p_model is not None: 
            self.add_link('p_model', p_model)
        else:
            self.p_model = None
        self.add_link('c_model', c_model)
        
    def _sort(self, xs):
        """ Sort a list of variables by their length descending, 
        returning the argsort too. 
        """
        xs, xs_ids = zip(*sorted(zip(xs, range(len(xs))),
                                       key=lambda x:len(x[0]),
                                       reverse=True))
        return xs, xs_ids
    
    def _unsort(self, xs, xs_ids):
        """ Unsort a lost of variables according to the ids. """
        return [ xs[i] for i in xs_ids ]
    
    def __call__(self, hs, ps):
        # convert to ids and sort by descending length
        hs, hs_ids = self._sort(hs)
        ps, ps_ids = self._sort(ps)
        
        # get sentence representations 
        hs_reps = self.h_model(hs)
        ps_reps = self.p_model(ps) if self.p_model else self.h_model(ps)
        
        # put them back in original order and convert to tensors
        hs_reps = F.vstack(self._unsort(hs_reps, hs_ids))
        ps_reps = F.vstack(self._unsort(ps_reps, ps_ids))
        
        monitor('hs_reps', hs_reps, self)
        monitor('ps_reps', ps_reps, self)
        
        # concatenate them, their diff, and elementwise product
        # 
        # Lili Mou, Men Rui, Ge Li, Yan Xu, Lu Zhang, Rui Yan,and Zhi Jin. 2016b.  
        # Natural language inference by tree-based  convolution  and  heuristic  matching.   
        # InProc. ACL.
        concat_rep = F.hstack([hs_reps, 
                               ps_reps, 
                               hs_reps - ps_reps, 
                               hs_reps * ps_reps])
        
        return self.c_model(concat_rep)
    
class NLILossModel(Chain):
    """ Wraps NLI Predictor with a cross entropy loss for training. """
    def __init__(self, nli_predictor):
        super(NLILossModel, self).__init__(nli_predictor=nli_predictor)
        
    def __call__(self, hs, ps, cs):
        cs_true = cs
        cs_pred = self.nli_predictor(hs, ps)
        loss = F.softmax_cross_entropy(cs_pred, cs_true)
        ch.reporter.report({'loss':loss}, self)
        self.accuracy = F.accuracy(cs_pred, cs_true)
        ch.reporter.report({'accuracy':accuracy}, self)
        precision, recall, f1, support = F.classification_summary(cs_pred, cs_true)
        ch.reporter.report({'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'support':support}, 
                           self)
        return loss