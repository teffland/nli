from chainer import Chain, functions as F, links as L

class MLP(Chain):
    def __init__(self, h_sizes, fs=[]):
        super(MLP, self).__init__()
        
        # create affine transforms
        self.hs = []
        for i, h_size in enumerate(h_sizes):
            h = L.Linear(None, h_size) # input size is inferred
            self.hs.append(h)
            self.add_link('h_{}'.format(i), h)
        
        # register nonlinearities (as chainer callables)
        if fs:
            assert len(fs) == (len(self.hs)-1), "Must have one less activation than affine transforms"
            self.fs = fs
        else:
            self.fs = [ F.relu for _ in range(len(self.hs)-1) ]
            
    def __call__(self, x):
        z = x
        for i, f in enumerate(self.fs):
            z = f(self.hs[i](z))
        return self.hs[-1](z)