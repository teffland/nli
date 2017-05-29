from chainer import Chain, functions as F, links as L

class CBOW(Chain):
    def __init__(self, embeddings):
        super(CBOW, self).__init__(
            token_embeddings=L.EmbedID(embeddings.shape[0], embeddings.shape[1], embeddings)
        )
    
    def __call__(self, xs):
        """ Sum up embeddings for x """
        xs_vecs = [ self.token_embeddings(x) for x in xs ]
        xs_vec = [ F.sum(x_vecs, axis=0) for x_vecs in xs_vecs ]
        return xs_vec
   

