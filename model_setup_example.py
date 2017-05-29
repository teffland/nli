from cbow import CBOW
from mlp import MLP
from nli_base import NLIPredictor, NLILossModel

def setup(config, data_setup_extras):
    cbow = CBOW(data_setup_extras['token_embeddings'])
    c_model= MLP(config['mlp_sizes'])
    predictor = NLIPredictor(cbow, c_model)
    loss_model = NLILossModel(predictor)
    return predictor, loss_model, {}