from cbow import CBOW
from mlp import MLP
from nli_base import NLIPredictor, NLILossModel

def setup(config, data_setup_results):
    cbow = CBOW(data_setup_results['token_embeddings'])
    c_model= MLP(config['mlp_sizes'])
    predictor_model = NLIPredictor(cbow, c_model)
    loss_model = NLILossModel(predictor_model)
    return {
        'loss_model':loss_model,
        'predictor_model':predictor_model
    }