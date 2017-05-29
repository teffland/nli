import argparse
import yaml
import importlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Name of experiment config yaml file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open(args.config))
    
    # setup the dataset
    data_config = config['data_setup']
    data_setup_module = importlib.import(data_config['setup_file'])
    train_iter, dev_iter, data_setup_extras = data_setup_module.setup(
        data_config['setup_config'])
    
    # setup the model
    model_config = config['model_setup']
    model_setup_module = importlib.import(model_config['setup_file'])
    predictor_model, loss_model, model_setup_extras = model_setup_module.setup(
        model_config['setup_config'], data_setup_extras)
    
    # setup the trainer
    trainer_config = config['trainer_setup']
    trainer_setup_module = importlib.import(trainer_config['setup_file'])
    trainer = trainer_setup_module.setup(
        trainer_config['setup_config'], data_setup_extras, model_setup_extras)
    
    # run it
    trainer.run()
    
    