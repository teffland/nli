""" Start training a new Chainer model from a config file. 

TODO: 
* [X] create experiment dir and subdirs
* [X] git commit with autoadd and add commit hash to config
* [X] output filled in config file into experiment dir
* [ ] implement early stopping with patience extension
* [ ] log examples per second, iterations per second, epochs per second, estimated time left
* [ ] email on error or completion
* [ ] description of experiment in yaml file
* [ ] config file can optionally provide dictionary of prereq models, that will be loaded apriori and passed into this experiment to allow for forking experiments that depend on previous models (but aren't just a continuation of the same experiment.

"""
import argparse
import yaml
import imp
import os
import os.path as osp
from datetime import datetime
import sh

import logging
logging.basicConfig(level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s: %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Name of experiment config yaml file')
    return parser.parse_args()

def setup_experiment_dir(config):
    """ Create the experiment output dir. """
    # user can override `results_`dir_name`
    # or they can optionally provide an additional prefix for the datatime dirname
    # or we will provide a defacto name
    dirname = ''
    if 'results_dir_prefix' in config:
        dirname += config['results_dir_prefix']
    if 'results_dir_name' in config:
        dirname += config['results_dir_name']
    else:
        dirname += '{}_experiment'.format(
            datetime.strftime(datetime.now(), '%b-%d-%Y-%H.%M.%f'))
          
    snapshot_dir = osp.join(dirname, 'snapshots')
    tmpsnapshot_dir = osp.join(dirname, 'tmpsnapshots')
    if not osp.exists(dirname): os.makedirs(dirname)
    if not osp.exists(snapshot_dir): os.makedirs(snapshot_dir)
    if not osp.exists(tmpsnapshot_dir): os.makedirs(tmpsnapshot_dir)
    return dirname

def commit(experiment_name):
    sh.git.commit('-a',
            m='"auto commit tracked files for new experiment: {}"'.format(experiment_name),
            allow_empty=True
        )
    commit_hash = sh.git('rev-parse', 'HEAD').strip()
    return commit_hash

def maybe_load_config(setup_config):
    """ Take a config object and do one of the following:
    
    * If `setup_config` is stringlike and is a yaml file,
      then load that as the config
    * If `setup_config` is a dictionary, then just pass that through.
    * Else complain.
    
    """
    if isinstance(setup_config, basestring):
        if setup_config.endswith('.yaml') and osp.exists(setup_config):
            return yaml.load(open(setup_config))
        else:
            logger.critical('Invalid configuration at {}'.format(setup_config))
            raise ValueError, 'Invalid configuration at {}:\n\
must be valid existing yaml'.format(setup_config)
    elif type(setup_config) is dict:
        return setup_config
    else:
        logger.critical('Invalid configuration in `setup_config`')
        raise ValueError, 'Invalid configuration in `setup_config`'

def import_module(module_name, setup_file):
    # todo handle specification of a module, not a file
    return imp.load_source(module_name, setup_file)
                                    
if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open(args.config))
    full_config = config # config that will be output in results for reproducibility
    
    # setup the dataset
    logger.info('Loading dataset')
    data_config = config['data_setup']
    data_setup_config = maybe_load_config(data_config['setup_config'])
    full_config['data_setup']['setup_config'] = data_setup_config
    
    data_setup_module = import_module('data_setup', data_config['setup_file'])
    data_setup_results = data_setup_module.setup(
        data_setup_config)
    
    # setup the model
    logger.info('Setting up models')
    model_config = config['model_setup']
    model_setup_config = maybe_load_config(model_config['setup_config'])
    full_config['model_setup']['setup_config'] = model_setup_config
    
    model_setup_module = import_module('model_setup', model_config['setup_file'])
    model_setup_results = model_setup_module.setup(
        model_setup_config, data_setup_results)
    
    # setup the trainer
    logger.info('Setting up training')
    full_config['cwd'] = os.getcwd()
    results_dir = setup_experiment_dir(config)
    full_config['results_dirname'] = results_dir
    full_config['commit_hash'] = commit(results_dir)
    
    trainer_config = config['trainer_setup']
    trainer_setup_config = maybe_load_config(trainer_config['setup_config'])
    trainer_setup_config['results_dirname'] = results_dir
    full_config['trainer_setup']['setup_config'] = trainer_setup_config
    
    trainer_setup_module = import_module('trainer_setup', trainer_config['setup_file'])
    trainer = trainer_setup_module.setup(
        trainer_setup_config, data_setup_results, model_setup_results)
    
    # save the full configuration
    logger.info('Saving experiment configuration')
    full_config['is_finished'] = False
    yaml.dump(full_config, open(osp.join(results_dir, 'full_config.yaml'),'w'))
    
    # run it
    logger.info('Running trainer')
    trainer.run()
    
    full_config['is_finished'] = True
    logger.info('Finished Experiment')
    
    