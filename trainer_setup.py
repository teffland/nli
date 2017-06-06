import chainer as ch
from chainer_bw import RetainGrad
from chainer_bw import VariableConverterUpdater, VariableConverterEvaluator
from chainer_bw import ActivationMonitorExtension, BackpropMonitorExtension
from chainer_bw import BetterLogReport
from nli_converter import NLIBatchConverter

def setup(config, data_setup_results, model_setup_results):
    # setup the batch generators
    batch_size = config['batch_size']
    k = config['max_examples']
    if k == 'all': k = len(data_setup_results['train_data'])
    train_iter = ch.iterators.SerialIterator(data_setup_results['train_data'][:k],
                                             batch_size, shuffle=True, repeat=True)
    dev_iter = ch.iterators.SerialIterator(data_setup_results['dev_data'][:k],
                                           batch_size, shuffle=False, repeat=False)
    loss_model = model_setup_results['loss_model']

    # optimize with adam
    optimizer = RetainGrad(ch.optimizers.Adam)(alpha=config['adam_alpha'])
    optimizer.setup(loss_model)

    # setup model runners
    converter = NLIBatchConverter(data_setup_results['token_vocab'],
                                  data_setup_results['class_vocab'])
    updater = VariableConverterUpdater(train_iter, optimizer, converter=converter)
    evaluator = VariableConverterEvaluator(dev_iter,
                                           loss_model, converter=converter)

    # setup trainer and extensions
    trainer = ch.training.Trainer(updater, (config['n_epoch'], 'epoch'), out=config['results_dirname'])

    eval_trigger = tuple(config['evaluation_trigger'])
    trainer.extend(evaluator, trigger=eval_trigger)

    # monitor the forward and backward activations/gradients/updates of the model
    trainer.extend(ActivationMonitorExtension())
    trainer.extend(BackpropMonitorExtension(loss_model))

    # log all montiored values to jsonl
    logger = BetterLogReport(trigger=(1,'iteration'))
    trainer.extend(logger)
    # also print a few choice ones out
    trainer.extend(ch.training.extensions.PrintReport([
        'epoch', 'main/loss', 'main/accuracy', 'validation/main/accuracy'],
        log_report=logger
    ))

    # snapshot the models at each epoch
    trainer.extend(ch.training.extensions.snapshot(
        filename='snapshots/snapshot_iter_{.updater.iteration}'),
        trigger=tuple(config['checkpoint_trigger']
    ))

    # snapshot the best so far also (for early stopping)
    trainer.extend(ch.training.extensions.snapshot(
        filename='snapshots/snapshot_best'),
        trigger=ch.training.triggers.MaxValueTrigger('validation/main/accuracy',
                                                     trigger=eval_trigger))

    return trainer
