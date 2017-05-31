import chainer as ch
from chainer_bw import RetainGrad
from chainer_bw import VariableConverterUpdater, VariableConverterEvaluator
from chainer_bw import ActivationMonitorExtension, BackpropMonitorExtension
from chainer_bw import BetterLogReport
from nli_converter import NLIBatchConverter


def setup(config, data_setup_results, model_setup_results):
    loss_model = model_setup_results['loss_model']
    
    optimizer = RetainGrad(ch.optimizers.Adam)(alpha=config['adam_alpha'])
    optimizer.setup(loss_model)
    
    converter = NLIBatchConverter(data_setup_results['token_vocab'], 
                                  data_setup_results['class_vocab'])

    updater = VariableConverterUpdater(data_setup_results['train_iter'], optimizer, converter=converter)
    evaluator = VariableConverterEvaluator(data_setup_results['dev_iter'], 
                                           loss_model, converter=converter)
    activation_monitor = ActivationMonitorExtension()
    backprop_monitor = BackpropMonitorExtension(loss_model)
    logger = BetterLogReport(trigger=(1,'iteration'))

    trainer = ch.training.Trainer(updater, (100, 'epoch'), out='result_test')
    trainer.extend(evaluator)
    trainer.extend(activation_monitor)
    trainer.extend(backprop_monitor)
    trainer.extend(logger)
    # trainer.extend(ch.training.extensions.LogReport(trigger=(1,'iteration'),
    #                                                 postprocess=postprocess))
    trainer.extend(ch.training.extensions.PrintReport([
        'epoch', 'main/loss', 'main/accuracy', 'validation/main/accuracy'],
        log_report=logger
    ))
    trainer.extend(ch.training.extensions.snapshot(
        filename='snapshots/snapshot_iter_{.updater.iteration}', 
        trigger=(1, 'epoch')
    ))
    trainer.extend(ch.training.extensions.snapshot(
        filename='snapshots/snapshot_best', 
        trigger=ch.training.triggers.MaxValueTrigger('validation/main/accuracy', (1,'epoch'))
    ))
    
    return trainer