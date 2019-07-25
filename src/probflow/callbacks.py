"""Callbacks during training.

The callbacks module contains classes for monitoring and adjusting the 
training process.

* :class:`.Callback` - abstract base class for all callbacks
* :class:`.LearningRateScheduler` - set the learning rate by epoch
* :class:`.MonitorMetric` - record a metric over the course of training
* :class:`.EarlyStopping` - stop training if some metric stops improving

----------

"""


__all__ = [
    'Callback',
    'LearningRateScheduler',
    'MonitorMetric',
    'EarlyStopping',
]



from probflow.core.base import BaseCallback
from probflow.data import DataGenerator
from probflow.utils.metrics import get_metric_fn



class Callback(BaseCallback):
    """

    TODO

    """
    
    def __init__(self, *args):
        pass


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        pass


    def on_train_end(self):
        """Will be called at the end of training"""
        pass



class LearningRateScheduler(Callback):
    """Set the learning rate as a function of the current epoch

    Parameters
    ----------
    fn : callable
        Function which takes the current epoch as an argument and returns a 
        learning rate.


    Examples
    --------

    TODO
    """
    
    def __init__(self, fn):
        
        # Check type
        if not callable(fn):
            raise TypeError('fn must be a callable')
        if not isinstance(fn(1), float):
            raise TypeError('fn must return a float')

        # Store function
        self.fn = fn
        self.epochs = 0


    def on_epoch_end(self):
        """Set the learning rate at the end of each epoch"""
        self.model.set_learning_rate(self.fn(self.epochs))
        self.epochs += 1



class MonitorMetric(Callback):
    """Monitor some metric on validation data

    """

    def __init__(self, x, y=None, metric='log_prob', verbose=True)

        # Store metric
        self.metric_fn = get_metric_fn(metric)

        # Store validation data
        if isinstance(x, DataGenerator):
            self.data = x
        else:
            self.data = DataGenerator(x, y, batch_size=x.shape[0], 
                                      shuffle=False)

        # Store metrics and epochs
        self.current_metric = np.nan
        self.current_epoch = 0
        self.metrics = []
        self.epochs = []
        self.verbose = verbose


    def on_epoch_end(self):
        """Compute metric on validation data"""
        self.current_metric = self.model.metric(self.data, 
                                                metric=self.metric_fn)
        self.current_epoch += 1
        self.metrics += [self.current_metric]
        self.epochs += [self.current_epoch]
        if self.verbose:
            print('Epoch {} \t{}: {}'.format(
                  self.current_epoch,
                  self.metric_fn.__name__,
                  self.current_metric))



class EarlyStopping(Callback):
    """Stop training early when some metric stops decreasing

    TODO

    Example
    -------

    To monitor the mean absolute error of a model, we can create a 
    :class:`.MonitorMetric` callback:

    .. code-block:: python

        monitor_mae = MonitorMetric(x_val, y_val, 'mse')
        early_stopping = EarlyStopping(lambda: monitor_mae.current_metric)

        model.fit(x_train, y_train, callbacks=[monitor_mae, early_stopping])

    TODO
    """
    
    def __init__(self, metric_fn, patience=0):

        # Check types
        if not isinstance(patience, int):
            raise TypeError('patience must be an int')
        if patience < 0:
            raise ValueError('patience must be non-negative')
        if not callable(metric_fn):
            raise TypeError('metric_fn must be a callable')

        # Store values
        self.metric_fn = metric_fn
        self.patience = patience
        self.best = np.Inf
        self.count = 0
        # TODO: restore_best_weights? using save_model and load_model?


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        metric = self.metric_fn()
        if metric < self.best:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                self.model.stop_training()
