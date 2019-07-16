"""Callbacks during training.

The callbacks module contains classes for monitoring and adjusting the 
training process.

"""


__all__ = [
    'Callback',
]



from probflow.core.base import BaseCallback



class Callback(BaseCallback):
    """

    TODO

    """
    
    def __init__(self, *args):
        pass


    def on_epoch_begin(self):
        """Will be called at the beginning of each training epoch"""
        pass


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        pass


    def on_train_begin(self):
        """Will be called at the beginning of training"""
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


    def on_epoch_begin(self):
        """Set the learning rate at the beginning of each epoch"""
        self.model.set_learning_rate(self.fn(self.epochs))
        self.epochs += 1



class EarlyStopping(Callback):
    """Stop training early when loss stops improving

    TODO

    """
    
    def __init__(self, patience=0, metric='val_loss'):

        # Check types
        if not isinstance(patience, int):
            raise TypeError('patience must be an int')
        if patience < 0:
            raise ValueError('patience must be non-negative')
        if not isinstance(metric, str):
            raise TypeError('metric must be a str')

        # Store values
        self.patience = patience
        self.metric = metric
        self.best = np.Inf
        self.count = 0
        # TODO: restore_best_weights?


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        metric = self.model.metrics[self.model.epochs][-1]
        if metric < self.best:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                self.model.stop_training()



# TODO: record loss/metric callback?



# TODO: model now has to have:
# metrics dict w/ 'train_loss' and 'val_loss' keys
# _is_training param which is a bool and stop training if it's false
# stop_training() method which sets _is_training to False