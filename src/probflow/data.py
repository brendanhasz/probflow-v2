"""Data utilities.

TODO: more info...

----------

"""


__all__ = [
    'DataGenerator',
]



import numpy as np
import pandas as pd

from probflow.core.settings import get_backend
from probflow.core.settings import get_datatype
from probflow.core.base import BaseDataGenerator



class DataGenerator(BaseDataGenerator):
    """Generate data to feed through a model.

    TODO

    """

    def __init__(self, x=None, y=None, batch_size=128, shuffle=True):

        # Check types
        data_types = (np.ndarray, pd.DataFrame, pd.Series)
        if x is not None and not isinstance(x, data_types):
            raise TypeError('x must be an ndarray, a DataFrame, or a Series')
        if y is not None and not isinstance(y, data_types):
            raise TypeError('y must be an ndarray, a DataFrame, or a Series')
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be an int')
        if batch_size < 1:
            raise ValueError('batch_size must be >0')
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False')

        # Check sizes are consistent
        if x is not None and y is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError('x and y must contain same number of samples')

        # Generative model?
        if y is None:
            y = x
            x = None

        # Batch size
        if y.shape[0] < batch_size:
            self._batch_size = y.shape[0]
        else:
            self._batch_size = batch_size

        # Store references to data
        self.x = x
        self.y = y

        # Shuffle data
        self.shuffle = shuffle
        self.on_epoch_end()
        

    @property
    def n_samples(self):
        """Number of samples in the dataset"""
        return self.y.shape[0]


    @property
    def batch_size(self):
        """Number of samples per batch"""
        return self._batch_size


    def __getitem__(self, index):
        """Generate one batch of data"""

        # Get shuffled indexes
        ix = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        # Get x data
        if self.x is None:
            x = None
        elif isinstance(self.x, pd.DataFrame):
            x = self.x.iloc[ix, :]
        elif isinstance(self.x, pd.Series):
            x = self.x.iloc[ix]
        else:
            x = self.x[ix, ...]

        # Get y data
        if isinstance(self.y, pd.DataFrame):
            x = self.y.iloc[ix, :]
        elif isinstance(self.y, pd.Series):
            x = self.y.iloc[ix]
        else:
            y = self.y[ix, ...]

        # Return both x and y
        return x, y


    def on_epoch_end(self):
        """Shuffle data each epoch"""
        if self.shuffle:
            self.ids = np.random.permutation(self.n_samples)
        else:
            self.ids = np.arange(self.n_samples, dtype=np.uint64)
