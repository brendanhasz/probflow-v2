"""Data utilities.

TODO: more info...

----------

"""



import numpy as np



def process_data(model, x=None, data=None, name='x'):
    """Process and validate one set of data.

    Parameters
    ----------
    x : |ndarray| or int or str or list of str or int
        Values of the dataset to process.  If ``data`` was passed as a 
        |DataFrame|, ``x`` can be an int or string or list of ints or strings
         specifying the columns of that |DataFrame| to process.

    Returns
    -------
    |ndarray|
        The processed data.
    """

    # Ensure we were passed a model object
    # TODO

    # Use training data if none passed
    if x is None:
        try:
            x = model._train[name]
        except AttributeError:
            raise RuntimeError(name+' cannot be None if model is not fit')

    # Numpy arrays
    if data is None:

        # Ensure data is numpy array
        if not isinstance(x, np.ndarray):
            raise TypeError(name+' must be a numpy ndarray')

    else:

        # Pandas DataFrame
        from pandas import DataFrame
        if isinstance(data, DataFrame):

            # Check type
            if not isinstance(x, (int, str, list)):
                raise TypeError(name+' must be an int, string, or list')
            if isinstance(x, list):
                if not all([isinstance(e, (int, str)) for e in x]):
                    raise TypeError(name+' must be a list of int or str')

            # Get the columns
            x = data.ix[:,x].values

        else:
            raise TypeError(name+' must be None or a pandas DataFrame')

    # Make data at least 2 dimensional (0th dim should be N)
    if x.ndim == 1:
        x = np.expand_dims(x, 1)

    # TODO: ensure x data shape matches model._ph['x'] shape (only if fit)

    return x


def process_xy_data(self, x=None, y=None, data=None):
    """Process and validate both x and y data

    Parameters
    ----------
    x : |ndarray| or int or str or list of str or int
        Values of the dataset to process.  If ``data`` was passed as a 
        |DataFrame|, ``x`` can be an int or string or list of ints or strings
         specifying the columns of that |DataFrame| to process.
    y : |ndarray| or int or str or list of str or int
        Values of a second dataset to process.  If ``data`` was passed as a 
        |DataFrame|, ``y`` can be an int or string or list of ints or strings
         specifying the columns of that |DataFrame| to process.

    Returns
    -------
    (|ndarray|, |ndarray|)
        The processed x and y data.
    """

    # Both or neither of x and y should be passed
    if x is None and y is not None or y is None and x is not None:
        raise TypeError('x and y should both be set or both be None')

    # Process both x and y data
    x = process_data(self, x, data, name='x')
    y = process_data(self, y, data, name='y')

    # Check that x and y have correct shape
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same number of samples')

    return x, y


def test_train_split(x, y, val_split, val_shuffle):
    """Split data into training and validation data

    Parameters
    ----------
    x : |ndarray|
        Independent variable values.
    y : |ndarray|
        Dependent variable values.
    val_split : float between 0 and 1
        Proportion of the data to use as validation data.
    val_shuffle : bool
        Whether to shuffle which data is used for validation.  If False,
        the last ``val_split`` proportion of the input data is used
        for validation.

    Returns
    -------
    (N, x_train, y_train, x_val, y_val)

        * N: number of training samples
        * x_train: independent variable values of the training data
        * y_train: dependent variable values of the training data
        * x_val: independent variable values of the validation data
        * y_val: dependent variable values of the validation data
    """
    if val_split > 0:
        num_val = int(val_split*x.shape[0])
        train_ix = np.full(x.shape[0], True)
        train_ix[-num_val:] = False
        if val_shuffle:
            train_ix = np.random.permutation(train_ix)
        val_ix = ~train_ix
        x_train = x[train_ix, ...]
        y_train = y[train_ix, ...]
        x_val = x[val_ix, ...]
        y_val = y[val_ix, ...]
    else:
        x_train = x
        y_train = y
        x_val = x
        y_val = y
    return x_train.shape[0], x_train, y_train, x_val, y_val


def initialize_shuffles(N, epochs, shuffle):
    """Initialize shuffling of the data across epochs"""
    shuffled_ids = np.empty((N, epochs), dtype=np.uint64)
    for epoch in range(epochs):
        if shuffle:
            shuffled_ids[:, epoch] = np.random.permutation(N)
        else:
            shuffled_ids[:, epoch] = np.arange(N, dtype=np.uint64)
    return shuffled_ids


def generate_batch(x, y, epoch, batch, batch_size, shuff_ids):
    """Generate data for one batch"""
    N = x.shape[0]
    a = batch*batch_size
    b = min(N, (batch+1)*batch_size)
    ix = shuff_ids[a:b, epoch]
    return x[ix, ...], y[ix, ...], [ix.shape[0]]
