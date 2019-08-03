"""Tests the probflow.data module"""



import pytest

import numpy as np

from probflow.data import *



def test_DataGenerator():
    """Tests probflow.data.DataGenerator"""

    # Create some data
    x = np.random.randn(100, 3)
    w = np.random.randn(3, 1)
    b = np.random.randn()
    y = x @ w + b
    
    # Create the generator
    dg = DataGenerator(x, y, batch_size=5)

    # Check properties
    assert dg.n_samples == 100
    assert dg.batch_size == 5
    assert dg.shuffle == False

    #len should return # batches per epoch
    assert len(dg) == 20

    # __getitem__ should return a batch
    x1, y1 = dg[0]
    assert isinstance(x1, np.ndarray)
    assert isinstance(y1, np.ndarray)
    assert x1.shape[0] == 5
    assert x1.shape[1] == 3
    assert y1.shape[0] == 5
    assert y1.shape[1] == 1
    
    # should return the same data if called twice
    x2, y2 = dg[0]
    assert np.all(x1==x2)
    assert np.all(y1==y2)
    
    # but not after shuffling
    dg.shuffle = True
    dg.on_epoch_end()
    x2, y2 = dg[0]
    assert np.all(x1!=x2)
    assert np.all(y1!=y2)
    
    # should be able to iterate over batches
    i = 0
    for xx, yy in dg:
        assert isinstance(xx, np.ndarray)
        assert isinstance(yy, np.ndarray)
        assert xx.shape[0] == 5
        assert xx.shape[1] == 3
        assert yy.shape[0] == 5
        assert yy.shape[1] == 1
        i += 1
    assert i == 20

    # should handle if y is None
    dg = DataGenerator(y=x, batch_size=5)
    for xx, yy in dg:
        assert xx is None
        assert isinstance(yy, np.ndarray)
        assert yy.shape[0] == 5
        assert yy.shape[1] == 3

    # and if y is None, should treat x as y (for generative models)
    dg = DataGenerator(x, batch_size=5)
    for xx, yy in dg:
        assert xx is None
        assert isinstance(yy, np.ndarray)
        assert yy.shape[0] == 5
        assert yy.shape[1] == 3

