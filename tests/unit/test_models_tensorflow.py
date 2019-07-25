"""Tests the probflow.models module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.core.settings import Sampling
import probflow.core.ops as O
from probflow.distributions import Normal
from probflow.parameters import *
from probflow.modules import *
from probflow.models import *



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_Model():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter()
            self.bias = Parameter()
            self.std = ScaleParameter()

        def __call__(self, x):
            return Normal(x*self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Fit the model
    x = np.random.randn(100)
    y = -x + 1
    my_model.fit(x, y, batch_size=5, epochs=10)




def test_Model_nesting():
    """Tests Model when it contains Modules and sub-modules"""
    pass
    # TODO



def test_ContinuousModel():
    """Tests probflow.models.ContinuousModel"""
    pass
    #TODO



def test_DiscreteModel():
    """Tests probflow.models.DiscreteModel"""
    pass
    #TODO



def test_CategoricalModel():
    """Tests probflow.models.CategoricalModel"""
    pass
    #TODO

