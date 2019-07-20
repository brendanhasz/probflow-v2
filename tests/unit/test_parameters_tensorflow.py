"""Tests the probflow.parameters module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.parameters import *



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_Parameter():
    """Tests the generic Parameter"""

    # Create the distribution
    param = Parameter()

    # Check defaults
    assert isinstance(param.shape, list)
    assert param.shape[0] == 1
    assert isinstance(param.untransformed_variables, dict)
    assert all(isinstance(p, str) for p in param.untransformed_variables)
    assert all(isinstance(p, tf.Variable)
               for _, p in param.untransformed_variables.items())

    # Shape should be >0
    with pytest.raises(ValueError):
        Parameter(shape=-1)
    with pytest.raises(ValueError):
        Parameter(shape=[20, 0, 1])
