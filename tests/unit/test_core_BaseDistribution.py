"""Tests probflow.parameters modules"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *


def test_BaseDistribution_fit():
    """Tests core.BaseDistribution.fit"""

    # Model = linear regression assuming error = 1
    weight = Parameter()
    bias = Parameter()
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    N = 1000
    true_weight = 0.5
    true_bias = -1
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise

    x = np.reshape(x, (N, 1))
    y = np.reshape(y, (N, 1))

    # Fit the model
    model.fit(x, y, epochs=1)

    # if you run it w/ a bunch of epochs it recovers the true params!
