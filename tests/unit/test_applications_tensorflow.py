"""Tests the probflow.applications module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import applications as apps



def test_LinearRegression():
    """Tests probflow.applications.LinearRegression"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Create the model
    model = apps.LinearRegression(5)

    # Fit the model
    model.fit(x, y)
    
    # Predictive functions
    model.predict(x)



def test_LogisticRegression():
    """Tests probflow.applications.LinearRegression"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    y = np.round(1.0/(1.0+np.exp(-y))).astype('int32')

    # Create the model
    model = apps.LogisticRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10)
    
    # Predictive functions
    model.predict(x)

