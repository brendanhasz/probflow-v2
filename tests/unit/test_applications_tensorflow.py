"""Tests the probflow.applications module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import applications as apps
from probflow.models import Model
from probflow.distributions import Normal



def test_LinearRegression():
    """Tests probflow.applications.LinearRegression"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Create the model
    model = apps.LinearRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
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
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)



def test_MultinomialLogisticRegression():
    """Tests probflow.applications.LinearRegression w/ >2 output classes"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 3).astype('float32')
    b = np.random.randn(1, 3).astype('float32')
    y = x@w + b
    y = np.argmax(y, axis=1).astype('int32')

    # Create the model
    model = apps.LogisticRegression(d=5, k=3)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)



def test_PoissonRegression():
    """Tests probflow.applications.PoissonRegression"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    y = np.random.poisson(lam=np.exp(y)).astype('float32')

    # Create the model
    model = apps.PoissonRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)



def test_DenseNetwork():
    """Tests probflow.applications.DenseNetwork"""

    class DenseNet(Model):

        def __init__(self, dims):
            self.net = apps.DenseNetwork(dims)

        def __call__(self, x):
            return Normal(self.net(x), 1.0)

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1

    # Create the model
    model = DenseNet([5, 20, 15, 1])

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)



def test_DenseRegression():
    """Tests probflow.applications.DenseRegression"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Create the model
    model = apps.DenseRegression([5, 20, 15, 1])

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)



def test_DenseClassifier():
    """Tests probflow.applications.DenseClassifier"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    y = np.round(1.0/(1.0+np.exp(-y))).astype('float32')

    # Create the model
    model = apps.DenseClassifier([5, 20, 15, 2])

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)



def test_MultinomialDenseClassifier():
    """Tests probflow.applications.DenseClassifier w/ >2 output classes"""

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 3).astype('float32')
    b = np.random.randn(1, 3).astype('float32')
    y = x@w + b
    y = np.argmax(y, axis=1).astype('int32')

    # Create the model
    model = apps.DenseClassifier([5, 20, 15, 3])

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=11)
    
    # Predictive functions
    model.predict(x)
