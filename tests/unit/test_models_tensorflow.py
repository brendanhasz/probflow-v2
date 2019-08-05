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
from probflow.data import DataGenerator



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_Model_0D():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter(name='Weight')
            self.bias = Parameter(name='Bias')
            self.std = ScaleParameter(name='Std')

        def __call__(self, x):
            return Normal(x*self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Fit the model
    x = np.random.randn(100).astype('float32')
    y = -x + 1
    my_model.fit(x, y, batch_size=5, epochs=10)

    # Shouldn't be training
    assert my_model._is_training is False

    # Should be able to set learning rate
    lr = my_model._learning_rate
    my_model.set_learning_rate(lr+1.0)
    assert lr != my_model._learning_rate

    # predictive samples
    samples = my_model.predictive_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    
    # aleatoric samples
    samples = my_model.aleatoric_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # epistemic samples
    samples = my_model.epistemic_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # predict
    samples = my_model.predict(x[:30])
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 30

    # metric
    metric = my_model.metric('mae', x[:30], y[:30])
    assert isinstance(metric, np.floating)
    metric = my_model.metric('mse', x[:30], y[:30])
    assert isinstance(metric, np.floating)
    assert metric >= 0
    
    # posterior_mean w/ no args should return all params
    val = my_model.posterior_mean()
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    
    # posterior_mean w/ str should return value of that param
    val = my_model.posterior_mean('Weight')
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    
    # posterior_mean w/ list of params should return only those params
    val = my_model.posterior_mean(['Weight', 'Std'])
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)

    # posterior_sample w/ no args should return all params
    val = my_model.posterior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    assert all(val[v].shape[1] == 1 for v in val)
    
    # posterior_sample w/ str should return sample of that param
    val = my_model.posterior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 2
    assert val.shape[0] == 20
    assert val.shape[1] == 1
    
    # posterior_sample w/ list of params should return only those params
    val = my_model.posterior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    assert all(val[v].shape[1] == 1 for v in val)

    # posterior_ci should return confidence intervals of all params by def
    val = my_model.posterior_ci(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 1 for v in val)
    assert all(val[v][1].ndim == 1 for v in val)
    assert all(val[v][0].shape[0] == 1 for v in val)
    assert all(val[v][1].shape[0] == 1 for v in val)

    # posterior_ci should return ci of only 1 if passed str
    val = my_model.posterior_ci('Weight', n=20)
    assert isinstance(val, tuple)
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)

    # posterior_ci should return specified cis if passed list of params
    val = my_model.posterior_ci(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 1 for v in val)
    assert all(val[v][1].ndim == 1 for v in val)
    assert all(val[v][0].shape[0] == 1 for v in val)
    assert all(val[v][1].shape[0] == 1 for v in val)

    # prior_sample w/ no args should return all params
    val = my_model.prior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    
    # prior_sample w/ str should return sample of that param
    val = my_model.prior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    assert val.shape[0] == 20
    
    # prior_sample w/ list of params should return only those params
    val = my_model.prior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)

    # log_prob should return log prob of each sample by default
    probs = my_model.log_prob(x[:30], y[:30])
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 30

    # log_prob should return sum if individually = False
    s_prob = my_model.log_prob(x[:30], y[:30], individually=False)
    assert isinstance(s_prob, np.floating)
    assert s_prob == np.sum(probs)

    # log_prob should return samples w/ distribution = True
    probs = my_model.log_prob(x[:30], y[:30], n=10, distribution=True)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 30
    assert probs.shape[1] == 10

    # log_prob should return samples w/ distribution = True
    probs = my_model.log_prob(x[:30], y[:30], n=10,
                               distribution=True, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 10

    # prob should return prob of each sample by default
    probs = my_model.prob(x[:30], y[:30])
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 30
    assert np.all(probs >= 0)

    # prob should return sum if individually = False
    s_prob = my_model.prob(x[:30], y[:30], individually=False)
    assert isinstance(s_prob, np.floating)

    # prob should return samples w/ distribution = True
    probs = my_model.prob(x[:30], y[:30], n=10, distribution=True)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 30
    assert probs.shape[1] == 10
    assert np.all(probs >= 0)

    # prob should return samples w/ distribution = True
    probs = my_model.prob(x[:30], y[:30], n=10,
                               distribution=True, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 10
    assert np.all(probs >= 0)



def test_Model_DataGenerators():
    """Tests the probflow.models.Model sampling/predictive methods when
    passed DataGenerators"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter(name='Weight')
            self.bias = Parameter(name='Bias')
            self.std = ScaleParameter(name='Std')

        def __call__(self, x):
            return Normal(x*self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Make a DataGenerator
    x = np.random.randn(100).astype('float32')
    y = -x + 1
    data = DataGenerator(x, y, batch_size=5)

    # Fit the model
    my_model.fit(data, epochs=10)

    # predictive samples
    samples = my_model.predictive_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100
    
    # aleatoric samples
    samples = my_model.aleatoric_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # epistemic samples
    samples = my_model.epistemic_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # predict
    samples = my_model.predict(data)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 100

    # metric
    metric = my_model.metric('mae', data)
    assert isinstance(metric, np.floating)
    metric = my_model.metric('mse', data)
    assert isinstance(metric, np.floating)
    assert metric >= 0
    


def test_Model_1D():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')
            self.std = ScaleParameter([1, 1], name='Std')

        def __call__(self, x):
            return Normal(x@self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=10)

    # predictive samples
    samples = my_model.predictive_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1
    
    # aleatoric samples
    samples = my_model.aleatoric_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # epistemic samples
    samples = my_model.epistemic_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # predict
    samples = my_model.predict(x[:30, :])
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 30
    assert samples.shape[1] == 1

    # metric
    metric = my_model.metric('mse', x[:30, :], y[:30, :])
    assert isinstance(metric, np.floating)
    metric = my_model.metric('mae', x[:30, :], y[:30, :])
    assert isinstance(metric, np.floating)
    assert metric >= 0
    
    # posterior_mean w/ no args should return all params
    val = my_model.posterior_mean()
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert val['Weight'].shape[0] == 5
    assert val['Weight'].shape[1] == 1
    assert val['Bias'].shape[0] == 1
    assert val['Bias'].shape[1] == 1
    assert val['Std'].shape[0] == 1
    assert val['Std'].shape[1] == 1
    
    # posterior_mean w/ str should return value of that param
    val = my_model.posterior_mean('Weight')
    assert isinstance(val, np.ndarray)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 1
    
    # posterior_mean w/ list of params should return only those params
    val = my_model.posterior_mean(['Weight', 'Std'])
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert val['Weight'].shape[0] == 5
    assert val['Weight'].shape[1] == 1
    assert val['Std'].shape[0] == 1
    assert val['Std'].shape[1] == 1

    # posterior_sample w/ no args should return all params
    val = my_model.posterior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 3 for v in val)
    assert val['Weight'].shape[0] == 20
    assert val['Weight'].shape[1] == 5
    assert val['Weight'].shape[2] == 1
    assert val['Bias'].shape[0] == 20
    assert val['Bias'].shape[1] == 1
    assert val['Bias'].shape[2] == 1
    assert val['Std'].shape[0] == 20
    assert val['Std'].shape[1] == 1
    assert val['Std'].shape[2] == 1
    
    # posterior_sample w/ str should return sample of that param
    val = my_model.posterior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 3
    assert val.shape[0] == 20
    assert val.shape[1] == 5
    assert val.shape[2] == 1
    
    # posterior_sample w/ list of params should return only those params
    val = my_model.posterior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 3 for v in val)
    assert val['Weight'].shape[0] == 20
    assert val['Weight'].shape[1] == 5
    assert val['Weight'].shape[2] == 1
    assert val['Std'].shape[0] == 20
    assert val['Std'].shape[1] == 1
    assert val['Std'].shape[2] == 1

    # posterior_ci should return confidence intervals of all params by def
    val = my_model.posterior_ci(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 2 for v in val)
    assert all(val[v][1].ndim == 2 for v in val)
    for i in range(1):
        assert val['Weight'][i].shape[0] == 5
        assert val['Weight'][i].shape[1] == 1
        assert val['Bias'][i].shape[0] == 1
        assert val['Bias'][i].shape[1] == 1
        assert val['Std'][i].shape[0] == 1
        assert val['Std'][i].shape[1] == 1

    # posterior_ci should return ci of only 1 if passed str
    val = my_model.posterior_ci('Weight', n=20)
    assert isinstance(val, tuple)
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)

    # posterior_ci should return specified cis if passed list of params
    val = my_model.posterior_ci(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 2 for v in val)
    assert all(val[v][1].ndim == 2 for v in val)
    for i in range(1):
        assert val['Weight'][i].shape[0] == 5
        assert val['Weight'][i].shape[1] == 1
        assert val['Std'][i].shape[0] == 1
        assert val['Std'][i].shape[1] == 1
    
    # prior_sample w/ no args should return all params
    val = my_model.prior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    
    # prior_sample w/ str should return sample of that param
    val = my_model.prior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    assert val.shape[0] == 20
    
    # prior_sample w/ list of params should return only those params
    val = my_model.prior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)



def test_Model_nesting():
    """Tests Model when it contains Modules and sub-modules"""

    class MyModule(Module):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')

        def __call__(self, x):
            return x@self.weight() + self.bias()

    class MyModel(Model):

        def __init__(self):
            self.module = MyModule()
            self.std = ScaleParameter([1, 1], name='Std')

        def __call__(self, x):
            return Normal(self.module(x), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=10)

    # predictive samples
    samples = my_model.predictive_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # kl loss should be greater for outer model
    assert my_model.kl_loss().numpy() > my_model.module.kl_loss().numpy()



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

