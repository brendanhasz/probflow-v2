"""Tests the probflow.modules module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.core.settings import Sampling
import probflow.core.ops as O
from probflow.parameters import *
from probflow.modules import *



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_Module():
    """Tests the Module abstract base class"""

    class TestModule(Module):

        def __init__(self):
            self.p1 = Parameter(name='TestParam1')
            self.p2 = Parameter(name='TestParam2', shape=[5, 4])

        def __call__(self, x):
            return O.sum(self.p2(), axis=None) + x*self.p1()

    the_module = TestModule()

    # parameters should return a list of all the parameters
    param_list = the_module.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in param_list]
    assert 'TestParam1' in param_names
    assert 'TestParam2' in param_names

    # trainable_variables should return list of all variables in the model
    var_list = the_module.trainable_variables
    assert isinstance(var_list, list)
    assert len(var_list) == 4
    assert all(isinstance(v, tf.Variable) for v in var_list)

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # calling a module should return a tensor
    x = tf.random.normal([5])
    sample1 = the_module(x)
    assert isinstance(sample1, tf.Tensor)
    assert sample1.ndim == 1
    assert sample1.shape[0] == 5

    # should be the same when sampling is off
    sample2 = the_module(x)
    assert np.all(sample1.numpy() == sample2.numpy())

    # outputs should be different when sampling is on
    with Sampling():
        sample1 = the_module(x)
        sample2 = the_module(x)
    assert np.all(sample1.numpy() != sample2.numpy())

    # A second test module which contains sub-modules
    class TestModule2(Module):

        def __init__(self, shape):
            self.mod = TestModule()
            self.p3 = Parameter(name='TestParam3', shape=shape)

        def __call__(self, x):
            return self.mod(x) + O.sum(self.p3(), axis=None)

    the_module = TestModule2([3, 2])

    # parameters should return a list of all the parameters
    param_list = the_module.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 3
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in param_list]
    assert 'TestParam1' in param_names
    assert 'TestParam2' in param_names
    assert 'TestParam3' in param_names

    # trainable_variables should return list of all variables in the model
    var_list = the_module.trainable_variables
    assert isinstance(var_list, list)
    assert len(var_list) == 6
    assert all(isinstance(v, tf.Variable) for v in var_list)

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # parent module's loss should be greater than child module's
    assert the_module.kl_loss().numpy() > the_module.mod.kl_loss().numpy()

    # calling a module should return a tensor
    x = tf.random.normal([5])
    sample1 = the_module(x)
    assert isinstance(sample1, tf.Tensor)
    assert sample1.ndim == 1
    assert sample1.shape[0] == 5

    # of the appropriate size
    x = tf.random.normal([5, 4])
    sample1 = the_module(x)
    assert isinstance(sample1, tf.Tensor)
    assert sample1.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 4



def test_Dense():
    """Tests probflow.modules.Dense"""

    # Should error w/ int < 1
    with pytest.raises(ValueError):
        dense = Dense(0, 1)
    with pytest.raises(ValueError):
        dense = Dense(5, -1)

    # Create the module
    dense = Dense(5, 1)

    # Test MAP outputs are same
    x = tf.random.normal([4, 5])
    samples1 = dense(x)
    samples2 = dense(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # Test samples are different
    with Sampling():
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # parameters should return [weights, bias]
    param_list = dense.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in dense.parameters]
    assert 'Dense_weights' in param_names
    assert 'Dense_bias' in param_names
    weights = [p for p in dense.parameters if p.name=='Dense_weights']
    assert weights[0].shape == [5, 1]
    bias = [p for p in dense.parameters if p.name=='Dense_bias']
    assert bias[0].shape == [1, 1]

    # kl_loss should return sum of KL losses
    kl_loss = dense.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # test Flipout
    with Sampling(flipout=True):
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1



def test_Sequential():
    """Tests probflow.modules.Sequential"""

    # Create the module
    seq = Sequential([
        Dense(5, 10),
        tf.nn.relu,
        Dense(10, 3),
        tf.nn.relu,
        Dense(3, 1),
    ])

    # Steps should be list
    assert isinstance(seq.steps, list)
    assert len(seq.steps) == 5

    # Test MAP outputs are the same
    x = tf.random.normal([4, 5])
    samples1 = seq(x)
    samples2 = seq(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # Test samples are different
    with Sampling():
        samples1 = seq(x)
        samples2 = seq(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # parameters should return list of all parameters
    param_list = seq.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 6
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in seq.parameters]
    assert 'Dense_weights' in param_names
    assert 'Dense_bias' in param_names
    param_shapes = [p.shape for p in seq.parameters]
    assert [5, 10] in param_shapes
    assert [1, 10] in param_shapes
    assert [10, 3] in param_shapes
    assert [1, 3] in param_shapes
    assert [3, 1] in param_shapes
    assert [1, 1] in param_shapes

    # kl_loss should return sum of KL losses
    kl_loss = seq.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0



def test_BatchNormalization():
    """Tests probflow.modules.BatchNormalization"""

    # Create the module
    bn = BatchNormalization([5])

    # Test MAP outputs are the same
    x = tf.random.normal([4, 5])
    samples1 = bn(x)
    samples2 = bn(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 5

    # Samples should actually be the same b/c using deterministic posterior
    with Sampling():
        samples1 = bn(x)
        samples2 = bn(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 5

    # parameters should return list of all parameters
    param_list = bn.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in bn.parameters]
    assert 'BatchNormalization_weight' in param_names
    assert 'BatchNormalization_bias' in param_names
    param_shapes = [p.shape for p in bn.parameters]
    assert [5] in param_shapes

    # kl_loss should return sum of KL losses
    kl_loss = bn.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # Test it works w/ dense layer and sequential
    seq = Sequential([
        Dense(5, 10),
        BatchNormalization(10),
        tf.nn.relu,
        Dense(10, 3),
        BatchNormalization(3),
        tf.nn.relu,
        Dense(3, 1),
    ])
    assert len(seq.parameters) == 10



def test_Embedding():
    """Tests probflow.modules.Embedding"""

    # Should error w/ int < 1
    with pytest.raises(ValueError):
        emb = Embedding(0, 1)
    with pytest.raises(ValueError):
        emb = Embedding(5, -1)

    # Create the module
    emb = Embedding(10, 5)

    # Check parameters
    assert len(emb.parameters) == 1
    assert emb.parameters[0].name == 'Embeddings'
    assert emb.parameters[0].shape == [10, 5]

    # Test MAP outputs are the same
    x = tf.random.uniform([20], minval=0, maxval=9, dtype=tf.dtypes.int32)
    samples1 = emb(x)
    samples2 = emb(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5

    # Samples should actually be the same b/c using deterministic posterior
    with Sampling():
        samples1 = emb(x)
        samples2 = emb(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5

    # kl_loss should return sum of KL losses
    kl_loss = emb.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

