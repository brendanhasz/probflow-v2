"""Backend-specific operations

The core.ops module contains operations which run using the current backend.

* :func:`.kl_divergence`
* :func:`.ones`
* :func:`.zeros`
* :func:`.sum`
* :func:`.prod`
* :func:`.mean`
* :func:`.std`
* :func:`.round`
* :func:`.abs`
* :func:`.square`
* :func:`.exp`
* :func:`.sqrt`
* :func:`.relu`
* :func:`.softplus`
* :func:`.sigmoid`
* :func:`.gather`

"""



__all__ = [
    'kl_divergence',
    'ones',
    'zeros',
    'sum',
    'prod',
    'mean',
    'std',
    'round',
    'abs',
    'square',
    'exp',
    'sqrt',
    'relu',
    'softplus',
    'sigmoid',
    'gather',
]



from probflow.core.settings import get_backend



def kl_divergence(P, Q):
    """Compute the Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    P : |tfp.Distribution| or |torch.Distribution|
        The first distribution
    Q : |tfp.Distribution| or |torch.Distribution|
        The second distribution

    Returns
    -------
    kld : Tensor
        The Kullback–Leibler divergence between P and Q (KL(P || Q))
    """
    if get_backend() == 'pytorch':
        import torch
        return torch.distributions.kl.kl_divergence(P, Q)
    else:
        import tensorflow_probability as tfp
        return tfp.distributions.kl_divergence(P, Q)



# TODO: all other ops used by probflow internals



def expand_dims(val, axis):
    """Add a singular dimension to a Tensor"""
    if get_backend() == 'pytorch':
        import torch
        return torch.unsqueeze(val, axis)
    else:
        import tensorflow as tf
        return tf.expand_dims(val, axis)



def ones(shape):
    """Tensor full of ones."""
    if get_backend() == 'pytorch':
        import torch
        return torch.ones(shape)
    else:
        import tensorflow as tf
        return tf.ones(shape)



def zeros(shape):
    """Tensor full of zeros."""
    if get_backend() == 'pytorch':
        import torch
        return torch.zeros(shape)
    else:
        import tensorflow as tf
        return tf.zeros(shape)



def sum(val, axis=-1):
    """The sum."""
    if get_backend() == 'pytorch':
        import torch
        return torch.sum(val, dim=axis)
    else:
        import tensorflow as tf
        return tf.reduce_sum(val, axis=axis)



def prod(val, axis=-1):
    """The product."""
    if get_backend() == 'pytorch':
        import torch
        return torch.prod(val, dim=axis)
    else:
        import tensorflow as tf
        return tf.reduce_prod(val, axis=axis)



def mean(val, axis=-1):
    """The mean."""
    if get_backend() == 'pytorch':
        import torch
        return torch.mean(val, dim=axis)
    else:
        import tensorflow as tf
        return tf.reduce_mean(val, axis=axis)



def std(val, axis=-1):
    """The uncorrected sample standard deviation."""
    if get_backend() == 'pytorch':
        import torch
        return torch.std(val, dim=axis)
    else:
        import tensorflow as tf
        return tf.math.reduce_std(val, axis=axis)



def round(val):
    """Round to the closest integer"""
    if get_backend() == 'pytorch':
        import torch
        return torch.round(val)
    else:
        import tensorflow as tf
        return tf.math.round(val)



def abs(val):
    """Absolute value"""
    if get_backend() == 'pytorch':
        import torch
        return torch.abs(val)
    else:
        import tensorflow as tf
        return tf.math.abs(val)



def square(val):
    """Power of 2"""
    if get_backend() == 'pytorch':
        import torch
        return val**2
    else:
        import tensorflow as tf
        return tf.math.square(val)



def sqrt(val):
    """The square root."""
    if get_backend() == 'pytorch':
        import torch
        return torch.sqrt(val)
    else:
        import tensorflow as tf
        return tf.sqrt(val)



def exp(val):
    """The natural exponent."""
    if get_backend() == 'pytorch':
        import torch
        return torch.exp(val)
    else:
        import tensorflow as tf
        return tf.exp(val)



def relu(val):
    """Linear rectification."""
    if get_backend() == 'pytorch':
        import torch
        return torch.nn.ReLU()(val)
    else:
        import tensorflow as tf
        return tf.nn.relu(val)



def softplus(val):
    """Linear rectification."""
    if get_backend() == 'pytorch':
        import torch
        return torch.nn.Softplus()(val)
    else:
        import tensorflow as tf
        return tf.math.softplus(val)



def sigmoid(val):
    """Sigmoid function."""
    if get_backend() == 'pytorch':
        import torch
        return torch.nn.Sigmoid()(val)
    else:
        import tensorflow as tf
        return tf.math.sigmoid(val)



def gather(vals, inds, axis=0):
    """Gather values by index"""
    if get_backend() == 'pytorch':
        import torch
        return torch.gather(vals, axis, inds)
    else:
        import tensorflow as tf
        return tf.gather(vals, inds, axis=axis)



def additive_logistic_transform(vals):
    """The additive logistic transformation"""
    # TODO: is this used?
    if get_backend() == 'pytorch':
        import torch
        raise NotImplementedError
    else:
        import tensorflow as tf
        ones_shape = tf.concat([vals.shape[:-1], [1]], axis=-1)
        exp_vals = tf.concat([tf.exp(vals), tf.ones(ones_shape)], axis=-1)
        return exp_vals/tf.reduce_sum(exp_vals, axis=-1, keepdims=True)



def add_col_of(vals, val):
    """Add a column of a value to a tensor"""
    if get_backend() == 'pytorch':
        import torch
        raise NotImplementedError
    else:
        import tensorflow as tf
        ones_shape = tf.concat([vals.shape[:-1], [1]], axis=-1)
        return tf.concat([vals, val*tf.ones(ones_shape)], axis=-1)
