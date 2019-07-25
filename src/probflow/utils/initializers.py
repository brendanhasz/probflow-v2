"""Initializers.

Functions to initialize posterior distribution variables.

* :func:`.xavier` - Xavier initializer
* :func:`.scale_xavier` - Xavier initializer scaled for scale parameters
* :func:`.pos_xavier` - positive-only initizlier

----------

"""



import numpy as np

from probflow.core.settings import get_backend
from probflow.core.settings import get_datatype



def xavier(shape):
    """Xavier initializer

    """
    scale = np.sqrt(2/sum(shape))
    if get_backend() == 'pytorch':
        # TODO: use truncated normal for torch
        import torch
        return torch.randn(shape, dtype=get_datatype()) * scale
    else:
        import tensorflow as tf
        return tf.random.truncated_normal(shape, mean=0.0, stddev=scale,
                                          dtype=get_datatype())



def scale_xavier(shape):
    """Xavier initializer for scale variables"""
    vals = xavier(shape)
    if get_backend() == 'pytorch':
        import torch
        numel = torch.prod(shape)
        return vals+2-2*torch.log(numel)/torch.log(10.0)
    else:
        import tensorflow as tf
        numel = float(tf.reduce_prod(shape))
        return vals+2-2*tf.math.log(numel)/tf.math.log(10.0)



def pos_xavier(shape):
    """Xavier initializer for positive variables"""
    vals = xavier(shape)
    if get_backend() == 'pytorch':
        import torch
        numel = torch.prod(shape)
        return vals + torch.log(numel)/torch.log(10.0)
    else:
        import tensorflow as tf
        numel = float(tf.reduce_prod(shape))
        return vals + tf.math.log(numel)/tf.math.log(10.0)
