"""Global settings

The core.settings module contains global settings about the backend to use,
what sampling method to use, the default device, and default datatype.


Backend
-------

Which backend to use.  Can be either 
`TensorFlow 2.0 <http://www.tensorflow.org/beta/>`_ 
or `PyTorch <http://pytorch.org/>`_.

* :func:`.get_backend`
* :func:`.set_backend`


Samples
-------

Whether and how many samples to draw from parameter posterior distributions.
If ``None``, the maximum a posteriori estimate of each parameter will be used.
If an integer greater than 0, that many samples from each parameter's posterior
distribution will be used.

* :func:`.get_samples`
* :func:`.set_samples`


"""


__all__ = [
    'get_backend',
    'set_backend',
    'get_samples',
    'set_samples',
    'get_flipout',
    'set_flipout',
    'Sampling',
]



# What backend to use
_BACKEND = 'tensorflow' #or pytorch



# Whether to sample from Parameter posteriors or use MAP estimates
_SAMPLES = None



# Whether to use flipout where possible
_FLIPOUT = False



def get_backend():
    return _BACKEND



def set_backend(backend):
    if isinstance(backend, str):
        if backend in ['tensorflow', 'pytorch']:
            _BACKEND = backend
        else:
            raise ValueError('backend must be either tensorflow or pytorch')
    else:
        raise ValueError('backend must be a string')



def get_samples():
    return _SAMPLES



def set_samples(samples):
    if samples is not None and not isinstance(samples, int):
        raise TypeError('samples must be an int or None')
    elif isinstance(samples, int) and samples < 1:
        raise ValueError('samples must be positive')
    else:
        _SAMPLES = samples



def get_flipout():
    return _FLIPOUT



def set_flipout(flipout):
    if isinstance(flipout, bool):
        _FLIPOUT = flipout
    else:
        raise TypeError('flipout must be True or False')



class Sampling():
    """Use sampling while within this context manager."""


    def __enter__(self, n=1, flipout=False):
        """Begin sampling.

        Keyword Arguments
        -----------------
        n : None or int > 0
            Number of samples (if any) to draw from parameters' posteriors.
            Default = 1
        flipout : bool
            Whether to use flipout where possible while sampling.
            Default = False
        """
        set_samples(n)
        set_flipout(flipout)


    def __exit__(self, _type, _val, _tb):
        """End sampling."""
        set_samples(None)
        set_flipout(False)



# TODO also setting sampling flag might be a problem when using @tf.function
# or pytorch jit/@script?
