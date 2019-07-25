"""Probability distributions.

The :mod:`.distributions` module contains classes to instantiate probability
distributions, which describe the likelihood of either a parameter or a
datapoint taking any given value.  Distribution objects are used to represent
both the predicted probability distribution of the data, and also the
parameters' posteriors and priors.

Continuous Distributions
------------------------

* :class:`.Deterministic`
* :class:`.Normal`
* :class:`.StudentT`
* :class:`.Cauchy`
* :class:`.Gamma`
* :class:`.InverseGamma`

Discrete Distributions
----------------------

* :class:`.Bernoulli`
* :class:`.Categorical`
* :class:`.Poisson`
* :class:`.Dirichlet`

----------

"""


__all__ = [
    'Deterministic',
    'Normal',
    'StudentT',
    'Cauchy',
    'Gamma',
    'InverseGamma',
    'Bernoulli',
    'Categorical',
    'Poisson',
    'Dirichlet',
]



import numpy as np

from probflow.core.settings import get_backend
from probflow.core.base import BaseDistribution



def _ensure_tensor_like(obj, name):
    """Determine whether an object can be cast to a Tensor"""
    if get_backend() == 'pytorch':
        import torch
        tensor_types = (torch.Tensor)
    else:
        import tensorflow as tf
        tensor_types = (tf.Tensor, tf.Variable)
    if not isinstance(obj, (int, float, np.ndarray, list)+tensor_types):
        raise TypeError(name+' must be Tensor-like')



class Deterministic(BaseDistribution):
    r"""A deterministic distribution.

    A 
    `deterministic distribution <https://en.wikipedia.org/wiki/Degenerate_distribution>`_
    is a continuous distribution defined over all real numbers, and has one
    parameter: 

    - a location parameter (``loc`` or :math:`k_0`) which determines the mean
      of the distribution.

    A random variable :math:`x` drawn from a deterministic distribution
    has probability of 1 at its location parameter value, and zero elsewhere:

    .. math::

        p(x) = 
        \begin{cases}
            1, & \text{if}~x=k_0 \\
            0, & \text{otherwise}
        \end{cases}

    TODO: example image of the distribution


    Parameters
    ----------
    loc : int, float, |ndarray|, or |Tensor|
        Mean of the deterministic distribution (:math:`k_0`).
        Default = 0
    """


    def __init__(self, loc=0):

        # Check input
        _ensure_tensor_like(loc, 'loc')

        # Store args
        self.loc = loc


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            raise NotImplementedError
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Deterministic(self.loc)



class Normal(BaseDistribution):
    r"""The Normal distribution.

    The 
    `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_
    is a continuous distribution defined over all real numbers, and has two
    parameters: 

    - a location parameter (``loc`` or :math:`\mu`) which determines the mean
      of the distribution, and 
    - a scale parameter (``scale`` or :math:`\sigma > 0`) which determines the
      standard deviation of the distribution.

    A random variable :math:`x` drawn from a normal distribution

    .. math::

        x \sim \mathcal{N}(\mu, \sigma)

    has probability

    .. math::

        p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}}
               \exp \left( -\frac{(x-\mu)^2}{2 \sigma^2} \right)

    TODO: example image of the distribution


    Parameters
    ----------
    loc : int, float, |ndarray|, or |Tensor|
        Mean of the normal distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, or |Tensor|
        Standard deviation of the normal distribution (:math:`\sigma`).
        Default = 1
    """


    def __init__(self, loc=0, scale=1):

        # Check input
        _ensure_tensor_like(loc, 'loc')
        _ensure_tensor_like(scale, 'scale')

        # Store args
        self.loc = loc
        self.scale = scale


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.normal.Normal(self.loc, self.scale)
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Normal(self.loc, self.scale)



class StudentT(BaseDistribution):
    r"""The Student-t distribution.

    The 
    `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_
    is a continuous distribution defined over all real numbers, and has three
    parameters:

    - a degrees of freedom parameter (``df`` or :math:`\nu > 0`), which
      determines how many degrees of freedom the distribution has,
    - a location parameter (``loc`` or :math:`\mu`) which determines the mean
      of the distribution, and
    - a scale parameter (``scale`` or :math:`\sigma > 0`) which determines the 
      standard deviation of the distribution.

    A random variable :math:`x` drawn from a Student's t-distribution

    .. math::

        x \sim \text{StudentT}(\nu, \mu, \sigma)

    has probability

    .. math::

        p(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu x} \Gamma
            (\frac{\nu}{2})} \left( 1 + \frac{x^2}{\nu} \right)^
            {-\frac{\nu+1}{2}}

    Where :math:`\Gamma` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    TODO: example image of the distribution


    Parameters
    ----------
    df : int, float, |ndarray|, or |Tensor|
        Degrees of freedom of the t-distribution (:math:`\nu`).
        Default = 1
    loc : int, float, |ndarray|, or |Tensor|
        Median of the t-distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, or |Tensor|
        Spread of the t-distribution (:math:`\sigma`).
        Default = 1
    """


    def __init__(self, df=1, loc=0, scale=1):

        # Check input
        _ensure_tensor_like(df, 'df')
        _ensure_tensor_like(loc, 'loc')
        _ensure_tensor_like(scale, 'scale')

        # Store args
        self.df = df
        self.loc = loc
        self.scale = scale


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.studentT.StudentT(self.df, self.loc, self.scale)
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.StudentT(self.df, self.loc, self.scale)


    def mean(self):
        """Compute the mean of this distribution.

        Note that the mean of a StudentT distribution is technically 
        undefined when df=1.
        """
        return self.loc



class Cauchy(BaseDistribution):
    r"""The Cauchy distribution.

    The 
    `Cauchy distribution <https://en.wikipedia.org/wiki/Cauchy_distribution>`_
    is a continuous distribution defined over all real numbers, and has two
    parameters: 

    - a location parameter (``loc`` or :math:`\mu`) which determines the
      median of the distribution, and 
    - a scale parameter (``scale`` or :math:`\gamma > 0`) which determines the
      spread of the distribution.

    A random variable :math:`x` drawn from a Cauchy distribution

    .. math::

        x \sim \text{Cauchy}(\mu, \gamma)

    has probability

    .. math::

        p(x) = \frac{1}{\pi \gamma \left[  1 + 
               \left(  \frac{x-\mu}{\gamma} \right)^2 \right]}

    The Cauchy distribution is equivalent to a Student's t-distribution with
    one degree of freedom.

    TODO: example image of the distribution


    Parameters
    ----------
    loc : int, float, |ndarray|, or |Tensor|
        Median of the Cauchy distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, or |Tensor|
        Spread of the Cauchy distribution (:math:`\gamma`).
        Default = 1
    """


    def __init__(self, loc=0, scale=1):

        # Check input
        _ensure_tensor_like(loc, 'loc')
        _ensure_tensor_like(scale, 'scale')

        # Store args
        self.loc = loc
        self.scale = scale


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.cauchy.Cauchy(self.loc, self.scale)
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Cauchy(self.loc, self.scale)


    def mean(self):
        """Compute the mean of this distribution.

        Note that the mean of a Cauchy distribution is technically undefined.
        """
        return self.loc



class Gamma(BaseDistribution):
    r"""The Gamma distribution.

    The 
    `Gamma distribution <https://en.wikipedia.org/wiki/Gamma_distribution>`_
    is a continuous distribution defined over all positive real numbers, and
    has two parameters: 

    - a shape parameter (``shape`` or :math:`\alpha > 0`, a.k.a. 
      "concentration"), and
    - a rate parameter (``rate`` or :math:`\beta > 0`).

    The ratio of :math:`\frac{\alpha}{\beta}` determines the mean of the
    distribution, and the ratio of :math:`\frac{\alpha}{\beta^2}` determines
    the variance.

    A random variable :math:`x` drawn from a Gamma distribution

    .. math::

        x \sim \text{Gamma}(\alpha, \beta)

    has probability

    .. math::

        p(x) = \frac{\beta^\alpha}{\Gamma (\alpha)} x^{\alpha-1}
               \exp (-\beta x)

    Where :math:`\Gamma` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    TODO: example image of the distribution


    Parameters
    ----------
    shape : int, float, |ndarray|, or |Tensor|
        Shape parameter of the gamma distribution (:math:`\alpha`).
    rate : int, float, |ndarray|, or |Tensor|
        Rate parameter of the gamma distribution (:math:`\beta`).

    """

    def __init__(self, concentration, rate):

        # Check input
        _ensure_tensor_like(concentration, 'concentration')
        _ensure_tensor_like(rate, 'rate')

        # Store args
        self.concentration = concentration
        self.rate = rate


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.gamma.Gamma(self.concentration, self.rate)
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Gamma(self.concentration, self.rate) 



class InverseGamma(BaseDistribution):
    r"""The Inverse-gamma distribution.

    The 
    `Inverse-gamma distribution <https://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_
    is a continuous distribution defined over all positive real numbers, and
    has two parameters: 

    - a shape parameter (``shape`` or :math:`\alpha > 0`, a.k.a. 
      "concentration"), and
    - a rate parameter (``rate`` or :math:`\beta > 0`, a.k.a. "scale").

    The ratio of :math:`\frac{\beta}{\alpha-1}` determines the mean of the
    distribution, and for :math:`\alpha > 2`, the variance is determined by:

    .. math ::

        \frac{\beta^2}{(\alpha-1)^2(\alpha-2)}

    A random variable :math:`x` drawn from an Inverse-gamma distribution

    .. math::

        x \sim \text{InvGamma}(\alpha, \beta)

    has probability

    .. math::

        p(x) = \frac{\beta^\alpha}{\Gamma (\alpha)} x^{-\alpha-1}
               \exp (-\frac{\beta}{x})

    Where :math:`\Gamma` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    TODO: example image of the distribution


    Parameters
    ----------
    concentration : int, float, |ndarray|, or |Tensor|
        Shape parameter of the inverse gamma distribution (:math:`\alpha`).
    scale : int, float, |ndarray|, or |Tensor|
        Rate parameter of the inverse gamma distribution (:math:`\beta`).

    """


    def __init__(self, concentration, scale):

        # Check input
        _ensure_tensor_like(concentration, 'concentration')
        _ensure_tensor_like(scale, 'scale')

        # Store args
        self.concentration = concentration
        self.scale = scale


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            raise NotImplementedError
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.InverseGamma(self.concentration, self.scale) 



class Bernoulli(BaseDistribution):
    r"""The Bernoulli distribution.

    The 
    `Bernoulli distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_
    is a discrete distribution defined over only two integers: 0 and 1.
    It has one parameter: 

    - a probability parameter (:math:`0 \leq p \leq 1`).

    A random variable :math:`x` drawn from a Bernoulli distribution

    .. math::

        x \sim \text{Bernoulli}(p)

    takes the value :math`1` with probability :math:`p`, and takes the value
    :math:`0` with probability :math:`p-1`.

    TODO: example image of the distribution

    TODO: specifying either logits or probs


    Parameters
    ----------
    logits : int, float, |ndarray|, or |Tensor|
        Logit-transformed probability parameter of the  Bernoulli 
        distribution (:math:`\p`)
    probs : int, float, |ndarray|, or |Tensor|
        Logit-transformed probability parameter of the  Bernoulli 
        distribution (:math:`\p`)
    """

    def __init__(self, logits=None, probs=None):

        # Check input
        if logits is None and probs is None:
            raise TypeError('either logits or probs must be specified')
        if logits is None:
            _ensure_tensor_like(probs, 'probs')
        if probs is None:
            _ensure_tensor_like(logits, 'logits')

        # Store args
        self.logits = logits
        self.probs = probs


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.bernoulli.Bernoulli(logits=self.logits, probs=self.probs) 
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Bernoulli(logits=self.logits, probs=self.probs) 



class Categorical(BaseDistribution):
    r"""The Categorical distribution.

    The 
    `Categorical distribution <https://en.wikipedia.org/wiki/Categorical_distribution>`_
    is a discrete distribution defined over :math:`N` integers: 0 through 
    :math:`N-1`. A random variable :math:`x` drawn from a Categorical
    distribution

    .. math::

        x \sim \text{Categorical}(\mathbf{\theta})

    has probability

    .. math::

        p(x=i) = p_i

    TODO: example image of the distribution

    TODO: logits vs probs


    Parameters
    ----------
    logits : int, float, |ndarray|, or |Tensor|
        Logit-transformed category probabilities 
        (:math:`\frac{\mathbf{\theta}}{1-\mathbf{\theta}}`)
    probs : int, float, |ndarray|, or |Tensor|
        Raw category probabilities (:math:`\mathbf{\theta}`)
    """

    def __init__(self, logits=None, probs=None):

        # Check input
        if logits is None and probs is None:
            raise TypeError('either logits or probs must be specified')
        if logits is None:
            _ensure_tensor_like(probs, 'probs')
        if probs is None:
            _ensure_tensor_like(logits, 'logits')

        # Store args
        self.logits = logits
        self.probs = probs


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.categorical.Categorical(logits=self.logits,
                                               probs=self.probs) 
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Categorical(logits=self.logits, probs=self.probs) 



class Poisson(BaseDistribution):
    r"""The Poisson distribution.

    The 
    `Poisson distribution <https://en.wikipedia.org/wiki/Poisson_distribution>`_
    is a discrete distribution defined over all non-negativve real integers,
    and has one parameter: 

    - a rate parameter (``rate`` or :math:`\lambda`) which determines the mean
      of the distribution.

    A random variable :math:`x` drawn from a Poisson distribution

    .. math::

        x \sim \text{Poisson}(\lambda)

    has probability

    .. math::

        p(x) = \frac{\lambda^x e^{-\lambda}}{x!}

    TODO: example image of the distribution


    Parameters
    ----------
    rate : int, float, |ndarray|, or |Tensor|
        Rate parameter of the Poisson distribution (:math:`\lambda`).
    """

    def __init__(self, rate):

        # Check input
        _ensure_tensor_like(rate, 'rate')

        # Store args
        self.rate = rate


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.poisson.Poisson(self.rate) 
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Poisson(self.rate) 



class Dirichlet(BaseDistribution):
    r"""The Dirichlet distribution.

    The 
    `Dirichlet distribution <http://en.wikipedia.org/wiki/Dirichlet_distribution>`_
    is a continuous distribution defined over the :math:`k`-simplex, and has
    one vector of parameters: 

    - concentration parameters (``concentration`` or :math:`\mathbf{\alpha}`),
    a vector of positive numbers which determine the relative likelihoods of 
    different categories represented by the distribution.

    A random variable (a vector) :math:`\mathbf{x}` drawn from a Dirichlet
    distribution

    .. math::

        \mathbf{x} \sim \text{Dirichlet}(\mathbf{\alpha})

    has probability

    .. math::

        p(\mathbf{x}) = \frac{1}{\mathbf{\text{B}}(\mathbf{\alpha})} 
                        \prod_{i=1}^K x_i^{\alpha_i-1}

    where :math:`\mathbf{\text{B}}` is the multivariate beta function.

    TODO: example image of the distribution


    Parameters
    ----------
    concentration : |ndarray|, or |Tensor|
        Concentration parameter of the Dirichlet distribution (:math:`\alpha`).
    """

    def __init__(self, concentration):

        # Check input
        _ensure_tensor_like(concentration, 'concentration')

        # Store args
        self.concentration = concentration


    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == 'pytorch':
            import torch.distributions as tod
            return tod.dirichlet.Dirichlet(self.concentration) 
        else:
            from tensorflow_probability import distributions as tfd
            return tfd.Dirichlet(self.concentration) 
