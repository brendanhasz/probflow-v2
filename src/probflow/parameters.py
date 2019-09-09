"""Parameters.

Parameters are values which characterize the behavior of a model.  When
fitting a model, we want to find the values of the parameters which
best allow the model to explain the data.  However, with Bayesian modeling
we want not only to find the single *best* value for each parameter, but a 
probability distribution which describes how likely any given value of 
a parameter is to be the best or true value.

Parameters have both priors (probability distributions which describe how
likely we think different values for the parameter are *before* taking into
consideration the current data), and posteriors (probability distributions 
which describe how likely we think different values for the parameter are
*after* taking into consideration the current data).  The prior is set 
to a specific distribution before fitting the model.  While the *type* of 
distribution used for the posterior is set before fitting the model, the 
shape of that distribution (the value of the parameters which define the
distribution) is optimized while fitting the model.
See the :ref:`math` section for more info.

The :class:`.Parameter` class can be used to create any probabilistic
parameter. 

For convenience, ProbFlow also includes some classes which are special cases
of a :class:`.Parameter`:

* :class:`.ScaleParameter` - standard deviation parameter
* :class:`.CategoricalParameter` - categorical parameter
* :class:`.BoundedParameter` - parameter which is bounded between 0 and 1
* :class:`.PositiveParameter` - parameter which is always greater than 0

----------

"""


__all__ = [
    'Parameter',
    'ScaleParameter',
    'CategoricalParameter',
    'DirichletParameter',
    'BoundedParameter',
    'PositiveParameter',
    'DeterministicParameter',
]



from typing import Union, List, Dict, Type, Callable

import numpy as np
import matplotlib.pyplot as plt

from probflow.core.settings import get_samples
from probflow.core.settings import get_backend
from probflow.core.settings import Sampling
from probflow.core.base import BaseParameter
from probflow.core.base import BaseDistribution
import probflow.core.ops as O
from probflow.distributions import Normal
from probflow.distributions import Gamma
from probflow.distributions import Categorical
from probflow.distributions import Dirichlet
from probflow.distributions import Deterministic
from probflow.utils.plotting import plot_dist
from probflow.utils.initializers import xavier
from probflow.utils.initializers import scale_xavier
from probflow.utils.initializers import pos_xavier



class Parameter(BaseParameter):
    r"""Probabilistic parameter(s).

    A probabilistic parameter :math:`\beta`.  The default posterior
    distribution is the :class:`.Normal` distribution, and the default prior
    is a :class:`.Normal` distribution with a mean of 0 and a standard
    deviation of 1.

    The prior for a |Parameter| can be set to any |Distribution| object
    (via the ``prior`` argument), and the type of distribution to use for the
    posterior can be set to any |Distribution| class (using the ``posterior``
    argument).

    The parameter can be given a specific name using the ``name`` argument.
    This makes it easier to access specific parameters after fitting the
    model (e.g. in order to view the posterior distribution).

    The number of independent parameters represented by this 
    :class:`.Parameter` object can be set using the ``shape`` argument.  For 
    example, to create a vector of 5 parameters, set ``shape=5``, or to create
    a 20x7 matrix of parameters set ``shape=[20,7]``.


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0,1)``
    transform : callable
        Transform to apply to the random variable.  For example, to create a
        parameter with an inverse gamma posterior, use
        ``posterior``=:class:`.Gamma`` and
        ``transform = lambda x: tf.reciprocal(x)``
        Default is to use no transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
        For example to transform the standard deviation parameter from 
        untransformed space to transformed, positive, space, use
        ``initializer={'scale': tf.random.randn}`` and
        ``var_transform={'scale': tf.nn.softplus}``
    name : str
        Name of the parameter(s).
        Default = ``'Parameter'``


    Attributes
    ----------
    initializer : Dict[str, callable]
        Initializer functions for each variable
    name : str
        Name of this |Parameter|
    posterior_fn : |Distribution| class
        Distribution to use for the variational posterior
    posterior : |Distribution| object
        This Parameter's variational posterior
    prior : |Distribution| object
        This parameter's prior
    shape : List[int]
        Shape of this parameter
    trainable_variables : List[Tensor]
        List of raw variable objects from the backend used for this Parameter
    transform : callable
        Transformation to apply to this parameter's variational distribution
    untransformed_variables : Dict[str, Tensor]
        Untransformed variables from the backend
    var_transform : Dict[str, callable]
        Transformations to apply to each variable
    variables : Dict[str, Tensor]
        Transformed variables from the backend
    

    Methods
    -------
    __init__(...)
        Instantiate a Parameter array.
    __call__
        Return a sample from the 
    kl_loss
    posterior_ci
    posterior_mean
    posterior_plot
    posterior_sample
    prior_plot
    prior_sample


    Examples
    --------

    TODO: creating variable

    TODO: creating variable w/ beta posterior

    TODO: plotting posterior dist

    """

    def __init__(self,
                 shape: Union[int, List[int]] = 1,
                 posterior: Type[BaseDistribution] = Normal,
                 prior: BaseDistribution = Normal(0, 1),
                 transform: Callable = lambda x: x,
                 initializer: Dict[str, Callable] = {'loc': xavier,
                                                     'scale': scale_xavier},
                 var_transform : Dict[str, Callable] = {'loc': lambda x: x,
                                                        'scale': O.softplus},
                 name: str = 'Parameter'):

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Check values
        if any(e<1 for e in shape):
            raise ValueError('all shapes must be >0')

        # Assign attributes
        self.shape = shape
        self.posterior_fn = posterior
        self.prior = prior
        self.transform = transform
        self.initializer = initializer
        self.var_transform = var_transform
        self.name = name

        # Create variables for the variational distribution
        self.untransformed_variables = dict()
        for var, init in initializer.items():
            if get_backend() == 'pytorch':
                self.untransformed_variables[var] = init(shape)
                self.untransformed_variables[var].requires_grad = True
            else:
                import tensorflow as tf
                self.untransformed_variables[var] = tf.Variable(init(shape))


    @property
    def trainable_variables(self):
        """Get a list of trainable variables from the backend"""
        return [e for _, e in self.untransformed_variables.items()]


    @property
    def variables(self):
        """Variables after applying their respective transformations"""
        return {name: self.var_transform[name](val)
                for name, val in self.untransformed_variables.items()}


    @property
    def posterior(self):
        """This Parameter's variational posterior distribution"""
        return self.posterior_fn(**self.variables)


    def __call__(self):
        """Return a sample from or the MAP estimate of this parameter.

        TODO

        Returns
        -------
        sample : Tensor
            A sample from this Parameter's variational posterior distribution
        """
        n_samples = get_samples()
        if n_samples is None:
            return self.transform(self.posterior.mean())
        elif n_samples == 1:
            return self.transform(self.posterior.sample())
        else:
            return self.transform(self.posterior.sample(n_samples))


    def kl_loss(self):
        """Compute the sum of the Kullback–Leibler divergences between this
        parameter's priors and its variational posteriors."""
        return O.sum(O.kl_divergence(self.posterior(), self.prior()),
                     axis=None)


    def posterior_mean(self):
        """Get the mean of the posterior distribution(s).

        TODO
        """
        return self().numpy()


    def posterior_sample(self, n: int = 1):
        """Sample from the posterior distribution.

        Parameters
        ----------
        n : int > 0
            Number of samples to draw from the posterior distribution.
            Default = 1

        Returns
        -------
        TODO
        """
        with Sampling(n=n):
            return self().numpy()


    def prior_sample(self, n: int = 1):
        """Sample from the prior distribution.


        Parameters
        ----------
        n : int > 0
            Number of samples to draw from the prior distribution.
            Default = 1


        Returns
        -------
        |ndarray|
            Samples from the parameter prior distribution.  If ``n>1`` of size
            ``(num_samples, self.prior.shape)``.  If ``n==1```, of size
            ``(self.prior.shape)``.
        """
        if n==1:
            return self.transform(self.prior.sample()).numpy()
        else:
            return self.transform(self.prior.sample(n)).numpy()


    def posterior_ci(self, ci: float = 0.95, n: int = 10000):
        """Posterior confidence intervals

        Parameters
        ----------
        ci : float
            Confidence interval for which to compute the upper and lower
            bounds.  Must be between 0 and 1.
            Default = 0.95
        n : int
            Number of samples to draw from the posterior distributions for
            computing the confidence intervals
            Default = 10,000

        Returns
        -------
        lb : float or |ndarray|
            Lower bound of the confidence interval
        ub : float or |ndarray|
            Upper bound of the confidence interval
        """

        # Check values
        if ci<0.0 or ci>1.0:
            raise ValueError('ci must be between 0 and 1')
        if n < 1:
            raise ValueError('n must be positive')

        # Sample from the posterior
        samples = self.posterior_sample(n=n)

        # Compute confidence intervals
        ci0 = 100 * (0.5 - ci/2.0)
        ci1 = 100 * (0.5 + ci/2.0)
        bounds = np.percentile(samples, q=[ci0, ci1], axis=0)
        return bounds[0, ...], bounds[1, ...]


    def posterior_plot(self, n=10000, style='fill', bins=20, ci=0.0,
                       bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the posterior distribution.

        Parameters
        ----------
        n : int
            Number of samples to take from each posterior distribution for
            estimating the density.  Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        alpha : float between 0 and 1
            Transparency of fill/histogram
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        """

        # Check inputs
        if not isinstance(n, int):
            raise TypeError('n must be an int')
        if n < 1:
            raise ValueError('n must be positive')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if not isinstance(ci, float):
            raise TypeError('ci must be a float')
        if ci<0.0 or ci>1.0:
            raise ValueError('ci must be between 0 and 1')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be between 0 and 1')

        # Sample from the posterior
        samples = self.posterior_sample(n=n)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel=self.name, style=style, bins=bins, 
                  ci=ci, bw=bw, alpha=alpha, color=color)

        # Label with parameter name
        plt.xlabel(self.name)


    def prior_plot(self, n=10000, style='fill', bins=20, ci=0.0,
                   bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the prior distribution.

        Parameters
        ----------
        n : int
            Number of samples to take from each prior distribution for
            estimating the density.  Default = 1000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        bins : int or list or |ndarray|
            Number of bins to use for the prior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        alpha : float between 0 and 1
            Transparency of fill/histogram
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        """

        # Check inputs
        if not isinstance(n, int):
            raise TypeError('n must be an int')
        if n < 1:
            raise ValueError('n must be positive')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if not isinstance(ci, float):
            raise TypeError('ci must be a float')
        if ci<0.0 or ci>1.0:
            raise ValueError('ci must be between 0 and 1')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be between 0 and 1')

        # Sample from the posterior
        samples = self.prior_sample(n=n)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel=self.name, style=style, bins=bins, 
                  ci=ci, bw=bw, alpha=alpha, color=color)

        # Label with parameter name
        plt.xlabel(self.name+' prior')


class ScaleParameter(Parameter):
    r"""Standard deviation parameter.

    This is a convenience class for creating a standard deviation parameter
    (:math:`\sigma`).  It is created by first constructing a variance 
    parameter (:math:`\sigma^2`) which uses an inverse gamma distribution as
    the variational posterior.

    .. math::

        \frac{1}{\sigma^2} \sim \text{Gamma}(\alpha, \beta)

    Then the variance is transformed into the standard deviation:

    .. math::

        \sigma = \sqrt{\sigma^2}

    By default, an inverse gamma prior is used:

        \frac{1}{\sigma^2} \sim \text{Gamma}(5, 5)


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.InverseGamma`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.InverseGamma` ``(5, 5)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use a square root transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'ScaleParameter'``

    Examples
    --------

    Use :class:`.ScaleParameter` to create a standard deviation parameter
    for a :class:`.Normal` distribution::

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=Gamma,
                 prior=Gamma(5, 5),
                 transform=lambda x: O.sqrt(1.0/x),
                 initializer={'concentration': pos_xavier, 
                              'rate': pos_xavier},
                 var_transform={'concentration': O.exp,
                                'rate': O.exp},
                 name='ScaleParameter'):
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class CategoricalParameter(Parameter):
    r"""Categorical parameter.

    This is a convenience class for creating a categorical parameter 
    :math:`\beta` with a Categorical posterior:

    .. math::

        \beta \sim \text{Categorical}(\mathbf{\theta})

    By default, a uniform prior is used.

    TODO: explain that a sample is an int in [0, k-1]


    Parameters
    ----------
    k : int > 2
        Number of categories.
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Categorical`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Categorical` ``(1/k)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use no transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'CategoricalParameter'``


    Examples
    --------

    TODO: creating variable

    """

    def __init__(self,
                 k: int = 2,
                 shape: Union[int, List[int]] = [],
                 posterior=Categorical,
                 prior=None,
                 transform=lambda x: x,
                 initializer={'probs': xavier},
                 var_transform={'probs': O.additive_logistic_transform},
                 name='CategoricalParameter'):

        # Check type of k
        if not isinstance(k, int):
            raise TypeError('k must be an integer')
        if k<2:
            raise ValueError('k must be >1')

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Create shape of underlying variable array
        shape = shape+[k-1]

        # Use a uniform prior
        if prior is None:
            prior = Categorical(O.ones(shape)/float(k))

        # Initialize the parameter
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)

        # shape should correspond to the sample shape
        self.shape = shape



class DirichletParameter(Parameter):
    r"""Dirichlet parameter.

    This is a convenience class for creating a parameter 
    :math:`\theta` with a Dirichlet posterior:

    .. math::

        \theta \sim \text{Dirichlet}(\mathbf{\alpha})

    By default, a uniform Dirichlet prior is used:

        \theta \sim \text{Dirichlet}(\mathbf{1})

    TODO: explain that a sample is a categorical prob dist (as compared to
    CategoricalParameter, where a sample is a single value)


    Parameters
    ----------
    k : int > 2
        Number of categories.
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Dirichlet`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Dirichlet` ``(1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use no transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'DirichletParameter'``


    Examples
    --------

    TODO: creating variable

    """

    def __init__(self,
                 k: int = 2,
                 shape: Union[int, List[int]] = [],
                 posterior=Dirichlet,
                 prior=None,
                 transform=lambda x: x,
                 initializer={'concentration': pos_xavier},
                 var_transform={'concentration': O.softplus},
                 name='DirichletParameter'):

        # Check type of k
        if not isinstance(k, int):
            raise TypeError('k must be an integer')
        if k<2:
            raise ValueError('k must be >1')

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Create shape of underlying variable array
        shape = shape+[k]

        # Use a uniform prior
        if prior is None:
            prior = Dirichlet(O.ones(shape))

        # Initialize the parameter
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class BoundedParameter(Parameter):
    r"""A parameter bounded on either side

    This is a convenience class for creating a parameter :math:`\beta` bounded
    on both sides.  It uses a logit-normal posterior distribution:

    .. math::

        \text{Logit}(\beta) \sim \text{Normal}(\mu, \sigma)


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use a sigmoid transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    min : float
        Minimum value the parameter can take.
        Default = 0.
    max : float
        Maximum value the parameter can take.
        Default = 1.
    name : str
        Name of the parameter(s).
        Default = ``'BoundedParameter'``

    Examples
    --------

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=Normal,
                 prior=Normal(0, 1),
                 transform=None,
                 initializer={'loc': xavier, 'scale': scale_xavier},
                 var_transform={'loc': lambda x: x, 'scale': O.softplus},
                 min: float = 0.0,
                 max: float = 1.0,
                 name='BoundedParameter'):

        # Check bounds
        if min > max:
            raise ValueError('min is larger than max')

        # Create the transform based on the bounds
        if transform is None:
            transform = lambda x: min + (max-min)*O.sigmoid(x)

        # Create the parameter
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class PositiveParameter(Parameter):
    r"""A parameter which takes only positive values.

    This is a convenience class for creating a parameter :math:`\beta` which 
    can only take positive values.  It uses a log-normal variational posterior
    distribution:

    .. math::

        \text{Log}(\beta) \sim \text{Normal}(\mu, \sigma)


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use an exponential transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    min : float
        Minimum value the parameter can take.
        Default = 0.
    max : float
        Maximum value the parameter can take.
        Default = 1.
    name : str
        Name of the parameter(s).
        Default = ``'PositiveParameter'``

    Examples
    --------

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=Normal,
                 prior=Normal(0, 1),
                 transform=O.exp,
                 initializer={'loc': xavier, 'scale': scale_xavier},
                 var_transform={'loc': lambda x: x, 'scale': O.softplus},
                 name='PositiveParameter'):
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class DeterministicParameter(Parameter):
    r"""A parameter which takes only a single value (i.e., the posterior is a 
    single point value, not a probability distribution).


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Deterministic`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use no transformation.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'PositiveParameter'``

    Examples
    --------

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=Deterministic,
                 prior=Normal(0, 1),
                 transform=lambda x: x,
                 initializer={'loc': xavier},
                 var_transform={'loc': lambda x: x},
                 name='DeterministicParameter'):
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)
