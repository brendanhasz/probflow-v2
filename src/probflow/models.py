"""Models.

Models are objects which take Tensor(s) as input, perform some computation on 
those Tensor(s), and output probability distributions.

TODO: more...

* :func:`.Model`
* :func:`.ContinuousModel`
* :func:`.DiscreteModel`
* :func:`.CategoricalModel`
* :func:`.save_model`
* :func:`.load_model`

----------

"""

__all__ = [
    'Model',
    'ContinuousModel',
    'DiscreteModel',
    'CategoricalModel',
    'save_model',
    'load_model',
]


from probflow.core.base import BaseParameter
from probflow.core.base import BaseDistribution
from probflow.core.base import BaseModule
from probflow.core.base import BaseModel
from probflow.core.base import BaseDataGenerator
from probflow.core.base import BaseCallback



class Model(BaseModel):
    """TODO


    """


    @abstractmethod
    def __init__(self, *args):
        pass


    @abstractmethod
    def __call__(self):
        """Perform the forward pass and return a distribution"""
        pass


    def fit(self, ...):
        """TODO

        """
        pass
        # TODO!


    def set_learning_rate(self, lr):
        """Set the learning rate used for this model's optimizer.

        TODO
        """
        pass
        # TODO


    def predictive_distribution(self, x, n=1000):
        """Draw samples from the model given x.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features"). 
        n : int
            Number of samples to draw from the model.


        Returns
        -------
        |ndarray|
            Samples from the predictive distribution.  Size
            (num_samples,x.shape[0],y.shape[0],...,y.shape[-1])        
        """
        pass
        # TODO


    def mean_distribution(self, x, n=1000):
        """Draw samples of the model's mean estimate given x.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features"). 
        n : int
            Number of samples to draw from the model.


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples,x.shape[0],y.shape[0],...,y.shape[-1])        
        """
        pass
        # TODO


    def predict(self, x, method='map'):
        """Predict dependent variable using the model

        TODO...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  
        method : 'map' or callable
            Whether to use maximum a posteriori estimation (``method='map'``),
            or to make predictions with some function of the predictive 
            distribution.  For example, to use the mean of the predictive
            distribution as the prediction, set ``method=np.mean``.


        Returns
        -------
        |ndarray|
            Predicted y-value for each sample in ``x``.  Of size
            (x.shape[0], y.shape[0], ..., y.shape[-1])


        Examples
        --------
        TODO: Docs...

        """
        pass
        # TODO



    def metric(self, x, y, metric):
        """Compute a metric of model performance.

        TODO: docs

        TODO: methods which just call this w/ a specific metric? for shorthand


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        metric : str or callable
            Metric to evaluate.  Available metrics:

            * 'log_prob' : log probability
            * 'acc': accuracy
            * 'accuracy': accuracy
            * 'mse': mean squared error
            * 'sse': sum of squared errors
            * 'mae': mean absolute error
            * callable: a function which takes (y_true, y_pred_dist)

            TODO: r^2, cross-entropy, etc

        Returns
        -------
        TODO
        """
        pass
        # TODO


    def posterior_mean(self, params=None):
        """Get the mean of the posterior distribution(s).

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.
            Default is to get the mean for all parameters in the model.


        Returns
        -------
        dict
            Means of the parameter posterior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarray|s with the posterior means.  The |ndarray|s are the same
            size as each parameter.
        """
        pass
        # TODO


    def posterior_sample(self, params=None, n=10000):
        """Draw samples from parameter posteriors.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.
            Default is to get a sample for all parameters in the model.
        num_samples : int
            Number of samples to take from each posterior distribution.
            Default = 1000


        Returns
        -------
        dict
            Samples from the parameter posterior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarray|s with the posterior samples.  The |ndarray|s are of size
            (``num_samples``, param.shape).
        """
        pass
        # TODO


    def posterior_plot(self,
                       params=None,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4):
        """Plot posterior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list
            List of parameters to plot.  Default is to plot the posterior of
            all parameters in the model.
        n : int
            Number of samples to take from each posterior distribution for
            estimating the density.  Default = 1000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns.
        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        """
        pass
        # TODO


    def prior_sample(self, params=None, n=10000):
        """Draw samples from parameter priors.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.
            Default is to sample priors of all parameters in the model.
        n : int
            Number of samples to take from each prior distribution.
            Default = 10000


        Returns
        -------
        dict
            Samples from the parameter prior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarray|s with the prior samples.  The |ndarray|s are of size
            (``n``,param.shape).
        """
        pass
        # TODO


    def prior_plot(self,
                   params=None,
                   n=10000,
                   style='fill',
                   cols=1,
                   bins=20,
                   ci=0.0,
                   bw=0.075,
                   color=None,
                   alpha=0.4):
        """Plot prior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : |None| or str or list of str
            List of parameters to plot.  Default is to plot the prior of
            all parameters in the model.
        n : int
            Number of samples to take from each prior distribution.
            Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns.
        bins : int or list or |ndarray|
            Number of bins to use for the prior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        """
        pass
        # TODO


    def log_prob(self, 
                 x, 
                 y,
                 individually=True,
                 distribution=False,
                 n=1000):
        """Compute the log probability of `y` given `x` and the model.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target"). 
        individually : bool
            If ``individually`` is True, returns log probability for each 
            sample individually, so return shape is ``(x.shape[0], ?)``.
            If ``individually`` is False, returns sum of all log probabilities,
            so return shape is ``(1, ?)``.
        distribution : bool
            If ``distribution`` is True, returns log probability posterior
            distribution (``n`` samples from the model),
            so return shape is ``(?, n)``.
            If ``distribution`` is False, returns log posterior probabilities
            using the maximum a posteriori estimate for each parameter,
            so the return shape is ``(?, 1)``.
        n : int
            Number of samples to draw for each distribution if 
            ``distribution=True``.

        Returns
        -------
        log_probs : |ndarray|
            Log probabilities. Shape is determined by ``individually``, 
            ``distribution``, and ``n`` kwargs.
        """
        pass
        # TODO


    def log_prob_by(self, 
                    x,
                    y,
                    x_by,
                    bins=30,
                    plot=True):
        """Log probability of observations ``y`` given ``x`` and the
        model as a function of independent variable(s) ``x_by``.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        x_by : int or str or list of int or list of str
            Which independent variable(s) to plot the log probability as a
            function of.  That is, which columns in ``x`` to plot by.
        bins : int
            Number of bins.
        plot : bool
            Whether to plot the data (if True), or just return the values.

        
        Returns
        -------
        log_probs : |ndarray|
            The average log probability as a function of ``x_by``.
            If x_by is an int or str, is of shape ``(bins,)``.
            If ``x_by`` is a list of length 2, ``prob_by`` is of shape
            ``(bins, bins)``.
        """
        pass
        # TODO


    def prob(self, 
             x, 
             y,
             individually=True,
             distribution=False,
             n=1000):
        """Compute the probability of `y` given `x` and the model.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target"). 
        individually : bool
            If ``individually`` is True, returns probability for each 
            sample individually, so return shape is ``(x.shape[0], ?)``.
            If ``individually`` is False, returns product of all probabilities,
            so return shape is ``(1, ?)``.
        distribution : bool
            If ``distribution`` is True, returns posterior probability
            distribution (``n`` samples from the model),
            so return shape is ``(?, n)``.
            If ``distribution`` is False, returns posterior probabilities
            using the maximum a posteriori estimate for each parameter,
            so the return shape is ``(?, 1)``.
        n : int
            Number of samples to draw for each distribution if 
            ``distribution=True``.

        Returns
        -------
        probs : |ndarray|
            Probabilities. Shape is determined by ``individually``, 
            ``distribution``, and ``n`` kwargs.
        """
        pass
        # TODO


    def prob_by(self, 
                x,
                y,
                x_by,
                bins=30,
                plot=True):
        """Probability of observations ``y`` given ``x`` and the
        model as a function of independent variable(s) ``x_by``.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        x_by : int or str or list of int or list of str
            Which independent variable(s) to plot the log probability as a
            function of.  That is, which columns in ``x`` to plot by.
        bins : int
            Number of bins.
        plot : bool
            Whether to plot the data (if True), or just return the values.

        
        Returns
        -------
        log_probs : |ndarray|
            The average log probability as a function of ``x_by``.
            If x_by is an int or str, is of shape ``(bins,)``.
            If ``x_by`` is a list of length 2, ``prob_by`` is of shape
            ``(bins, bins)``.
        """
        pass
        # TODO


    def summary(self):
        """Print a summary of the model and its parameters.

        TODO

        """
        pass
        # TODO


    def __del__(self):
        """Delete the model, its sub-modules, and its parameters.

        TODO

        """
        pass
        # TODO



class ContinuousModel(Model):
    """TODO


    """

    """
    inherits all methods from Model, and adds:

    * :meth:`.predictive_distribution_plot`
    * :meth:`.confidence_intervals`
    * :meth:`.pred_dist_prc`
    * :meth:`.pred_dist_covered`
    * :meth:`.pred_dist_coverage`
    * :meth:`.coverage_by`
    * :meth:`.calibration_curve`
    * :meth:`.r_squared`
    * :meth:`.residuals`
    * :meth:`.residuals_plot`

    """