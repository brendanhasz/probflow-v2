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



import matplotlib.pyplot as plt

from probflow.core.base import BaseParameter
from probflow.core.base import BaseDistribution
from probflow.core.base import BaseModule
from probflow.core.base import BaseModel
from probflow.core.base import BaseDataGenerator
from probflow.core.base import BaseCallback

from probflow.modules import Module

from probflow.utils.plotting import plot_dist


# TODO: might not need to inherit BaseModel, if it's totally unused...

class Model(BaseModel, Module):
    """Abstract base class for probflow models.


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Module`:

    * :func:`~probflow.models.Model.__init__`
    * :func:`~probflow.models.Model.__call__`
    * :func:`~probflow.models.Model.parameters`
    * :func:`~probflow.models.Model.kl_loss`

    and adds the following model-specific methods:

    * :func:`~probflow.models.Model.fit`
    * :func:`~probflow.models.Model.stop_training`
    * :func:`~probflow.models.Model.set_learning_rate`
    * :func:`~probflow.models.Model.predictive_distribution`
    * :func:`~probflow.models.Model.mean_distribution`
    * :func:`~probflow.models.Model.predict`
    * :func:`~probflow.models.Model.metric`
    * :func:`~probflow.models.Model.posterior_mean`
    * :func:`~probflow.models.Model.posterior_sample`
    * :func:`~probflow.models.Model.posterior_plot`
    * :func:`~probflow.models.Model.prior_sample`
    * :func:`~probflow.models.Model.prior_plot`
    * :func:`~probflow.models.Model.log_prob`
    * :func:`~probflow.models.Model.log_prob_by`
    * :func:`~probflow.models.Model.prob`
    * :func:`~probflow.models.Model.prob_by`
    * :func:`~probflow.models.Model.summary`
    * :func:`~probflow.models.Model.__del__`

    """


    def fit(self, ...):
        """TODO

        """
        pass
        # TODO!
        # TODO: _learning_rate, update optimizer if changed
        # TODO: _is_training, stop training if false



    def stop_training(self):
        """Stop the training loop.

        TODO
        """
        self._is_training = False



    def set_learning_rate(self, lr):
        """Set the learning rate used for this model's optimizer.

        TODO
        """

        # Check type
        if not isinstance(lr, float):
            raise ValueError('lr must be a float')

        # Set the learning rate
        self._learning_rate = lr



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



    def metric(self, x, y=None, metric='log_prob'):
        """Compute a metric of model performance.

        TODO: docs

        TODO: methods which just call this w/ a specific metric? for shorthand


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| to generate both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target"). 
        metric : str or callable
            Metric to evaluate.  Available metrics:

            * 'lp': log likelihood sum
            * 'log_prob': log likelihood sum
            * 'accuracy': accuracy
            * 'acc': accuracy
            * 'mean_squared_error': mean squared error
            * 'mse': mean squared error
            * 'sum_squared_error': sum squared error
            * 'sse': sum squared error
            * 'mean_absolute_error': mean absolute error
            * 'mae': mean absolute error
            * 'r_squared': coefficient of determination
            * 'r2': coefficient of determination
            * 'recall': true positive rate
            * 'sensitivity': true positive rate
            * 'true_positive_rate': true positive rate
            * 'tpr': true positive rate
            * 'specificity': true negative rate
            * 'selectivity': true negative rate
            * 'true_negative_rate': true negative rate
            * 'tnr': true negative rate
            * 'precision': precision
            * 'f1_score': F-measure
            * 'f1': F-measure
            * callable: a function which takes (y_true, y_pred_dist)


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
                    x_by,
                    x,
                    y=None,
                    bins=30,
                    plot=True):
        """Log probability of observations ``y`` given ``x`` and the
        model as a function of independent variable(s) ``x_by``.

        TODO: docs...

        Parameters
        ----------
        x_by : int or str or list of int or list of str
            Which independent variable(s) to plot the log probability as a
            function of.  That is, which columns in ``x`` to plot by.
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
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
             y=None,
             individually=True,
             distribution=False,
             n=1000):
        """Compute the probability of `y` given `x` and the model.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
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
                x_by,
                x,
                y=None,
                bins=30,
                plot=True):
        """Probability of observations ``y`` given ``x`` and the
        model as a function of independent variable(s) ``x_by``.

        TODO: docs...

        Parameters
        ----------
        x_by : int or str or list of int or list of str
            Which independent variable(s) to plot the log probability as a
            function of.  That is, which columns in ``x`` to plot by.
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
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
    """Abstract base class for probflow models where the dependent variable 
    (the target) is continuous and 1-dimensional.


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Model`:

    * :func:`~probflow.models.ContinuousModel.__init__`
    * :func:`~probflow.models.ContinuousModel.__call__`
    * :func:`~probflow.models.ContinuousModel.parameters`
    * :func:`~probflow.models.ContinuousModel.kl_loss`
    * :func:`~probflow.models.ContinuousModel.fit`
    * :func:`~probflow.models.ContinuousModel.stop_training`
    * :func:`~probflow.models.ContinuousModel.set_learning_rate`
    * :func:`~probflow.models.ContinuousModel.predictive_distribution`
    * :func:`~probflow.models.ContinuousModel.mean_distribution`
    * :func:`~probflow.models.ContinuousModel.predict`
    * :func:`~probflow.models.ContinuousModel.metric`
    * :func:`~probflow.models.ContinuousModel.posterior_mean`
    * :func:`~probflow.models.ContinuousModel.posterior_sample`
    * :func:`~probflow.models.ContinuousModel.posterior_plot`
    * :func:`~probflow.models.ContinuousModel.prior_sample`
    * :func:`~probflow.models.ContinuousModel.prior_plot`
    * :func:`~probflow.models.ContinuousModel.log_prob`
    * :func:`~probflow.models.ContinuousModel.log_prob_by`
    * :func:`~probflow.models.ContinuousModel.prob`
    * :func:`~probflow.models.ContinuousModel.prob_by`
    * :func:`~probflow.models.ContinuousModel.summary`
    * :func:`~probflow.models.ContinuousModel.__del__`

    and adds the following continuous-model-specific methods:

    * :func:`~probflow.models.ContinuousModel.confidence_intervals`
    * :func:`~probflow.models.ContinuousModel.pred_dist_plot`
    * :func:`~probflow.models.ContinuousModel.pred_dist_prc`
    * :func:`~probflow.models.ContinuousModel.pred_dist_covered`
    * :func:`~probflow.models.ContinuousModel.pred_dist_coverage`
    * :func:`~probflow.models.ContinuousModel.coverage_by`
    * :func:`~probflow.models.ContinuousModel.calibration_curve`
    * :func:`~probflow.models.ContinuousModel.r_squared`
    * :func:`~probflow.models.ContinuousModel.residuals`
    * :func:`~probflow.models.ContinuousModel.residuals_plot`

    """


    def confidence_intervals(self, 
                             x,
                             ci=0.95,
                             n=1000):
        """Compute confidence intervals on predictions for ``x``.

        TODO: docs


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        ci : float between 0 and 1
            Inner proportion of predictive distribution to use a the
            confidence interval.
            Default = 0.95
        n : int
            Number of samples from the posterior predictive distribution to
            take to compute the confidence intervals.
            Default = 1000

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.
        """

        # Sample from the predictive distribution
        pred_dist = self.predictive_distribution(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # Compute percentiles of the predictive distribution
        lb = 100*(1.0-ci)/2.0
        q = [lb, 100.0-lb]
        prcs = np.percentile(pred_dist, q, axis=0)
        return prcs[0, :], prcs[1, :]


    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4,
                       individually=False):
        """Plot posterior predictive distribution from the model given ``x``.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns (if 
            ``individually=True``.
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
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        """

        # Sample from the predictive distribution
        pred_dist = self.predictive_distribution(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # Plot the predictive distributions
        N = pred_dist.shape[1]
        if individually:
            rows = np.ceil(N/cols)
            for i in range(N):
                plt.subplot(rows, cols, i+1)
                plot_dist(pred_dist[:,i], xlabel='Datapoint '+str(i), 
                          style=style, bins=bins, ci=ci, bw=bw, alpha=alpha, 
                          color=color)
        else:
            plot_dist(pred_dist, xlabel='Dependent Variable', style=style, 
                      bins=bins, ci=ci, bw=bw, alpha=alpha, color=color)


    def predictive_prc(self, x, y=None, n=1000):
        """Compute the percentile of each observation along the posterior
        predictive distribution.

        TODO: Docs...  Returns a percentile between 0 and 1

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 1000

        Returns
        -------
        prcs : |ndarray| of float between 0 and 1
        """

        # Sample from the predictive distribution
        pred_dist = self.predictive_distribution(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # Return percentiles of true y data along predictive distribution
        #inds = np.argmax((np.sort(pred_dist, 0) >
        #                  y.reshape(1, x.shape[0], -1)),
        #                 axis=0)
        # TODO
        return inds/float(n)

        # TODO: check for when true y value is above max pred_dist val!
        # I think argmax returns 0 when that's the case, which is
        # obviously not what we want


    def pred_dist_covered(self, x, y=None, n=1000, ci=0.95):
        """Compute whether each observation was covered by a given confidence
        interval.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.pred_dist_covered` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 1000
        ci : float between 0 and 1
            Confidence interval to use.

        Returns
        -------
        TODO
        """

        # Check types
        if not isinstance(ci, float):
            if isinstance(ci, int):
                ci = float(ci)
            else:
                raise TypeError('ci must be a float')
        if ci < 0.0 or ci > 1.0:
            raise ValueError('ci must be between 0 and 1')

        # Compute the predictive percentile of each observation
        pred_prcs = self.predictive_prc(x, y=y, n=n)

        # Determine what samples fall in the inner ci proportion
        lb = (1.0-ci)/2.0
        ub = 1.0-lb
        return (pred_prcs>=lb) & (pred_prcs<ub)


    def pred_dist_coverage(self, x, y=None, n=1000, ci=0.95):
        """Compute what percent of samples are covered by a given confidence
        interval.

        TODO: Docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 1000
        ci : float between 0 and 1
            Confidence interval to use.


        Returns
        -------
        prc_covered : float between 0 and 1
            Proportion of the samples which were covered by the predictive
            distribution's confidence interval.
        """
        return self.pred_dist_covered(x, y=y, n=n, ci=ci).mean()


    def coverage_by(self, 
                    x_by,
                    x, 
                    y=None,
                    ci=0.95, 
                    bins=30, 
                    plot=True,
                    true_line_kwargs={},
                    ideal_line_kwargs={}):
        """Compute and plot the coverage of a given confidence interval
        of the posterior predictive distribution as a
        function of specified independent variables.

        TODO: Docs...



        Parameters
        ----------
        x_by : int or str or list of int or list of str
            Which independent variable(s) to plot the log probability as a
            function of.  That is, which columns in ``x`` to plot by.
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        ci : float between 0 and 1
            Inner percentile to find the coverage of.  For example, if 
            ``ci=0.95``, will compute the coverage of the inner 95% of the 
            posterior predictive distribution.
        bins : int
            Number of bins to use for x_by
        plot : bool
            Whether to plot the coverage.  Default = True
        true_line_kwargs : dict
            Dict to pass to matplotlib.pyplot.plot for true coverage line
        ideal_line_kwargs : dict
            Dict of args to pass to matplotlib.pyplot.plot for ideal coverage
            line.


        Returns
        -------
        xo : |ndarray|
            Values of x_by corresponding to bin centers.
        co : |ndarray|
            Coverage of the ``ci`` confidence interval of the predictive
            distribution in each bin.
        """



        # Compute whether each sample was covered by the predictive interval
        covered = self.pred_dist_covered(x, y=y, n=n, ci=ci)

        # Plot coverage proportion as a fn of x_by cols of x
        # TODO: how to handle if x is data generator?
        """
        xo, co = plot_by(x[:, x_by], 100*covered, bins=bins,
                         plot=plot, label='Actual', **true_line_kwargs)
        """

        # Line kwargs
        if 'linestyle' not in ideal_line_kwargs:
            ideal_line_kwargs['linestyle'] = '--'
        if 'color' not in ideal_line_kwargs:
            ideal_line_kwargs['color'] = 'k'

        # Also plot ideal line
        if plot and isinstance(x_by, int):
            plt.axhline(100*ci, label='Ideal', **ideal_line_kwargs)
            plt.legend()
            plt.ylabel(str(100*ci)+'% predictive interval coverage')
            plt.xlabel('Value of '+str(x_by))

        return xo, co


    def calibration_curve(self,
                          x,
                          y=None,
                          split_by=None,
                          bins=10,
                          plot=True):
        """Plot and/or return calibration curve.

        Plots and returns the calibration curve (the percentile of the posterior
        predictive distribution on the x-axis, and the percent of samples which
        actually fall into that range on the y-axis).


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        split_by : int
            Draw the calibration curve independently for datapoints
            with each unique value in `x[:,split_by]` (a categorical
            column).
        bins : int, list of float, or |ndarray|
            Bins used to compute the curve.  If an integer, will use
            `bins` evenly-spaced bins from 0 to 1.  If a vector,
            `bins` is the vector of bin edges.
        plot : bool
            Whether to plot the curve

        Returns
        -------
        cx : |ndarray|
            Vector of percentiles (the middle of each percentile
            bin).  Length is determined by `bins`.
        cy : |ndarray|
            Vector of percentages of samples which fell within each
            percentile bin of the posterior predictive distribution.

        See Also
        --------
        predictive_distribution : used to generate the posterior
            predictive distribution.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        """
        pass
        # TODO


    def r_squared(self,
                  x,
                  y=None,
                  n=1000,
                  plot=True):
        """Compute the Bayesian R-squared value.

        Compute the Bayesian R-squared distribution :ref:`[1] <ref_r_squared>`.
        TODO: more info and docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        n : int
            Number of posterior draws to use for computing the r-squared
            distribution.  Default = `1000`.
        plot : bool
            Whether to plot the r-squared distribution

        Returns
        -------
        |ndarray|
            Samples from the r-squared distribution.  Size: ``(num_samples,)``.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        References
        ----------
        .. _ref_r_squared:
        .. [1] Andrew Gelman, Ben Goodrich, Jonah Gabry, & Aki Vehtari.
            R-squared for Bayesian regression models.
            *The American Statistician*, 2018.
            https://doi.org/10.1080/00031305.2018.1549100
        """
        pass
        #TODO


    def residuals(self, x, y=None):
        """Compute the residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").

        Returns
        -------
        TODO

        """
        pass
        # TODO


    def residuals_plot(self, x, y=None):
        """Plot the distribution of residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").

        """
        pass
        # TODO



class DiscreteModel(ContinuousModel):
    """Abstract base class for probflow models where the dependent variable 
    (the target) is discrete (e.g. drawn from a Poisson distribution).


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Model`:

    * :func:`~probflow.models.DiscreteModel.__init__`
    * :func:`~probflow.models.DiscreteModel.__call__`
    * :func:`~probflow.models.DiscreteModel.parameters`
    * :func:`~probflow.models.DiscreteModel.kl_loss`
    * :func:`~probflow.models.DiscreteModel.fit`
    * :func:`~probflow.models.DiscreteModel.stop_training`
    * :func:`~probflow.models.DiscreteModel.set_learning_rate`
    * :func:`~probflow.models.DiscreteModel.predictive_distribution`
    * :func:`~probflow.models.DiscreteModel.mean_distribution`
    * :func:`~probflow.models.DiscreteModel.predict`
    * :func:`~probflow.models.DiscreteModel.metric`
    * :func:`~probflow.models.DiscreteModel.posterior_mean`
    * :func:`~probflow.models.DiscreteModel.posterior_sample`
    * :func:`~probflow.models.DiscreteModel.posterior_plot`
    * :func:`~probflow.models.DiscreteModel.prior_sample`
    * :func:`~probflow.models.DiscreteModel.prior_plot`
    * :func:`~probflow.models.DiscreteModel.log_prob`
    * :func:`~probflow.models.DiscreteModel.log_prob_by`
    * :func:`~probflow.models.DiscreteModel.prob`
    * :func:`~probflow.models.DiscreteModel.prob_by`
    * :func:`~probflow.models.DiscreteModel.summary`
    * :func:`~probflow.models.DiscreteModel.__del__`

    and also inherits several methods from :class:`.ContinuousModel`:

    * :func:`~probflow.models.DiscreteModel.confidence_intervals`
    * :func:`~probflow.models.DiscreteModel.pred_dist_prc`
    * :func:`~probflow.models.DiscreteModel.pred_dist_covered`
    * :func:`~probflow.models.DiscreteModel.pred_dist_coverage`
    * :func:`~probflow.models.DiscreteModel.coverage_by`
    * :func:`~probflow.models.DiscreteModel.calibration_curve`
    * :func:`~probflow.models.DiscreteModel.residuals`

    but overrides the following discrete-model-specific methods:

    * :func:`~probflow.models.DiscreteModel.pred_dist_plot`
    * :func:`~probflow.models.DiscreteModel.residuals_plot`

    """

    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4,
                       individually=False):
        """Plot posterior predictive distribution from the model given ``x``.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns (if 
            ``individually=True``.
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
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        """

        # Sample from the predictive distribution
        pred_dist = self.predictive_distribution(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # TODO: plot discretely

        # Plot the predictive distributions
        N = pred_dist.shape[1]
        if individually:
            rows = np.ceil(N/cols)
            for i in range(N):
                plt.subplot(rows, cols, i+1)
                #plot_dist(pred_dist[:,i], xlabel='Datapoint '+str(i), 
                #          style=style, bins=bins, ci=ci, bw=bw, alpha=alpha, 
                #          color=color)
        else:
            #plot_dist(pred_dist, xlabel='Dependent Variable', style=style, 
            #          bins=bins, ci=ci, bw=bw, alpha=alpha, color=color)


    def r_squared(self, *args, **kwargs):
        """Cannot compute R squared for a discrete model"""
        raise RuntimeError('Cannot compute R squared for a discrete model')


    def residuals_plot(self, x, y=None):
        """Plot the distribution of residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").

        """
        pass
        # TODO: plot discretely



class CategoricalModel(Model):
    """Abstract base class for probflow models where the dependent variable 
    (the target) is categorical (e.g. drawn from a Bernoulli distribution).


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Model`:

    * :func:`~probflow.models.CategoricalModel.__init__`
    * :func:`~probflow.models.CategoricalModel.__call__`
    * :func:`~probflow.models.CategoricalModel.parameters`
    * :func:`~probflow.models.CategoricalModel.kl_loss`
    * :func:`~probflow.models.CategoricalModel.fit`
    * :func:`~probflow.models.CategoricalModel.stop_training`
    * :func:`~probflow.models.CategoricalModel.set_learning_rate`
    * :func:`~probflow.models.CategoricalModel.predictive_distribution`
    * :func:`~probflow.models.CategoricalModel.mean_distribution`
    * :func:`~probflow.models.CategoricalModel.predict`
    * :func:`~probflow.models.CategoricalModel.metric`
    * :func:`~probflow.models.CategoricalModel.posterior_mean`
    * :func:`~probflow.models.CategoricalModel.posterior_sample`
    * :func:`~probflow.models.CategoricalModel.posterior_plot`
    * :func:`~probflow.models.CategoricalModel.prior_sample`
    * :func:`~probflow.models.CategoricalModel.prior_plot`
    * :func:`~probflow.models.CategoricalModel.log_prob`
    * :func:`~probflow.models.CategoricalModel.log_prob_by`
    * :func:`~probflow.models.CategoricalModel.prob`
    * :func:`~probflow.models.CategoricalModel.prob_by`
    * :func:`~probflow.models.CategoricalModel.summary`
    * :func:`~probflow.models.CategoricalModel.__del__`

    and adds the following categorical-model-specific methods:

    * :func:`~probflow.models.CategoricalModel.pred_dist_plot`
    * :func:`~probflow.models.CategoricalModel.calibration_curve`

    """


    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4,
                       individually=False):
        """Plot posterior predictive distribution from the model given ``x``.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns (if 
            ``individually=True``.
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
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        """

        # Sample from the predictive distribution
        pred_dist = self.predictive_distribution(x, n=n)

        # TODO


    def calibration_curve(self,
                          x,
                          y=None,
                          split_by=None,
                          bins=10,
                          plot=True):
        """Plot and return calibration curve.

        Plots and returns the calibration curve (estimated
        probability of outcome vs the true probability of that
        outcome).

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |Tensor| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or |Tensor|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        split_by : int
            Draw the calibration curve independently for datapoints
            with each unique value in `x[:,split_by]` (a categorical
            column).
        bins : int, list of float, or |ndarray|
            Bins used to compute the curve.  If an integer, will use
            `bins` evenly-spaced bins from 0 to 1.  If a vector,
            `bins` is the vector of bin edges.
        plot : bool
            Whether to plot the curve

        #TODO: split by continuous cols as well? Then will need to define bins or edges too

        TODO: Docs...

        """
        #TODO
        pass