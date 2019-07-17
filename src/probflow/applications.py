"""Ready-made models.

The applications module contains pre-built |Models| and |Modules|.

"""


__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'PoissonRegression',
    'DenseNetwork',
    'DenseRegression',
    'DenseClassifier',
]



import probflow.core.ops as O
from probflow.parameters import Parameter
from probflow.parameters import ScaleParameter
from probflow.distributions import Normal
from probflow.distributions import Categorical
from probflow.distributions import Poisson
from probflow.modules import Module
from probflow.modules import Dense
from probflow.models import ContinuousModel
from probflow.models import DiscreteModel
from probflow.models import CategoricalModel



class LinearRegression(ContinuousModel):
    r"""A multiple linear regression

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    dims : int
        Dimensionality of the independent variable (number of features)

    Attributes
    ----------
    weights : :class:`.Parameter`
        Regression weights
    bias : :class:`.Parameter`
        Regression intercept
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, dims):
        self.weights = Parameter([dims, 1])
        self.bias = Parameter()
        self.std = ScaleParameter()


    def __call__(self, x):
        return Normal(x @ self.weights() + self.bias(), self.std())



class LogisticRegression(CategoricalModel):
    r"""A logistic regression

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    dims : int
        Dimensionality of the independent variable (number of features)

    Attributes
    ----------
    weights : :class:`.Parameter`
        Regression weights
    bias : :class:`.Parameter`
        Regression intercept
    """
    def __init__(self, dims):
        self.weights = Parameter([dims, 1])
        self.bias = Parameter()


    def __call__(self, x):
        return Bernoulli(x @ self.weights() + self.bias())



class PoissonRegression(DiscreteModel):
    r"""A Poisson regression (a type of generalized linear model)

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    dims : int
        Dimensionality of the independent variable (number of features)

    Attributes
    ----------
    weights : :class:`.Parameter`
        Regression weights
    bias : :class:`.Parameter`
        Regression intercept
    """
    def __init__(self, dims):
        self.weights = Parameter([dims, 1])
        self.bias = Parameter()


    def __call__(self, x):
        return Poisson(O.exp(x @ self.weights() + self.bias()))



class DenseNetwork(Module):
    r"""A multilayer dense neural network

    TODO: warning about how this is a Module not a Model

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    dims : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features).
    activation : callable
        Activation function to apply to the outputs of each layer.
        Note that the activation function will not be applied to the outputs
        of the final layer.
        Default = :math:`\max ( 0, x )`

    Attributes
    ----------
    layers : List[:class:`.Dense`]
        List of :class:`.Dense` neural network layers to be applied
    activations : List[callable]
        Activation function for each layer
    """

    def __init__(self, dims, activation=O.relu):
        self.activations = [activation for i in range(len(dims)-2)]
        self.activations += [lambda x: x]
        self.layers = [Dense(dims[i], dims[i+1])
                       for i in range(len(dims)-1)]


    def __call__(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x



class DenseRegression(ContinuousModel):
    r"""A regression using a multilayer dense neural network

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    dims : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        dimensionality of the dependent variable (number of dimensions of the
        target).

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        mean
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, dims):
        self.network = DenseNetwork(dims)
        self.std = ScaleParameter()


    def __call__(self, x):
        return Normal(self.network(x), self.std())



class DenseClassifier(CategoricalModel):
    r"""A classifier which uses a multilayer dense neural network

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    dims : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        number of classes of the target.

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        class probabilities
    """

    def __init__(self, dims):
        self.network = DenseNetwork(dims)


    def __call__(self, x):
        return Categorical(self.network(x))
