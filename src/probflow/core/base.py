"""Abstract base classes.

The core.base module contains abstract base classes (ABCs) for all of
ProbFlow’s classes.

"""


from abc import ABC, abstractmethod
from math import ceil

from probflow.core.settings import get_backend



class BaseParameter(ABC):
    """Abstract base class for ProbFlow Parameters"""

    @abstractmethod
    def __init__(self, *args):
        pass


    @abstractmethod
    def __call__(self):
        """Return a sample from or the MAP estimate of this parameter."""
        pass


    @abstractmethod
    def kl_loss(self):
        """Compute the sum of the Kullback–Leibler divergences between this
        parameter's priors and its variational posteriors."""
        pass


    @abstractmethod
    def posterior_mean(self):
        """Get the mean of the posterior distribution(s)."""



class BaseDistribution(ABC):
    """Abstract base class for ProbFlow Distributions"""


    @abstractmethod
    def __init__(self, *args):
        pass


    @abstractmethod
    def __call__(self):
        """Get the distribution object from the backend"""
        pass


    def prob(self, y):
        """Compute the probability of some data given this distribution"""
        if get_backend() == 'pytorch':
            return self.__call__().log_prob(y).exp()
        else:
            return self.__call__().prob(y)


    def log_prob(self, y):
        """Compute the log probability of some data given this distribution"""
        return self.__call__().log_prob(y)


    def mean(self):
        """Compute the mean of this distribution"""
        return self.__call__().mean()


    def sample(self, n=1):
        """Generate a random sample from this distribution"""
        if get_backend() == 'pytorch':
            if isinstance(n, int) and n > 1:
                return self.__call__().rsample()
            else:
                return self.__call__().rsample(n)
        else:
            if isinstance(n, int) and n > 1:
                return self.__call__().sample()
            else:
                return self.__call__().sample(n)



class BaseModule(ABC):
    """Abstract base class for ProbFlow Modules"""

    @abstractmethod
    def __init__(self, *args):
        """Initialize the module (abstract method)"""
        pass


    @abstractmethod
    def __call__(self):
        """Perform forward pass (abstract method)"""
        pass



class BaseModel(ABC):
    """Abstract base class for ProbFlow Models"""
    pass
    # TODO: might not need BaseModel, if it's totally unused...



class BaseDataGenerator(ABC):
    """Abstract base class for ProbFlow DataGenerators"""

    @abstractmethod
    def __init__(self, *args):
        pass


    @property
    @abstractmethod
    def n_samples(self):
        """Number of samples in the dataset"""
        pass


    @property
    @abstractmethod
    def batch_size(self):
        """Number of samples per batch"""
        pass


    @abstractmethod
    def __getitem__(self, index):
        """Generate one batch of data"""
        pass


    def __len__(self):
        """Number of batches per epoch"""
        return int(ceil(self.n_samples/self.batch_size))


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        pass



class BaseCallback(ABC):
    """Abstract base class for ProbFlow Callbacks"""

    @abstractmethod
    def __init__(self, *args):
        pass


    @abstractmethod
    def on_epoch_begin(self):
        """Will be called at the beginning of each training epoch"""
        pass


    @abstractmethod
    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        pass


    @abstractmethod
    def on_train_begin(self):
        """Will be called at the beginning of training"""
        pass


    @abstractmethod
    def on_train_end(self):
        """Will be called at the end of training"""
        pass
