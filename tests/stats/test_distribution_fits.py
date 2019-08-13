"""Tests the statistical accuracy of fitting some distributions"""



import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import probflow as pf



def is_close(a, b, th=1e-5):
    """Check two values are close"""
    return np.abs(a-b) < th



def test_fit_normal():
    """Test fitting a normal distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    mu = np.random.randn()
    sig = np.exp(np.random.randn())
    x = np.random.randn(N, 1).astype('float32')
    x = x*sig + mu

    class NormalModel(pf.Model):
        def __init__(self):
            self.mu = pf.Parameter(name='mu')
            self.sig = pf.ScaleParameter(name='sig')
        def __call__(self):
            return pf.Normal(self.mu(), self.sig())

    # Create and fit model
    model = NormalModel()
    model.fit(x, batch_size=100, epochs=1000, learning_rate=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci('mu')
    assert lb < mu
    assert ub > mu
    assert is_close(mu, model.posterior_mean('mu'), th=0.2)

    # Check inferences for std are correct
    lb, ub = model.posterior_ci('sig')
    assert lb < sig
    assert ub > sig
    assert is_close(sig, model.posterior_mean('sig'), th=0.2)



# TODO: StudentT



# TODO: Cauchy



# TODO: Gamma



# TODO: InverseGamma



def test_fit_bernoulli():
    """Test fitting a bernoulli distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    prob = 0.7
    x = (tf.random.uniform([N]) < prob).numpy().astype('float32')

    class BernoulliModel(pf.Model):
        def __init__(self):
            self.prob = pf.BoundedParameter(name='prob')
        def __call__(self):
            return pf.Bernoulli(probs=self.prob())

    # Create and fit model
    model = BernoulliModel()
    model.fit(x, batch_size=100, epochs=1000, learning_rate=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci('prob')
    assert lb < prob
    assert ub > prob
    assert is_close(prob, model.posterior_mean('prob'), th=0.1)



# TODO: Categorical



def test_fit_poisson():
    """Test fitting a poisson distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    rate = 5
    x = tf.random.poisson([N], rate).numpy()

    class PoissonModel(pf.Model):
        def __init__(self):
            self.rate = pf.Parameter(name='rate')
        def __call__(self):
            return pf.Poisson(tf.exp(self.rate()))

    # Create and fit model
    model = PoissonModel()
    model.fit(x, batch_size=100, epochs=1000, learning_rate=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci('rate')
    assert np.exp(lb) < rate
    assert np.exp(ub) > rate
    assert is_close(rate, np.exp(model.posterior_mean('rate')), th=1.0)

    # TODO: uh doesn't seem to be working.
    import pdb; pdb.set_trace()


# TODO: Dirichlet
