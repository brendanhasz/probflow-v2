Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them.  If you're interested in tackling one of them, I'd be thrilled! 
`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!


Backlog
-------

* Stats tests
* Speed tests on large dataset (looked like there was some kind of autograph warning?)
* Model evaluation methods (ones to be used in readme)
* Tests for those
* README / index
* User guide
* Examples
* Docs for everything implemented so far
* Merge back to main repo and release 2.0.0
* Different plotting methods for different types of dists (both for Parameter priors/posteriors and predictive distribution plots)
* All model evaluation methods + specialized types of models
* Make Module.trainable_variables return tf.Variables which are properties of module+sub-modules as well (and not neccesarily in parameters, also allow embedding of tf.Modules?)
* Real-world examples w/ public BigQuery datasets
* Bayes estimate / decision methods
* Convolutional modules


Issues
------

* Poisson currently requires y values to be floats? I think that's a TFP/TF 2.0 issue though (in their sc there's the line ``tf.maximum(y, 0.)``, which throws an error when y is of an int type).  Could cast inputs to float in pf.distributions.Poisson.__init__...


