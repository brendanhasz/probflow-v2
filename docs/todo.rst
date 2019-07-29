Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them.  If you're interested in tackling one of them, I'd be thrilled! 
`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!


Backlog
-------

* Make predict, _sample, metrics, prob, log_prob etc methods of Model handle when x is a DataGenerator (maybe convert to DG if not already, and then assume DG?)
* Tests for applications
* Tests for utils?
* Stats test and test speed on large dataset (looked like there was some kind of autograph warning w/ kl div?)
* Model evaluation methods (ones to be used in readme)
* Tests for those
* Docs for everything implemented so far
* README
* User guide
* Examples
* Make sure API is fully documented
* Different plotting methods for different types of dists (both for Parameter priors/posteriors and predictive distribution plots)
* All model evaluation methods + specialized types of models
* Bayes estimate / decision methods
* Make Module.trainable_variables return tf.Variables which are properties of module+sub-modules as well (and not neccesarily in parameters, also allow embedding of tf.Modules?)
* Convolutional modules

