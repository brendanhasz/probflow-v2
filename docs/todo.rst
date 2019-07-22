Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them.  If you're interested in tackling one of them, I'd be thrilled and
`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!


Backlog
-------

* Model basics
* Name registry
* Tests
* Docs
* Model evaluation methods
* Tests for those
* Make Module.trainable_variables return 
* Bayes estimate / decision methods
* Different plotting methods for different types of dists (both for Parameter
  priors/posteriors and predictive distribution plots)
* Convolutional modules


Notes
-----


Name registry
^^^^^^^^^^^^^

Have a set of names in core.settings, and funcs to add a name to that set or remove it (which is called during ``Parameter.__del__``)
