"""Tests the statistical accuracy of a Linear Regression w/ ProbFlow"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import probflow as pf



# TODO: test that a fit recovers the true parameters used to generate data
