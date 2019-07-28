"""Metrics.

Evaluation metrics

* :func:`.log_prob`
* :func:`.acc`
* :func:`.accuracy`
* :func:`.mse`
* :func:`.sse`
* :func:`.mae`

----------

"""



__all__ = [
    'log_prob',
    'accuracy',
    'mean_squared_error',
    'sum_squared_error',
    'mean_absolute_error',
    'r_squared',
    'true_positive_rate',
    'true_negative_rate',
    'precision',
    'f1_score',
    'get_metric_fn',
]



import numpy as np
import pandas as pd



def y_true_numpy(fn):
    """Cast y_true to numpy before computing metric"""

    def metric_fn(y_true, y_pred_dist):
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            return fn(y_true.values(), y_pred_dist)
        elif isinstance(y_true, np.ndarray):
            return fn(y_true, y_pred_dist)
        else:
            return fn(y_true.numpy(), y_pred_dist)

    return metric_fn



@y_true_numpy
def log_prob(y_true, y_pred_dist):
    """Sum of the log probabilities of predictions."""
    return np.sum(y_pred_dist.log_prob(y_true).numpy())



@y_true_numpy
def accuracy(y_true, y_pred_dist):
    """Accuracy of predictions."""
    return np.mean(y_pred_dist.mean().numpy() == y_true)



@y_true_numpy
def mean_squared_error(y_true, y_pred_dist):
    """Mean squared error."""
    return np.mean(np.square(y_true-y_pred_dist.mean().numpy()))



@y_true_numpy
def sum_squared_error(y_true, y_pred_dist):
    """Sum of squared error."""
    return np.sum(np.square(y_true-y_pred_dist.mean().numpy()))



@y_true_numpy
def mean_absolute_error(y_true, y_pred_dist):
    """Mean absolute error."""
    return np.mean(np.abs(y_true-y_pred_dist.mean().numpy()))



@y_true_numpy
def r_squared(y_true, y_pred_dist):
    """Coefficient of determination."""
    ss_tot = np.sum(np.square(y_true-np.mean(y_true)))
    ss_res = np.sum(np.square(y_true-y_pred_dist.mean().numpy()))
    return 1.0 - ss_res / ss_tot



@y_true_numpy
def true_positive_rate(y_true, y_pred_dist):
    """True positive rate aka sensitivity aka recall."""
    p = np.sum(y_true == 1)
    tp = np.sum((y_pred_dist.mean().numpy() == y_true) & (y_true == 1))
    return tp/p



@y_true_numpy
def true_negative_rate(y_true, y_pred_dist):
    """True negative rate aka specificity aka selectivity."""
    n = np.sum(y_true == 0)
    tn = np.sum((y_pred_dist.mean().numpy() == y_true) & (y_true == 0))
    return tn/n



@y_true_numpy
def precision(y_true, y_pred_dist):
    """Precision."""
    ap = np.sum(y_pred_dist.mean().numpy())
    tp = np.sum((y_pred_dist.mean().numpy() == y_true) & (y_true == 1))
    return tp/ap



@y_true_numpy
def f1_score(y_true, y_pred_dist):
    """F-measure."""
    p = precision(y_true, y_pred_dist)
    r = true_positive_rate(y_true, y_pred_dist)
    return 2*(p*r)/(p+r)



# TODO: jaccard_similarity



# TODO: roc_auc



# TODO: cross_entropy



def get_metric_fn(metric):
    """Get a function corresponding to a metric string"""

    # List of valid metric strings
    metrics = {
        'lp': log_prob,
        'log_prob': log_prob,
        'accuracy': accuracy,
        'acc': accuracy,
        'mean_squared_error': mean_squared_error,
        'mse': mean_squared_error,
        'sum_squared_error': sum_squared_error,
        'sse': sum_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mae': mean_absolute_error,
        'r_squared': r_squared,
        'r2': r_squared,
        'recall': true_positive_rate,
        'sensitivity': true_positive_rate,
        'true_positive_rate': true_positive_rate,
        'tpr': true_positive_rate,
        'specificity': true_negative_rate,
        'selectivity': true_negative_rate,
        'true_negative_rate': true_negative_rate,
        'tnr': true_negative_rate,
        'precision': precision,
        'f1_score': f1_score,
        'f1': f1_score,
        #'jaccard_similarity': jaccard_similarity,
        #'jaccard': jaccard_similarity,
        #'roc_auc': roc_auc,
        #'auroc': roc_auc,
        #'auc': roc_auc,
    }

    # Return the corresponding function
    if callable(metric):
        return metric
    elif isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(metric+' is not a valid metric string. '+
                             'Valid strings are: '+', '.join(metrics.keys()))
        else:
            return metrics[metric]
    else:
        raise TypeError('metric must be a str or callable')

