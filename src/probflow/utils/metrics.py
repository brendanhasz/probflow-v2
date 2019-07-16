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


import numpy as np

import probflow.core.ops as O



def log_prob(y_true, y_pred_dist):
    """Sum of the log probabilities of predictions."""
    return O.sum(y_pred_dist.log_prob(y_true))



def accuracy(y_true, y_pred_dist):
    """Accuracy of predictions."""
    return O.mean(O.round(y_pred_dist.mean()) == y_true)



def mean_squared_error(y_true, y_pred_dist):
    """Mean squared error."""
    return O.mean(O.square(y_true-y_pred_dist.mean()))



def sum_squared_error(y_true, y_pred_dist):
    """Sum of squared error."""
    return O.sum(O.square(y_true-y_pred_dist.mean()))



def mean_absolute_error(y_true, y_pred_dist):
    """Mean absolute error."""
    return O.mean(O.abs(y_true-y_pred_dist.mean()))



def r_squared(y_true, y_pred_dist):
    """Coefficient of determination."""
    ss_tot = O.sum(O.square(y_true-O.mean(y_true)))
    ss_res = O.sum(O.square(y_true-y_pred_dist.mean()))
    return 1.0 - ss_res / ss_tot



# TODO: recall
def true_positive_rate(y_true, y_pred_dist):
    """True positive rate aka sensitivity aka recall."""
    
# TODO: specificity

# TODO: precision

# TODO: false_negative_rate

# TODO: false_positive_rate


def f1_score(y_true, y_pred_dist):
    """F-measure."""
    p = precision(y_true, y_pred_dist)
    r = true_positive_rate(y_true, y_pred_dist)
    return 2*(p*r)/(p+r)


# TODO: jaccard_similarity

# TODO: roc_auc



def get_metric_fn(metric):
    """Get a function corresponding to a metric string"""

    # List of valid metric strings
    metrics = [
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
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,
        'true_positive_rate': recall,
        'tpr': recall,
        'f1_score': f1_score,
        'f1': f1_score,
        'jaccard_similarity': jaccard_similarity,
        'jaccard': jaccard_similarity,
        'roc_auc': roc_auc,
        'auroc': roc_auc,
        'auc': roc_auc,
    ]

    # Error if invalid metric string
    if metric not in metrics:
        raise ValueError(metric+' is not a valid metric string. '+
                         'Valid strings are: '+', '.join(metrics.keys()))

    # Return the corresponding function
    return metrics[metric]