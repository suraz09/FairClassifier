import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'configs'))

import constant

"""
Class to determine the fairness metrics used to assess the classifier
"""
class FairMetrics:
    def __init__(self):
        pass

    """
    Calculates the P% rule of predicted values of given sensitive attribute
    :param y_pred, Prediction values of the classifier
    :param z_values, the list of sensitive attributes of the dataset
    :param threshold, threshold for the classifier to decide fairness (0.5) for binary classification.
    """
    def p_rule(selfs, y_pred, z_values, threshold = constant.THRESHOLD):
        y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
        y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
        odds = y_z_1.mean() / y_z_0.mean()
        return np.min([odds, 1 / odds]) * 100

