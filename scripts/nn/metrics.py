# --*-- coding: utf-8 --*--
"""
This module defines some metrics functions for research.
"""
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def prec_rec_fscore(target: torch.Tensor, prediction: torch.Tensor) -> tuple:
    """
    Calculate precision, recall and f1 score for gold and predicted labels

    Args:
        target (torch.Tensor/np.array/list): true labels
        prediction (torch.Tensor/np.array/list): predicted labels

    Return:
        tuple: tuple of (precision, recall, f1_score) in this order
    """
    precision, recall, f1_score, _ = precision_recall_fscore_support(target, prediction, average='macro')
    return precision, recall, f1_score


def accuracy_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    Calculate accuracy score for gold and predicted labels

    Args:
        target (torch.Tensor/np.array/list): true labels
        prediction (torch.Tensor/np.array/list): predicted labels

    Return:
        float: accuracy score
    """
    corrects = (prediction == target).sum()
    return corrects / len(target)


def wrong_accuracy_label_score(target: torch.Tensor, prediction: torch.Tensor, labels: list, error=True) -> np.array:
    """
    Recalculate the confusion matrix for labeling error, right predicted error is set to 0,
    the others are calculated based on percentages of the given numbers of gold labels
    except for the matched predicted labels

    Args:
        target (torch.Tensor/np.array/list): true labels
        prediction (torch.Tensor/np.array/list): predicted labels
        labels (list[int]): list of given labels/classes
        error (bool): whether to create wrong labeling error matrix

    Return:
        np.array: new confusion matrix (transposed), with gold labels on x axis, predicted labels on y axis
                    (normal confusion matrices are reversed)
    """

    matrix = confusion_matrix(y_true=target, y_pred=prediction, labels=labels)
    label_sum = np.sum(matrix, axis=1)

    if error:
        for i in range(len(label_sum)):
            label_sum[i] -= matrix[i, i]
            matrix[i, i] = 0

    for i, (vec, lsum) in enumerate(zip(matrix, label_sum)):
        if lsum != 0:
            matrix[i] = np.around(vec * 100 / lsum)

    return matrix.T