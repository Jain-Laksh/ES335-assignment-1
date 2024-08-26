from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    assert y_hat.size == y.size
    assert y.size>0

    correct_predictions = (y_hat == y).sum()
    accuracy_value = correct_predictions / y.size
    
    return accuracy_value


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert y.size>0

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    predicted_positives = (y_hat == cls).sum()

    if predicted_positives == 0:
        return 0.0
    
    precision_value = true_positives / predicted_positives
    
    return precision_value


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert y.size>0

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    actual_positives = (y == cls).sum()

    if actual_positives == 0:
        return 0.0
    
    recall_value = true_positives / actual_positives
    
    return recall_value


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size
    assert y.size>0

    mse = np.mean((y_hat - y) ** 2)
    rmse_value = np.sqrt(mse)

    return rmse_value


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y.size>0

    mae_value = np.mean(np.abs(y_hat - y))
    
    return mae_value
