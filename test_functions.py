from typing import List
from sklearn.linear_model import Lasso
from optimizers import Nesterov_Optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
from math import sqrt
import copy
import numpy as np


def test_models(X_train, y_train):
    methods = ['accelerated', 'gradient', 'dual_gradient']
    models = []
    for method in methods:
        model = Nesterov_Optimizers()
        model.fit(X_train, y_train, method=method)
        print(np.linalg.norm(model.get_coef()))
        models.append(model)
    model = Lasso()
    model.fit(X_train, y_train)
    models.append(model)
    return models


def get_measures(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mae_perc = mean_absolute_percentage_error(y_true, y_pred)
    return [rmse, mae, mae_perc]


def models_quality(X_test, y_test, models):
    results = []
    for model in models:
        predictions = model.predict(X_test)
        results.append(get_measures(y_test, predictions))
    return pd.DataFrame(results, columns=['RMSE', 'MAE', 'MAE_perc'],
                        index=['accelerated', 'gradient', 'dual_gradient', 'lasso'])


# z projektu z roku wyżej

def get_regression(p, n):
    X = np.empty(shape=(0, p))
    y = np.empty(shape=(0, 1))
    np.random.seed(1)
    mean = np.random.rand(p)
    covariance = np.eye(p)*0.01
    X = np.concatenate([X, np.random.multivariate_normal(mean, covariance, size=n)])
    a = np.random.rand(p)
    eps = np.random.rand(n)
    y = X.dot(a) + eps
    return X, np.squeeze(y)

