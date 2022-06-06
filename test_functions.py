from typing import List
from sklearn.linear_model import Lasso
from optimizers import Nesterov_Optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
from math import sqrt


def test_models(X_train, y_train, eps=1e-3, max_iter=2500):
    methods = ['accelerated', 'gradient', 'dual_gradient']
    models = []
    model = Nesterov_Optimizers(eps=eps, max_iter=max_iter)
    for method in methods:
        model.fit(X_train, y_train, method=method)
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
