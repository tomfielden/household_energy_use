import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

"""
>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> # y = 1 * x_0 + 2 * x_1 + 3
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> reg = LinearRegression().fit(X, y)
>>> reg.score(X, y)
1.0
>>> reg.coef_
array([1., 2.])
>>> reg.intercept_ 
3.0000...
>>> reg.predict(np.array([[3, 5]]))
array([16.])
"""

HOUSEHOLD_ENERGY_FILENAME = 'recs2015_public_v4.csv' 

household_df = pd.read_csv(HOUSEHOLD_ENERGY_FILENAME)

basian_ridge = linear_model.BayesianRidge(
    alpha_1=1e-06,
    alpha_2=1e-06,
    compute_score=False,
    copy_X=True,
    fit_intercept=True,
    lambda_1=1e-06,
    lambda_2=1e-06,
    n_iter=300,
    normalize=False,
    tol=0.001,
    verbose=True
)

# KWh for first TV
KWHTV1 = household_df['KWHTV1']

# Pick some interesting features
tv_features = ['NUMFREEZ', 'TEMPNITE']

# Fit the model
household_model = basian_ridge.fit(
    household_df[tv_features],
    KWHTV1
)

household_model.coef_
household_model.intercept_
