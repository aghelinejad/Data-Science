# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#############################################################################
# load data
da = pd.read_csv("50_Startups.csv")
x = da.iloc[:, :3].values
y = da.iloc[:, -1].values
#############################################################################
# investigation on feature relations

# Polting variables
plt.scatter(x[:, 0], y)
plt.scatter(x[:, 1], y)
plt.scatter(x[:, 2], y)
#############################################################################
# data preprocessing

# feature scaling
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
x_sc = x_scaler.fit_transform(x)
y_scaler = StandardScaler()
y_sc = y_scaler.fit_transform(y.reshape(len(y), 1))
#############################################################################
# train-test separation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
#############################################################################
# Fitting multiple Regression models

# Multilinear regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train, y_train)
r2_lin = linreg.score(x_test, y_test)

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 2)
x_poly = polyreg.fit_transform(x_train)
polyreg_1 = LinearRegression()
polyreg_1.fit(x_poly, y_train)

# Decision tree regression
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(criterion = 'friedman_mse', splitter = 'best', random_state = 10)
treereg.fit(x_train, y_train)
r2_tree = treereg.score(x_test, y_test)

# Random forest regression
from sklearn.ensemble import RandomForestRegressor
randreg = RandomForestRegressor(criterion = 'mae', n_estimators = 200, n_jobs = -1, random_state = 10)
randreg.fit(x_train, y_train)
r2_rand = randreg.score(x_test, y_test)

# XGBoost
from xgboost import XGBRegressor
boostreg = XGBRegressor(booster = 'gbtree', learning_rate = 0.5, n_estimators = 90, random_state = 10)
boostreg.fit(x_train, y_train)
r2_boost = boostreg.score(x_test, y_test)
#############################################################################
# model predictions
y_pred_lin = linreg.predict(x_test)
y_pred_poly = polyreg_1.predict(polyreg.fit_transform(x_test))
y_pred_tree = treereg.predict(x_test)
y_pred_rand = randreg.predict(x_test)
y_pred_boost = boostreg.predict(x_test)
#############################################################################
# model evaluation and validation

# R2 metric
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred_lin),
      r2_score(y_test, y_pred_poly),
      r2_score(y_test, y_pred_tree),
      r2_score(y_test, y_pred_rand),
      r2_score(y_test, y_pred_boost))

# k-fold cross validation
from sklearn.model_selection import cross_val_score

linscore = cross_val_score(estimator = linreg, X = x_train, y = y_train, cv = 10)
accuracy_lin = linscore.mean()

polyscore = cross_val_score(estimator = polyreg_1, X = polyreg.fit_transform(x_train), y = y_train, cv = 10)
accuracy_poly = polyscore.mean()

treescore = cross_val_score(estimator = treereg, X = x_train, y = y_train, cv = 10)
accuracy_tree = treescore.mean()

randscore = cross_val_score(estimator = randreg, X = x_train, y = y_train, cv = 10)
accuracy_rand = randscore.mean()

boostscore = cross_val_score(estimator = boostreg, X = x_train, y = y_train, cv = 10)
accuracy_boost = boostscore.mean()
#############################################################################
# model optimization

# grid search
from sklearn.model_selection import GridSearchCV

parameters_tree = [{"criterion": ("mse", "mae", "friedman_mse"), "splitter": ("best", "random")}]
grid_search_tree = GridSearchCV(estimator = treereg, param_grid = parameters_tree, scoring = "r2", cv = 10)
grid_search_tree.fit(x_train, y_train)

grid_search_tree.best_score_
grid_search_tree.best_params_
grid_search_tree.best_estimator_

parameters_rand = [{"n_estimators": [150, 200, 250], "criterion": ("mse", "mae")}]
grid_search_rand = GridSearchCV(estimator = randreg, param_grid = parameters_rand, scoring = "r2", cv = 10, n_jobs = -1)
grid_search_rand.fit(x_train, y_train)

grid_search_rand.best_score_
grid_search_rand.best_params_
grid_search_rand.best_estimator_

parameters_boost = [{"learning_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "n_estimators": [90, 100, 120], "booster": ("gbtree", "gblinear")}]
grid_search_boost = GridSearchCV(estimator = boostreg, param_grid = parameters_boost, scoring = "r2", cv = 10, n_jobs = -1)
grid_search_boost.fit(x_train, y_train)

grid_search_boost.best_score_
grid_search_boost.best_params_
grid_search_boost.best_estimator_