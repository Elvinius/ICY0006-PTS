#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import common.feature_selection as feat_sel
import common.test_env as test_env

boston = load_boston()
#print(boston.keys())
#print(boston.DESCR)
bostondf = pd.DataFrame(boston.data, columns=boston.feature_names)
bostondf['MEDV'] = boston.target
print(bostondf.head())
print(bostondf.describe())
# check for missing values in all the columns
print("[INFO] df isnull():\n {}".format(bostondf.isnull().sum()))

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in bostondf.items():
    sns.boxplot(y=k, data=bostondf, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('results/figure_1.png', papertype='a4')

# set the size of the figure
sns.set(rc={'figure.figsize':(12, 8)})

g = sns.PairGrid(bostondf, vars=['LSTAT', 'RM', 'CRIM', 'NOX', 'MEDV'], height=1.5, aspect=1.5)
g = g.map_diag(plt.hist)
g = g.map_lower(sns.regplot, lowess=True, scatter_kws={'s': 15, 'alpha':0.3}, 
                line_kws={'color':'red', 'linewidth': 2})
g = g.map_upper(sns.kdeplot, n_levels=15, cmap='coolwarm')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
plt.show()

#MEDV distributions

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in bostondf.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('results/figure_2.png', papertype='a4')

#the pairwise correlation plot on our data.
plt.figure(figsize=(20, 10))
sns.heatmap(bostondf.corr().abs(),  annot=True)
plt.savefig('results/figure_3.png', papertype='a4')

def print_metrics(y_true, y_pred, label):
    # Feel free to extend it with additional metrics from sklearn.metrics
    print('%s R squared: %.2f' % (label, r2_score(y_true, y_pred)))


def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y)
    return linear_regression(X_sel, y, print_text='Linear regression with feature selection')


def polynomial_regression(X, y, print_text='Polynomial regression'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train)
    print_metrics(y_test, lin_reg.predict(polynomial_features.fit_transform(X_test)), print_text)
    return polynomial_features


def svr_regression(X, y, print_text='SVR regression'):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = sc.fit_transform(np.expand_dims(y, axis=1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = SVR(kernel='rbf', gamma='auto')
    reg.fit(X_train, np.ravel(y_train))
    print_metrics(np.squeeze(y_test), np.squeeze(reg.predict(X_test)), 'SVR')


def decision_tree_regression(X, y, print_text='Decision tree regression'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = DecisionTreeRegressor(criterion="mae", max_depth=None)
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), 'Decision tree regression')


def random_forest_regression(X, y, print_text='Random forest regression'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = RandomForestRegressor(n_estimators=10, random_state=0)
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)


if __name__ == '__main__':
    test_env.versions(['numpy', 'statsmodels', 'sklearn'])

    # https://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset
    X, y = load_boston(return_X_y=True)

    linear_regression(X, y)
    linear_regression_selection(X, y)
    polynomial_regression(X, y)
    svr_regression(X, y)
    decision_tree_regression(X, y)
    random_forest_regression(X, y)
    print('Done')