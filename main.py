#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import common.feature_selection as feat_sel
import common.test_env as test_env

boston = load_boston()

#give overall description and overview of the dataset
print(boston.keys())
print(boston.DESCR)
bostondf = pd.DataFrame(boston.data, columns=boston.feature_names)
bostondf['MEDV'] = boston.target

# check for missing values in all the columns
print("[INFO] dataset isnull():\n {}".format(bostondf.isnull().sum()))

# remove MEDV outliers (MEDV = 50.0)
bostondf = bostondf[~(bostondf['MEDV'] >= 50.0)]
print(np.shape(bostondf))

#Visualization of distributions
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in bostondf.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('results/figure_01.png', papertype='a4')

#descriptive analysis/quantitative overview
print(bostondf.head())
print(bostondf.describe())
medv_mode = bostondf['MEDV'].mode()
print('MEDV mode: ', medv_mode[0])

#distribution of the important variables with mean and median
clr = ['blue', 'green', 'red']
fig, axs = plt.subplots(ncols=3,figsize=(15,3))

plt.figure(1)

for i, var in enumerate(['RM', 'LSTAT', 'PTRATIO']):
    plt.subplot(131 + i)
    sns.distplot(bostondf[var],  color = clr[i])
    plt.axvline(bostondf[var].mean(), color=clr[i], linestyle='solid', linewidth=2)
    plt.axvline(bostondf[var].median(), color=clr[i], linestyle='dashed', linewidth=2)
plt.savefig('results/figure_02.png', papertype='a4')


#the distribution of the target variable
sns.distplot(bostondf['MEDV'], bins=30)
plt.show()

# set the size of the figure
sns.set(rc={'figure.figsize':(12, 8)})

g = sns.PairGrid(bostondf, vars=['LSTAT', 'RM', 'CRIM', 'NOX', 'MEDV'], height=1.5, aspect=1.5)
g = g.map_diag(plt.hist)
g = g.map_lower(sns.regplot, lowess=True, scatter_kws={'s': 15, 'alpha':0.3}, 
                line_kws={'color':'red', 'linewidth': 2})
g = g.map_upper(sns.kdeplot, n_levels=15, cmap='coolwarm')   
plt.savefig('results/figure_03.png', papertype='a4')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
plt.show()

#Boxplot visualisation of quantitative overview of the dataset
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in bostondf.items():
    sns.boxplot(y=k, data=bostondf, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('results/figure_04.png', papertype='a4')

#boxplot and scatterplot visualisation for RM and LSTAT
fig, axes = plt.subplots(2, 2)
fig.suptitle("Scatterplot and Boxplot for LSTAT and RM")

sns.regplot(x=bostondf['LSTAT'], y=bostondf['MEDV'], lowess=True, scatter_kws={'s': 25, 'alpha':0.3},
            line_kws={'color':'red', 'linewidth': 2}, ax=axes[0, 0])

sns.boxplot(x=bostondf['LSTAT'], ax=axes[0, 1])

sns.regplot(x=bostondf['RM'], y=bostondf['MEDV'], lowess=True, scatter_kws={'s': 25, 'alpha':0.3},
            line_kws={'color':'red', 'linewidth': 2}, ax=axes[1, 0])

sns.boxplot(x=bostondf['RM'], ax=axes[1, 1]).set(xlim=(3, 9))
plt.savefig('results/figure_05.png', papertype='a4')
plt.show()

#histogram representation of some key features
fig, axes = plt.subplots(2, 2)
fig.suptitle("Histogram of Key Features")

sns.distplot(bostondf['LSTAT'], bins=30, ax=axes[0, 0])
sns.distplot(bostondf['RM'], bins=30, ax=axes[0, 1])
sns.distplot(bostondf['CRIM'], bins=30, ax=axes[1, 0])
sns.distplot(bostondf['NOX'], bins=30, ax=axes[1, 1])
plt.savefig('results/figure_06.png', papertype='a4')
plt.show()


#the pairwise correlation plot on the data.
plt.figure(figsize=(20, 10))
sns.heatmap(bostondf.corr().abs(),  annot=True)
plt.savefig('results/figure_07.png', papertype='a4')

# Scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX']
x = bostondf.loc[:,column_sels]
y = bostondf['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('results/figure_08.png', papertype='a4')
plt.show()

#Age feature details 
sns.distplot(bostondf['AGE'].dropna(),kde=False,color='darkred',bins=40)
plt.savefig('results/figure_09.png', papertype='a4')
plt.show()


#Crime rate visualisation
sns.distplot(bostondf['CRIM'].dropna(),kde=False,color='green',bins=40)
plt.savefig('results/figure_10.png', papertype='a4')
plt.show()

#Lower status rate visualisation
sns.distplot(bostondf['LSTAT'].dropna(),kde=False,color='darkblue',bins=40)
plt.savefig('results/figure_11.png', papertype='a4')
plt.show()

#pupil-teacher ratio visualisation
sns.distplot(bostondf['PTRATIO'].dropna(),kde=False,color='pink',bins=40)
plt.savefig('results/figure_12.png', papertype='a4')
plt.show()

#Room number visualisation
sns.distplot(bostondf['RM'].dropna(),kde=False,color='darkorange',bins=40)
plt.savefig('results/figure_13.png', papertype='a4')
plt.show()

#Calculate the linear regression with inpendent variables
prices = bostondf['MEDV']
features = bostondf.drop('MEDV', axis = 1)    
    
reg = LinearRegression()
pt_ratio = bostondf['PTRATIO'].values.reshape(-1, 1)
reg.fit(pt_ratio, prices)
plt.plot(pt_ratio, reg.predict(pt_ratio), color='red', linewidth=1)
plt.scatter(pt_ratio, prices, alpha=0.5, c=prices)
plt.xlabel('PTRATIO')
plt.ylabel('PRICE')
plt.savefig('results/figure_14.png', papertype='a4')
plt.show()


l_stat = bostondf['LSTAT'].values.reshape(-1, 1)
reg.fit(l_stat, prices)
plt.plot(l_stat, reg.predict(l_stat), color='red', linewidth=1)
plt.scatter(l_stat, prices, alpha=0.5, c=prices)
plt.xlabel('LSTAT')
plt.ylabel('PRICE')
plt.savefig('results/figure_15.png', papertype='a4')
plt.show()

prices = bostondf['MEDV']
features = bostondf.drop('MEDV', axis = 1)
reg = LinearRegression()
rm = bostondf['RM'].values.reshape(-1, 1)
reg.fit(rm, prices)
plt.plot(rm, reg.predict(rm), color='red', linewidth=1)
plt.scatter(rm, prices, alpha=0.5, c=prices)
plt.xlabel('RM')
plt.ylabel('PRICE')
plt.savefig('results/figure_16.png', papertype='a4')
plt.show()

#Stage 5: try regression with different algorithms and training/testing sets
X = pd.DataFrame(np.c_[bostondf['LSTAT'], bostondf['RM'], bostondf['CRIM'], bostondf['NOX']], columns=['LSTAT', 'RM', 'CRIM', 'NOX'])
Y = bostondf['MEDV']

print(X.shape)
print(Y.shape)

# split the training and test data set in 75% : 25%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression()
lm.fit(X_train, Y_train)

print('Linear Regression coefficients: {}'.format(lm.coef_))
print('Linear Regression intercept: {}'.format(lm.intercept_))

# model evaluation for training set
y_train_predict = lm.predict(X_train)

# calculating the intercept and slope for the regression line
b, m = np.polynomial.polynomial.polyfit(Y_train, y_train_predict, 1)

sns.scatterplot(Y_train, y_train_predict, alpha=0.4)
sns.regplot(Y_train, y_train_predict, truncate=True, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'red', 'linewidth': 2})
sns.lineplot(np.unique(Y_train), np.unique(np.poly1d(b + m * np.unique(Y_train))), linewidth=0.5, color='b')
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Training Set]")
plt.show()

#Get the linear model performance for training set
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
print("The linear model performance for training set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

# model evaluation for testing set
y_test_predict = lm.predict(X_test)

sns.scatterplot(Y_test, y_test_predict, alpha=0.4)
sns.regplot(Y_test, y_test_predict, truncate=True, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2})
 
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Test Set]")
 
plt.show()

#Get the linear model performance for test set
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

print("\nThe linear model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
LR_MAPE= MAPE(Y_test,y_test_predict)
print("MAPE: ",LR_MAPE)

y_train_residual = y_train_predict - Y_train
y_test_residual = y_test_predict - Y_test

plt.subplot(1, 2, 1)
sns.distplot(y_train_residual, bins=15)
plt.title('Residual Histogram for Training Set')

plt.subplot(1, 2, 2)
sns.distplot(y_test_residual, bins=15)
plt.title('Residual Histogram for Test Set')

plt.show()

fig, axes = plt.subplots()
fig.suptitle('Residual plot of Training and Test set')

# Plot the residuals after fitting a linear model
sns.residplot(y_train_predict, y_train_residual, lowess=True, color="b", ax=axes, label='Training Set', 
              scatter_kws={'s': 25, 'alpha':0.3})

sns.residplot(y_test_predict, y_test_residual, lowess=True, color="g", ax=axes, label='Test Set',
              scatter_kws={'s': 25})

legend = axes.legend(loc='upper left', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#f9e79f')

plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.show()

#create some polynomial features
poly_features = PolynomialFeatures(degree=2)
# transform the features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
     
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
# predicting on test data-set
y_test_predicted = poly_model.predict(poly_features.fit_transform(X_test))

y_train_residual = y_train_predicted - Y_train
y_test_residual = y_test_predicted - Y_test

plt.subplot(1, 2, 1)
sns.distplot(y_train_residual, bins=15)
plt.title('Residual Histogram for Training Set [Polynomial Model]')
plt.subplot(1, 2, 2)
sns.distplot(y_test_residual, bins=15)
plt.title('Residual Histogram for Test Set [Polynomial Model]')
plt.show()

sns.scatterplot(Y_train, y_train_predicted, alpha=0.4)
sns.regplot(Y_train, y_train_predicted, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2}, order=2)
 
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Training Set]")
 
plt.show()

sns.scatterplot(Y_test, y_test_predicted, alpha=0.4)
sns.regplot(Y_test, y_test_predicted, scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2}, order=2)
 
plt.xlabel("Actual Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$ [Test Set]")
 
plt.show()

# evaluating the model on training data-set
rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
r2_train = r2_score(Y_train, y_train_predicted)
     
print("The polynomial model performance for the training set")
print("RMSE of training set is {}".format(rmse_train))
print("R2 score of training set is {}".format(r2_train))

# evaluating the model on test data-set
rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predicted))
r2_test = r2_score(Y_test, y_test_predicted)

print("The polynomial model performance for the test set")
print("RMSE of test set is {}".format(rmse_test))
print("R2 score of test set is {}".format(r2_test))
PR_MAPE= MAPE(Y_test,y_test_predicted)
print("MAPE: ",PR_MAPE)

#calculate the r-squared value by using different regression models
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
    
#run the regression functions and show the results
if __name__ == '__main__':
    test_env.versions(['numpy', 'statsmodels', 'sklearn'])
    X, y = load_boston(return_X_y=True)

    linear_regression(X, y)
    linear_regression_selection(X, y)
    polynomial_regression(X, y)
    svr_regression(X, y)
    decision_tree_regression(X, y)
    random_forest_regression(X, y)
    print('Done')