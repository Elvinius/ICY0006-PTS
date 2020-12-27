## Boston Housing Dataset analysis 

### Stage 1
1. The Boston Housing Dataset is a dataset collected by the U.S. Census Service concerning housing in the suburb areas of Boston, Massachusetts. A model trained on this data that is seen as a good fit could then be used to make certain predictions about a home, particularly, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.

2. Creators: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.  

3. The dataset consists of the following columns (variables):

<strong>CRIM</strong> - per capita crime rate by town <br>
<strong>ZN</strong> - proportion of residential land zoned for lots over 25,000 sq.ft. <br>
<strong>INDUS</strong> - proportion of non-retail business acres per town. <br>
<strong>CHAS</strong> - Charles River dummy variable (1 if tract bounds river; 0 otherwise) <br>
<strong>NOX</strong> - nitric oxides concentration (parts per 10 million) <br>
<strong>RM</strong>- average number of rooms per dwelling <br>
<strong>AGE</strong> - proportion of owner-occupied units built prior to 1940 <br>
<strong>DIS</strong> - weighted distances to five Boston employment centres <br>
<strong>RAD</strong> - index of accessibility to radial highways <br>
<strong>TAX</strong> - full-value property-tax rate per $10,000 <br>
<strong>PTRATIO</strong> - pupil-teacher ratio by town <br>
<strong>B</strong> - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town <br>
<strong>LSTAT</strong> - % lower status of the population <br>
<strong>MEDV</strong> - Median value of owner-occupied homes in $1000's <br>

4. As it is seen from the boxplots, columns like CRIM, ZN, RM, B have outliers. MEDV outliers (MEDV = 50.0) have been removed before plotting more distributions.

According to the histograms, columns CRIM, ZN, B has highly skewed distributions. Besides, MEDV looks to have a normal distribution (the predictions) and other colums seem to have normal or bimodel ditribution of data except CHAS (which is a discrete variable).

### Stage 2

In our dataset two data variables - ZN (proportion of residential land zoned for lots over 25,000 sq.ft.) with 0 for 25th, 50th percentiles and CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise) with 0 for 25th, 50th and 75th percentiles are noticeable. These happen as both variables are conditional, plus categorical ones. It can be assumed that these variables may not be useful in regression tasks such as predicting MEDV (Median value of owner-occupied homes).

Another interesing fact on the dataset is the max value of MEDV. It seems to be censored at around 50.00 (corresponding to a median price of $50,000). Based on that, values above 50.00 may not help to predict MEDV. 

As per boxplots of different variables, the following can be inferred:

1. The crime rate of housing districts is usually very low: the graph is near 0-10% area with a few outliers till 88%. The conclusion is that most of the housing data is from zones having crime rate near 0.25%.
2. <strong>ZN: the proportion of the residential land zoned for lots over 25,000 sq.ft.</strong> <br>
According to the boxplot, the median percentage is around 0.1% meaning that most of the people own houses where the number of the houses is limited.
3. <strong>INDUS: proportion of non-retail business acres per town.</strong><br>
INDUS has median around 9-10% implying that people prefer the residential areas where one out of every 10 people is a non-retail businessman.
4. <strong>LSTAT: lower status of the population with percentage </strong> <br>
It can be inferred that people tend to live in the areas where there are the less poor people.

### Stage 3

It is interesting to observe that the highest correlations are observed between INDUS and NOX, as well as those between TAX and RAD and TAX and INDUS. It is reasonable that nitrogen oxide levels, as well as tax rates are the highest near industrial areas. These are possible sources of multicollinearity, each explaining the same thing as far as how they impact variation in MEDV calculation.

When it comes to MEDV, it is found that the average number of the rooms has one of the highest positive correlation, while pupil-teacher ratio and LSTAT have the highest negative correlations. TAX and RAD are also highly correlated features.

Overall, the variables LSTAT, INDUS, RM, TAX, NOX, PTRATIO has a correlation score above 0.5 with MEDV which is a good indication of using them as predictors.

As per second step of this stage, six independent variables - namely, 'LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX' were used against dependent MEDV in the scatter plots. With the help of these features the MEDV prices can be predicted. 

Additionally, by choosing RM, LSTAT, and PTRATIO the linear regression model has been implemented and scatter plots with line have been created. As per observations:

a) An increase in the value of 'RM' should raise the value of 'MEDV', since there are probably bigger residences in that neighborhood and it means that bigger dwellings cost more money. <br>
b) A higher 'LSTAT' would probably represent a poorer neighborhood, thus, the target variable which is 'MEDV' would decrease if 'LSTAT' increases. <br>
c) PTRATIO is inversely correlated.

### Stage 5

For the regression models using two sets (training and testing) the following functions have been created:

```
def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg
```

```
def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y)
    return linear_regression(X_sel, y, print_text='Linear regression with feature selection')
```

```
def polynomial_regression(X, y, print_text='Polynomial regression'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train)
    print_metrics(y_test, lin_reg.predict(polynomial_features.fit_transform(X_test)), print_text)
    return polynomial_features
```

```
def svr_regression(X, y, print_text='SVR regression'):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = sc.fit_transform(np.expand_dims(y, axis=1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = SVR(kernel='rbf', gamma='auto')
    reg.fit(X_train, np.ravel(y_train))
    print_metrics(np.squeeze(y_test), np.squeeze(reg.predict(X_test)), 'SVR')
```

```
def decision_tree_regression(X, y, print_text='Decision tree regression'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = DecisionTreeRegressor(criterion="mae", max_depth=None)
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), 'Decision tree regression')
```

```
def random_forest_regression(X, y, print_text='Random forest regression'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = RandomForestRegressor(n_estimators=10, random_state=0)
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
```

It can be concluded that the straight regression line is unable to capture the patterns in the dataset which is an example to underfitting. To overcome this, there is a need to increase the complexity of the model. This was done by converting the original features into their higher order polynomial terms by using the PolynomialFeatures class provided by scikit-learn. Then the training was conducted by using Linear Regression. The model was trained with 75/25 proportion scale.

According to the different types of regression models used for the task, the R-squared gets the value around 0.60. This seems fine as the metrics not being 100% does not mean that a chosen model is not good. 

Looking at LSTAT and RM scatter plots it can also be observed that a considerable part of observations fall near the regression line. 

