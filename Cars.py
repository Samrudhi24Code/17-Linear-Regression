# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:45:00 2025
@author: Dell
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import pylab

# Loading the dataset
cars = pd.read_csv(r"E:\Honars(DS)\Data Science\17-Linear Regression\cars.csv")

# EDA - Central tendency, dispersion, and graphical representation
cars.describe()

# Graphical Representation
plt.bar(height=cars.HP, x=np.arange(1, 82, 1))
plt.hist(cars.HP)
plt.boxplot(cars.HP)

# Jointplot - Scatterplot between HP and MPG
sns.jointplot(x=cars['HP'], y=cars['MPG'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['HP'])

# QQ Plot
stats.probplot(cars.MPG, dist="norm", plot=pylab)
plt.show()

# Pairplot for relationships between variables
sns.pairplot(cars.iloc[:, :])
plt.show()

# Correlation matrix
print(cars.corr())

# Initial Linear Regression Model
ml1 = smf.ols('MPG ~ WT + VOL + SP + HP', data=cars).fit()
print(ml1.summary())

# Influence plot to identify influential points
sm.graphics.influence_plot(ml1)
plt.show()

# Dropping the influential point (index 76)
cars_new = cars.drop(cars.index[[76]])

# Rebuilding the regression model without the influential point
ml_new = smf.ols('MPG ~ WT + VOL + SP + HP', data=cars_new).fit()
print(ml_new.summary())

# Variance Inflation Factor (VIF) Calculation
rsq_hp = smf.ols('HP ~ WT + VOL + SP', data=cars).fit().rsquared
vif_hp = 1 / (1 - rsq_hp)

rsq_wt = smf.ols('WT ~ HP + VOL + SP', data=cars).fit().rsquared
vif_wt = 1 / (1 - rsq_wt)

rsq_vol = smf.ols('VOL ~ HP + WT + SP', data=cars).fit().rsquared
vif_vol = 1 / (1 - rsq_vol)

rsq_sp = smf.ols('SP ~ HP + WT + VOL', data=cars).fit().rsquared
vif_sp = 1 / (1 - rsq_sp)

# Creating a DataFrame for VIF values
d1 = {'Variable': ['HP', 'WT', 'VOL', 'SP'], 'VIF': [vif_hp, vif_wt, vif_vol, vif_sp]}
vif_frame = pd.DataFrame(d1)
print(vif_frame)

# Dropping WT due to high VIF and rebuilding the model
final_ml = smf.ols('MPG ~ VOL + SP + HP', data=cars).fit()
print(final_ml.summary())

# Prediction
pred = final_ml.predict(cars)

# QQ Plot for residuals
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Residual Plot to check independence of errors
sns.residplot(x=pred, y=cars.MPG, lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('Fitted vs Residuals')
plt.show()

# Influence plot
sm.graphics.influence_plot(final_ml)
plt.show()

# Splitting the data into train and test sets
cars_train, cars_test = train_test_split(cars, test_size=0.2, random_state=42)

# Preparing the model on train data
model_train = smf.ols('MPG ~ VOL + SP + HP', data=cars_train).fit()
print(model_train.summary())

# Predicting on test data
test_pred = model_train.predict(cars_test)

# Calculating test error
test_error = cars_test.MPG - test_pred

# Calculating RMSE for the test set
test_rmse = np.sqrt(np.mean(test_error ** 2))
print("Test RMSE:", test_rmse)

# Calculating RMSE for the train set (optional)
train_pred = model_train.predict(cars_train)
train_error = cars_train.MPG - train_pred
train_rmse = np.sqrt(np.mean(train_error ** 2))
print("Train RMSE:", train_rmse)













