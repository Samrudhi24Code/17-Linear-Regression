# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:45:00 2025
@author: Dell
"""

# Problem Statement:
# The dataset contains various features of cars (e.g., Weight, Volume, Speed, Horsepower) 
# and their corresponding Miles Per Gallon (MPG). The objective is to build a multiple 
# linear regression model that predicts MPG based on the independent variables like 
# Weight (WT), Volume (VOL), Speed (SP), and Horsepower (HP).
#
# The goals of this analysis are:
# 1. To explore the relationship between MPG and the car features.
# 2. To identify potential influential data points that may affect model performance.
# 3. To detect and mitigate multicollinearity issues through Variance Inflation Factor (VIF).
# 4. To develop a robust model for predicting MPG using the given features.
# 5. To evaluate the model's performance using metrics like RMSE (Root Mean Square Error).
#
# This can help in predicting the fuel efficiency of cars, which can be valuable for 
# car manufacturers, environmental agencies, and consumers interested in understanding 
# and improving fuel economy.


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import pylab

# Purpose: Load the dataset for analysis. In this case, it's about car attributes and their impact on MPG (miles per gallon).
cars = pd.read_csv(r"E:\Honars(DS)\Data Science\17-Linear Regression\cars.csv")

# Exploratory Data Analysis (EDA)
# EDA helps us understand the distribution and central tendency of the data (e.g., mean, median).
cars.describe()

# Graphical Representation of data
# Visualizing 'HP' (Horse Power) using bar, histogram, and box plot to understand its distribution.
plt.bar(height=cars.HP, x=np.arange(1, 82, 1))
plt.hist(cars.HP)
plt.boxplot(cars.HP)

# Jointplot: Check the relationship between 'HP' and 'MPG'.
sns.jointplot(x=cars['HP'], y=cars['MPG'])

# Countplot: Frequency distribution of 'HP'.
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['HP'])

# QQ Plot: Checking for normality of the 'MPG' data.
stats.probplot(cars.MPG, dist="norm", plot=pylab)
plt.show()

# Pairplot: Checking relationships between variables.
sns.pairplot(cars.iloc[:, :])
plt.show()

# Correlation matrix: Identifying correlations between numeric variables.
print(cars.corr())

# Initial Linear Regression Model
# We're building a model to predict 'MPG' (target variable) based on other car attributes (predictors like 'WT', 'VOL', etc.).
ml1 = smf.ols('MPG ~ WT + VOL + SP + HP', data=cars).fit()
print(ml1.summary())

# Purpose: Identify influential points in the data that may affect the model's reliability.
# Influence plot to detect outliers and influential points.
sm.graphics.influence_plot(ml1)
plt.show()

# Dropping the influential point (index 76) identified above
cars_new = cars.drop(cars.index[[76]])

# Rebuilding the regression model without the influential point
ml_new = smf.ols('MPG ~ WT + VOL + SP + HP', data=cars_new).fit()
print(ml_new.summary())

# Variance Inflation Factor (VIF) Calculation:
# Checking for multicollinearity between predictors by calculating VIF.
# High VIF indicates multicollinearity, which we want to avoid.
rsq_hp = smf.ols('HP ~ WT + VOL + SP', data=cars).fit().rsquared
vif_hp = 1 / (1 - rsq_hp)

rsq_wt = smf.ols('WT ~ HP + VOL + SP', data=cars).fit().rsquared
vif_wt = 1 / (1 - rsq_wt)

rsq_vol = smf.ols('VOL ~ HP + WT + SP', data=cars).fit().rsquared
vif_vol = 1 / (1 - rsq_vol)

rsq_sp = smf.ols('SP ~ HP + WT + VOL', data=cars).fit().rsquared
vif_sp = 1 / (1 - rsq_sp)

# Creating a DataFrame for the VIF values.
d1 = {'Variable': ['HP', 'WT', 'VOL', 'SP'], 'VIF': [vif_hp, vif_wt, vif_vol, vif_sp]}
vif_frame = pd.DataFrame(d1)
print(vif_frame)

# Dropping 'WT' due to high VIF and rebuilding the regression model
# Purpose: Reduce multicollinearity for more reliable predictions.
final_ml = smf.ols('MPG ~ VOL + SP + HP', data=cars).fit()
print(final_ml.summary())

# Making predictions using the final model.
pred = final_ml.predict(cars)

# Checking the residuals using a QQ plot to ensure they follow a normal distribution.
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Residual Plot to check for homoscedasticity (equal variance of errors).
# Purpose: Check if errors are randomly distributed with constant variance.
sns.residplot(x=pred, y=cars.MPG, lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('Fitted vs Residuals')
plt.show()

# Influence plot to check the impact of influential points.
sm.graphics.influence_plot(final_ml)
plt.show()

# Splitting the data into training and test sets
# Purpose: Train the model on a portion of data (train), then evaluate it on unseen data (test) to assess model generalization.
cars_train, cars_test = train_test_split(cars, test_size=0.2, random_state=42)

# Preparing the model on the training data.
model_train = smf.ols('MPG ~ VOL + SP + HP', data=cars_train).fit()
print(model_train.summary())

# Predicting on test data.
test_pred = model_train.predict(cars_test)

# Calculating the test error (difference between actual and predicted values).
test_error = cars_test.MPG - test_pred

# Calculating RMSE (Root Mean Squared Error) for the test set to measure model performance.
test_rmse = np.sqrt(np.mean(test_error ** 2))
print("Test RMSE:", test_rmse)

# Calculating RMSE for the training set (optional).
train_pred = model_train.predict(cars_train)
train_error = cars_train.MPG - train_pred
train_rmse = np.sqrt(np.mean(train_error ** 2))
print("Train RMSE:", train_rmse)
