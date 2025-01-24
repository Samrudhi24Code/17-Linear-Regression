# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:55:52 2025

@author: Dell

Problem Statement:
This dataset contains information about the Waist circumference and the AT (some attribute) for a set of individuals. 
The goal of this analysis is to predict AT (dependent variable) based on Waist (independent variable). 
We aim to build multiple linear regression models to find the best fit for predicting AT based on Waist. 
Various transformations of the data, including log transformations, are used to explore different relationships.
The models are evaluated based on metrics such as R-squared and Root Mean Squared Error (RMSE).

Business Goals:
1. To develop an accurate predictive model for AT (dependent variable) using Waist (independent variable).
2. To assess which model provides the best predictive power for AT.
3. To determine the feasibility of using transformations (logarithmic and polynomial) to improve model accuracy.
4. To assess the performance of the final selected model in predicting both the training and test data.

The analysis can help understand how Waist circumference influences the AT attribute, which could have applications 
in health and fitness, specifically in understanding body measurements and their correlation with other physical attributes.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
wcat = pd.read_csv(r"E:\Honars(DS)\Data Science\17-Linear Regression\wc-at.csv")

# EDA (Exploratory Data Analysis)
wcat.info()  # Information about the dataset
wcat.describe()  # Summary statistics for the dataset

# Visualizing Waist and AT data
plt.bar(height=wcat.AT, x=np.arange(1, 110, 1))
sns.displot(wcat.AT)
# Data is normal but right skewed

plt.boxplot(wcat.AT)
# No outliers but right skewed

plt.bar(height=wcat.Waist, x=np.arange(1, 110, 1))
sns.displot(wcat.Waist)
# Data is normal bimodal
plt.boxplot(wcat.Waist)
# No outliers but right skewed

######################################
# Bivariate analysis: Waist vs AT
plt.scatter(x=wcat.Waist, y=wcat.AT)
# Data is linearly scattered, direction positive, strength: poor
# Now, let us check the correlation coefficient
np.corrcoef(wcat.Waist, wcat.AT)

cov_output = np.cov(wcat.Waist, wcat.AT)[0, 1]
cov_output
# Positive correlation, means correlation will be positive

###############################################

'''Let us check feasibility of model'''

import statsmodels.formula.api as smf

# First simple linear model
model = smf.ols('AT ~ Waist', data=wcat).fit()
# Y is AT and X is Waist
model.summary()
# R-squared=0.67 < 0.85, there is scope for improvement
# p-value=0.000 < 0.05, hence acceptable
# Coefficients: Intercept=-215.98, Slope=3.45

pred1 = model.predict(pd.DataFrame(wcat.Waist))
pred1
######################

# Regression Line
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred1, 'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()
##############################

# Error Calculations
res1 = wcat.AT - pred1  # Residuals
mse1 = np.mean(res1**2)  # Mean Squared Error
rmse1 = np.sqrt(mse1)  # Root Mean Squared Error

print("Mean Squared Error:", mse1)
print("Root Mean Squared Error:", rmse1)

################################################
# Let us try another model with log transformation

plt.scatter(x=np.log(wcat.Waist), y=wcat.AT)
# Data is linearly scattered, direction positive, strength: poor
# Now let us check the correlation
np.corrcoef(np.log(wcat.Waist), wcat.AT)

model2 = smf.ols('AT ~ np.log(Waist)', data=wcat).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(wcat.Waist))
pred2
############################
# Regression Line

plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred2, 'r')
plt.legend(['Predicted line', 'Observed Data2'])
plt.show()

#########################
# Error Calculation

res2 = wcat.AT - pred2
np.mean(res1)
res_sqr2 = res2 * res2

mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#########################################

# Let us try another model

# Data is linearly scattered, direction positive, strength: poor
# Now let us check the correlation
np.corrcoef(wcat.Waist, np.log(wcat.AT))

model3 = smf.ols('np.log(AT) ~ Waist', data=wcat).fit()

model3.summary()

pred3 = model3.predict(pd.DataFrame(wcat.Waist))
pred3_at = np.exp(pred3)
pred3_at
########################

# Regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, 'r')
plt.legend(['Predicted Line', 'Observed data_model3'])
plt.show()
##############################

# Error Calculation
res3 = wcat.AT - pred3_at

res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

###############################
# Try another model with polynomial terms

model4 = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data=wcat).fit()

# Y is log(AT) and X=Waist
model4.summary()

# R-Squared=0.779 < 0.85, there is scope for improvement
# p-value=0.0000 < 0.05, hence acceptable

pred4 = model4.predict(pd.DataFrame(wcat.Waist))
pred4

pred4_at = np.exp(pred4)
pred4_at
#############################################
# Regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred4, 'r')
plt.legend(['Predicted Line', 'Observed data_model4'])
plt.show()
##############################
# Error Calculation
res4 = wcat.AT - pred4_at

res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Best model
# Collect metrics for each model
metrics = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3", "Model 4"],
    "R-Squared": [model.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
    "RMSE": [rmse1, rmse2, rmse3, rmse4]
})
print(metrics)

#######################################
# Now we have to generalize the best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(wcat, test_size=0.2)
plt.scatter(train.Waist, np.log(train.AT))
plt.scatter(test.Waist, np.log(test.AT))

final_model = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data=wcat).fit()

final_model.summary()

# R-squared: 0.779, there is scope for improvement
# p-value=0.000 < 0.05, hence acceptable

test_pred = final_model.predict(pd.DataFrame(test))
test_pred_at = np.exp(test_pred)
test_pred_at

train_pred = final_model.predict(pd.DataFrame(train))
train_pred_at = np.exp(train_pred)
train_pred_at

# Evaluation of test model
test_res = test.AT - test_pred_at
test_sqr = test_res * test_res
test_mse = np.mean(test_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse
# RMSE for test data

##########################################
train_res = train.AT - train_pred_at
train_sqr = train_res * train_res
train_mse = np.mean(train_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse
# RMSE for train data
