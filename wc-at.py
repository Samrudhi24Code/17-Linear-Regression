# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:55:52 2025

@author: Dell
"""

import pandas as pd
import numpy as np
import seaborn as sns
wcat = pd.read_csv(r"E:\Honars(DS)\Data Science\17-Linear Regression\wc-at.csv")

#EDA
wcat.info()
wcat.describe()

#Average waist is 91.90 and min is 63.50 and max is 121

import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
sns.displot(wcat.AT)
#Data is normal but right shewed

plt.boxplot(wcat.AT)
#No outliers but right skewed

plt.bar(height=wcat.Waist,x=np.arange(1,110,1))
sns.displot(wcat.Waist)
#Data is normal bimodel
plt.boxplot(wcat.Waist)
#No outliers but right skewed

######################################
#bivariant analysis
plt.scatter(x=wcat.Waist,y=wcat.AT)
#Data is lineary scattered,direction positive,strenght:poor
#Now let us check the correlation coefficent

np.corrcoef(wcat.Waist,wcat.AT)

cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
#635.9100064135235 is postive means correlation will be positive

###############################################

'''Let us check fesability of model model'''

import statsmodels.formula.api as smf

#First sample linear model
model=smf.ols('AT~Waist',data=wcat).fit()
#Y is AT and X is waist
model.summary()
#R-sqaured=0.67<0.85 there is scope of improvement
#p=00<0.05 hence acceptable
#bita-0=-215.98
#bita-1=3.45

pred1=model.predict(pd.DataFrame(wcat.Waist))
pred1
######################

#Regression Line
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.legend(['Predicted line','Observed Data'])
plt.show()
##############################

#Error Calculations

# Error Calculations
res1 = wcat.AT - pred1  # Residuals
mse1 = np.mean(res1**2)  # Mean Squared Error
rmse1 = np.sqrt(mse1)  # Root Mean Squared Error

print("Mean Squared Error:", mse1)
print("Root Mean Squared Error:", rmse1)

################################################
#Let us try for another model

plt.scatter(x=np.log(wcat.Waist),y=wcat.AT)
#Data is linerly scatter ,direction positive,strenght:poor
#Now let us check the correlation
np.corrcoef(np.log(wcat.Waist),wcat.AT)

model2= smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()

pred2=model2.predict(pd.DataFrame(wcat.Waist))
pred2
############################
#Regression Line

plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend(['Predicted line','Observed Data2'])
plt.show()

#########################
#Error Calculation

res2=wcat.AT-pred2
np.mean(res1)
res_sqr2=res2*res2

mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#########################################

#Let us try for another model

plt.scatter(x=np.log(wcat.Waist),y=wcat.AT)
#Data is linerly scatter ,direction positive,strenght:poor
#Now let us check the correlation
np.corrcoef(np.log(wcat.Waist),wcat.AT)

model2= smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()

pred2=model2.predict(pd.DataFrame(wcat.Waist))
pred2
############################
#Regression Line

plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend(['Predicted line','Observed Data2'])
plt.show()

#########################
#Error Calculation

res2=wcat.AT-pred2
np.mean(res1)
res_sqr2=res2*res2

mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#########################################


#Let us make logY and X as is
plt.scatter(x=(wcat.Waist),y=np.log(wcat.AT))

#Data is linearly scatter direction is postive,negative,strenght :poor


#Now let us check for coffient correlation
np.corrcoef(wcat.Waist,np.log(wcat.AT))

model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()

model3.summary()

pred3=model3.predict(pd.DataFrame(wcat.Waist))
pred3_at=np.exp(pred3)
pred3_at
########################

#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend(['Predicted Line','Observed data_model3'])
plt.show()
##############################

#Error Calculation
res3=wcat.AT-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#38.52900175807143 there is chnge in r


###############################
#Try another model

#Now let us make Y=log(AT) and X=Waist,X*X=Waist.Waist

#polynomial model
#Here r can not be calculated

model4 = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data=wcat).fit()

#Y is log(AT)and X=Waist
model4.summary()

#R-Squared=0.779<0.85 there is scope of improvement
#p=0.0000<0.05 hence acceptable
#Bita-0=7.8241
#Bita-1=0.2239


pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4

pred4_at=np.exp(pred4)
pred4_at
#############################################
#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted Line','Observed data_model4'])
plt.show()
##############################
#Error Calculation
res4=wcat.AT-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#Best model
# Collect metrics for each model
metrics = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3", "Model 4"],
    "R-Squared": [model.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
    "RMSE": [rmse1, rmse2, rmse3, rmse4]
})
print(metrics)
#######################################
#Now we have to generalize the best model

from sklearn.model_selection import train_test_split

train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))

plt.scatter(test.Waist,np.log(test.AT))

final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()

#here y is log(AT) and X=Waist

final_model.summary()
#R-squared:  0.779 there is scope of improvement
#p=0.000<0.05 hence acceptable

test_pred=final_model.predict(pd.DataFrame(test))

test_pred_at=np.exp(test_pred)
test_pred_at

train_pred=final_model.predict(pd.DataFrame(train))

train_pred_at=np.exp(train_pred)
train_pred_at

#Evalution of test model

test_res=test.AT-test_pred_at
test_sqr=test_res*test_res
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#21.535099566566036

##########################################
train_res=train.AT-train_pred_at
train_sqr=train_res*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#34.42884550089754













































