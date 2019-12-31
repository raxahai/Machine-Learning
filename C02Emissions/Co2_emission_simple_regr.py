import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
import pandas as pd 
import numpy as np 
from sklearn import linear_model

df = pd.read_csv('FuelConsumption.csv',encoding= "ISO-8859-1")
# df = df.dropna(axis = 1, how = 'all')
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

plt.scatter(df['ENGINESIZE'],df["CO2EMISSIONS"],color = "blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print(f"the predicted value is: {test_y_hat[0][0]}")
print(f"the actual test value is: {test_y[0][0]}")
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )