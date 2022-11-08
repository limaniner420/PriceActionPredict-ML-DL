import math
from tkinter.tix import X_REGION
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import data.preprocess as pre

pd.options.mode.chained_assignment = None  # default='warn'

try:
    file = pd.read_json("data.json")
    data = pd.DataFrame(file['MSFT']['chart'])    
    weight = 0
except Exception as e:
    print("The stock data haven't fetched, please fetch first in the fetch.py")
    exit()

def SVM(data: pd.DataFrame): #predict future price using current price

    pre.apply_indicators(data)

    forecast = 360 #number of data point for forecast
    data.index = pd.to_datetime(data['date'])
    data = data[['close']] # without indicator
    #data = data[['close', 'ma_cross', 'macd_cross', 'rsi', 'volatility', 'K', 'D', 'ma_long', 'ma_short']] # with indicator
    data.dropna(inplace = True)
    print(data)
    # TestData = data.tail(forecast) #last 360
    # RealData = data.head(len(data) - forecast) # all in front of 360

    ipt = 1
    data['prediction'] = data[['close']].shift(-ipt)
    # RealData['prediction'] = RealData[['close']].shift(-ipt)
    # TestData['Actual_prediction'] = TestData[['close']].shift(-ipt)

    X = np.array(data.drop(['prediction'], axis=1)) #t
    X = X[:-ipt]
  
    y = np.array(data['prediction']) #t + 1
    y = y[:-ipt]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, shuffle =False)

    # epsilon = width of the tube around the hyperplane. 
    # C = how much we care about the error. Larger number means better fit but risk overfitting.
    # gamma = how much influence a single training example has

    model = SVR(kernel='rbf', C = 1e3)
    model.fit(x_train, y_train)
    a = model.predict(x_test)
    print('r2_score RBF: ',r2_score(a, y_test))
    print('MAE RBF: ', mean_absolute_error(a, y_test))
    print('MSE RBF: ', mean_squared_error(a, y_test))
    print("MAPE RBF: ", mean_absolute_percentage_error(a, y_test))
    print('rmse RBF: ',mean_squared_error(a, y_test), '\n')


    # model2 = SVR(kernel='linear', C=1e3)
    # model2.fit(x_train, y_train)
    # b = model2.predict(x_test)
    # print('r2_score Linear: ',r2_score(b, y_test))
    # print('MAE Linear: ', mean_absolute_error(b, y_test))
    # print('MSE Linear: ', mean_squared_error(b, y_test,squared=True))
    # print("MAPE RBF: ", mean_absolute_percentage_error(b, y_test))
    # print('rmse Linear: ',mean_squared_error(b, y_test,squared=False) , '\n')


    # model3 = SVR(kernel='poly', C=1e3, degree=2)      
    # model3.fit(x_train, y_train)
    # c = model3.predict(x_test)
    # print('r2_score Poly: ',r2_score(c, y_test))
    # print('MAE Poly: ', mean_absolute_error(c, y_test))
    # print('MSE Poly: ', mean_squared_error(c, y_test,squared=True))
    # print("MAPE RBF: ", mean_absolute_percentage_error(c, y_test))
    # print('rmse Poly: ',mean_squared_error(c, y_test,squared=False))

    plt.style.use('fivethirtyeight')
    plt.figure(figsize = (20,10))
    plt.plot(a, color='blue', label='Prediction rbf', linewidth = 1)
    # plt.plot(b, color='green', label='Prediction linear', linewidth = 1)
    # plt.plot(c, color='red', label='Prediction poly', linewidth = 1)
    plt.plot(y_test, color='black', label='Data', linewidth = 1)
    plt.title('SVM')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.xticks(rotation=70)
    plt.legend()
    plt.savefig("SVR")
    plt.show()

SVM(data)
