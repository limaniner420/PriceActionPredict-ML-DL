import math
from tkinter.tix import X_REGION
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pd.options.mode.chained_assignment = None  # default='warn'

try:
    file = pd.read_json("real data.json")
    data = pd.DataFrame(file['GOOG']['chart'])    
    weight = 0
except Exception as e:
    print("The stock data haven't fetched, please fetch first in the fetch.py")
    exit()

def SVM(data: pd.DataFrame):

    # data.index = pd.to_datetime(data['date'])

    data['priceDate'] = (pd.to_datetime(data['priceDate']) - dt.datetime(1970,1,1)).dt.total_seconds()
    
    ipt = 360

    TestData = data.tail(ipt)
    df = data.head(len(data) - ipt)

    days = list()
    close_price = list()

    Testdays = list()
    Testclose_price = list()

    df_days = df.loc[:, 'priceDate'] 
    df_close = df.loc[:,'close']

    df_testdays = TestData.loc[:, 'priceDate'] 
    df_testclose = TestData.loc[:,'close']

    # independent dataset
    for day in df_days:
        #days.append([int(day.split('-')[2])])
        days.append([float(day)])

    #dependent dataset
    for close in df_close:
        close_price.append(float(close))


    for day in df_testdays:
        Testdays.append([float(day)])

    for close in df_testclose:
        Testclose_price.append(float(close))

    
    model = SVR(kernel='rbf',C=150) # gamma =0.000000000000001
    # model2 = SVR(kernel='linear', C=1e3)
    # model3 = SVR(kernel='poly', C=1e3, degree=2)

    model.fit(days, close_price)
    # model2.fit(days, close_price)
    # model3.fit(days, close_price)

    # accuracyRBF = model.score(x_test, y_test)
    # print("Accuracy RBF:", accuracyRBF)

    # accuracyLinear = model2.score(x_test, y_test)
    # print("Accuracy Linear:", accuracyLinear)

    # accuracyPoly = model3.score(x_test, y_test)
    # print("Accuracy Poly:", accuracyPoly)

    # temp = pd.DataFrame
    # temp = data[-360:]

    # forecast = np.array(data.drop(['prediction'],1))[-360:]

    # a = model.predict(forecast)
    # b = model2.predict(forecast)
    # c = model3.predict(forecast)
    # temp['predictRBF'] = a
    # temp['predictLinear'] = b
    # temp['predictPoly'] = c
    # #print(temp)

    # # x_test['predictRBF'] = a #maybe do cumulative sum to see overall improvement
    # # x_test['predictLinear'] = b
    # # x_test['predictPoly'] = c

    # #x_test['chgPer_cum'] = x_test['changePercent'].cumsum()

    # # x_test = x_test.sort_values(by='date')

    # a = model.predict(y_train.reshape(-1, 1))
    
    y_train = np.array(TestData['close'])
    a = model.predict(np.array(df_testdays).reshape(-1, 1))

    accuracyRBF = model.score(np.array(df_testdays).reshape(-1, 1), a)
    # print("Accuracy RBF:", accuracyRBF)

    print('r2_score RBF: ',r2_score(y_train, a))
    print('MAE RBF: ', mean_absolute_error(y_train, a))
    print('MSE RBF: ', mean_squared_error(y_train, a,squared=True))
    print('rmse RBF: ',mean_squared_error(y_train, a,squared=False), '\n')

    # b = model2.predict(y_train.reshape(-1, 1))
    # print('r2_score Linear: ',r2_score(y_train, b))
    # print('MAE Linear: ', mean_absolute_error(y_train, b))
    # print('MSE Linear: ', mean_squared_error(y_train, b,squared=True))
    # print('rmse Linear: ',mean_squared_error(y_train, b,squared=False) , '\n')

    # c = model3.predict(y_train.reshape(-1, 1))
    # print('r2_score Poly: ',r2_score(y_train, c))
    # print('MAE Poly: ', mean_absolute_error(y_train, c))
    # print('MSE Poly: ', mean_squared_error(y_train, c,squared=True))
    # print('rmse Poly: ',mean_squared_error(y_train, c,squared=False))

    #TO check value of individual day
    # print(df_testdays)
    # print(df_testclose)

    index = 1000 #899 to 1258

    print('predicted:', model.predict([[df_testdays.loc[index]]]))
    print('actual:', df_testclose.loc[index])

    # df_days = df.loc[:, 'priceDate'] 
    # df_close = df.loc[:,'close']

    # print(df_days)
    # print(df_close)

    # index2 = 898 # 0 to 898

    # print('predicted:', model.predict([[df_days.loc[index2]]]))
    # print('actual:', df_close.loc[index2])

    plt.style.use('fivethirtyeight')
    plt.figure(figsize = (16,9))
    plt.plot(days, close_price, color = 'red', label = 'Data')
    plt.plot(days, model.predict(days), color='black', label='train RBF')
    # plt.plot(days, model2.predict(days), color='green', label='linear')
    # plt.plot(days, model3.predict(days), color='blue', label='poly')

    #Test data for forecast
    plt.plot(Testdays, Testclose_price, color = 'red')
    plt.plot(Testdays, model.predict(Testdays), color='blue', label='predicted RBF')
    # plt.plot(days, model2.predict(days), color='green', label='linear')
    # plt.plot(days, model3.predict(days), color='blue', label='poly')
    plt.title('SVM')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()



SVM(data)
