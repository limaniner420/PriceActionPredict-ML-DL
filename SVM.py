import math
from tkinter.tix import X_REGION
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
pd.options.mode.chained_assignment = None  # default='warn'

try:
    file = pd.read_json("real data.json")
    data = pd.DataFrame(file['GOOG']['chart'])    
    weight = 0
except Exception as e:
    print("The stock data haven't fetched, please fetch first in the fetch.py")
    exit()

def SVM(data: pd.DataFrame):

    data.index = pd.to_datetime(data['date'])
    #data = data.drop(['date'], axis = 1)
    data = data[['date','open', 'high', 'low', 'close', 'changePercent']]
    data['OpenClose'] = data['open'] - data['close']
    data['HighLow'] = data['high'] - data['low']
    pd.set_option('display.max_rows', None)

    #print(data)
    # data_target = data.filter(['changePercent'])
    # target = data_target.values
    # training_data_len = math.ceil(len(target)* 0.855) # training set has 75% of the data
    # training_data_len
    # sc = StandardScaler()
    # training_scaled_data = sc.fit_transform(target)

    #X = data[['changePercent']]
    X = data[['OpenClose', 'HighLow', 'changePercent', 'close']]
    #X = data[['OpenClose', 'HighLow']] #Actual value
    #X = training_scaled_data
    #X = data[['changePercent']]
    #X = X[:-30] #first 270 days
    #print(X)

    y = np.where(data['changePercent'] >= 0, 1, -1) #Predicted outcome
    #data['prediction'] = training_scaled_data
    #y = np.where(data['close'].shift(-1) > data['close'], 1, 0) #Predicted outcome
    #y = data['prediction']
    #y = np.array(data['prediction'].shift(-30))
    #y = y[:-30]
    #print(y)

    # for i in range(0,len(X)):
    #     print(X[i], y[i])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.145)

    model = SVR(kernel='rbf', C=1e3, gamma='scale')
    model2 = SVR(kernel='linear', C=1e3)
    model3 = SVR(kernel='poly', C=1e3, degree=2)

    model.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)

    accuracyRBF = model.score(x_test, y_test)
    print("Accuracy RBF:", accuracyRBF)

    accuracyLinear = model2.score(x_test, y_test)
    print("Accuracy Linear:", accuracyLinear)

    accuracyPoly = model3.score(x_test, y_test)
    print("Accuracy Poly:", accuracyPoly)

    a = model.predict(x_test)
    b = model2.predict(x_test)
    c = model3.predict(x_test)

    x_test['predictRBF'] = a #maybe do cumulative sum to see overall improvement
    x_test['predictLinear'] = b
    x_test['predictPoly'] = c

    #x_test['chgPer_cum'] = x_test['changePercent'].cumsum()

    x_test = x_test.sort_values(by='date')

    print(x_test)

    plt.style.use('fivethirtyeight')
    plt.figure(figsize = (12,9))
    plt.plot(x_test.index, x_test['changePercent'] * 10, color='black', label='Data')
    plt.plot(x_test.index, x_test['predictRBF'], color='red', label='rbf')
    plt.plot(x_test.index, x_test['predictLinear'], color='green', label='linear')
    plt.plot(x_test.index, x_test['predictPoly'], color='blue', label='poly')
    plt.title('SVM')
    plt.ylabel('Change Percent')
    plt.xlabel('Date')
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()

SVM(data)
