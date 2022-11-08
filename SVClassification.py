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
    file = pd.read_json("data.json")
    data = pd.DataFrame(file['MSFT']['chart'])    
    weight = 0
except Exception as e:
    print("The stock data haven't fetched, please fetch first in the fetch.py")
    exit()

def SVM(data: pd.DataFrame): # y = price up or down based on pervious day, X = independent avr for prediction such as closing price

    data.index = pd.to_datetime(data['date'])
    data = data[['date','open', 'high', 'low', 'close', 'changePercent']]
    data['OpenClose'] = data['open'] - data['close']
    data['HighLow'] = data['high'] - data['low']
    pd.set_option('display.max_rows', None)

    X = data[['OpenClose', 'HighLow','close']]

    y = np.where(data['close'].shift(-1) > data['close'], 1, 0) #Predicted outcome

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.145)

    model = SVC(kernel='rbf', C=1e3, gamma='scale')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracyRBF = accuracy_score(y_test, y_pred)
    print("Accuracy RBF:", accuracyRBF)

    a = model.predict(X)

    data['Predicted_Signal'] = a #maybe do cumulative sum to see overall improvement
    data['Return'] = data.close.pct_change()
    data['Strategy_Return'] = data.Return *data.Predicted_Signal.shift(1)
    data['Cum_Ret'] = data['Return'].cumsum()
    data['Cum_Strategy'] = data['Strategy_Return'].cumsum()

    x_test = x_test.sort_values(by='date')

    plt.style.use('fivethirtyeight')
    plt.figure(figsize = (12,9))
    plt.plot(data['Cum_Ret'],color='red')
    plt.plot(data['Cum_Strategy'],color='blue')
    plt.title('SVM')
    plt.ylabel('Change Percent')
    plt.xlabel('Date')
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()

SVM(data)
