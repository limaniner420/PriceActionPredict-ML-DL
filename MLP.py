from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data.preprocess as pre

SET_ESS = ["date", "close", "high", "low", "open", "symbol", "volume", "change", "changePercent"]

file = pd.read_json("data.json")
data = pd.DataFrame(file['GOOG']['chart'])[SET_ESS]
pre.apply_indicators(data)

data = data.drop(["high", "low", "open", "symbol", "volume", "change", "changePercent"], axis = 1).rename(columns = {"close" : "t"})

# for i in range(1, 1):
#     label = "t-" + str(i)
#     data[label] = data["t"].shift(i)
data["t+1"] = data["t"].shift(-1)

data.dropna(inplace = True)
data.set_index("date",inplace = True)

X_train, X_test, y_train, y_test = train_test_split(data.drop(["t+1"], axis = 1), data["t+1"], test_size=0.25, shuffle=False)

pipeline = make_pipeline(StandardScaler(), mlp.MLPRegressor(max_iter = 1000, shuffle = False))
pipeline.fit(X_train, y_train)
p = pipeline.predict(X_test)

print('r2_score: ',r2_score(y_test, p))
print('MAE: ', mean_absolute_error(y_test, p))
print('MSE: ', mean_squared_error(y_test, p, squared=True))
print("MAPE: ", mean_absolute_percentage_error(y_test, p))
print('RMSE: ', mean_squared_error(y_test, p, squared=False))

plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Closing Price', fontsize = 16)
plt.xticks(np.arange(0, len(data.values)+1,37),rotation=70)
plt.plot(y_test, linewidth=1)
plt.plot(p, linewidth=1)
plt.legend(['Actual', 'Predictions'], loc = 'lower right')
plt.savefig("predictions")
plt.show()
